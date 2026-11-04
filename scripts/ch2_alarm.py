#!/usr/bin/env python3
"""
scripts/ch2_alarm.py — can channel 2 warn a client that an attack has started?
==============================================================================
An attack starts part way through federation, on one organisation, with production
option A aggregation. Each client draws fresh flows from its own pool every round,
because a fixed sample would make any change-based signal look unrealistically clean.

Signals, each per client per round:
  ch2     signed cosine of the head update against the root head update (server side)
  local   the client's own share of high-reconstruction-error flows (no server needed)
  ch1     signed cosine of the AE update against the root AE update (server side)
  fixed   production's UNDER_ATTACK rule, ch2 > CH2_UNDER_ATTACK_THRESHOLD

The change alarm fires when a signal moves away from that client's previous HISTORY
rounds by more than a limit, set at the 95th percentile of clean runs on separate seeds.

Fixed before any result was seen. Channel 2 passes on a dataset only if, pooled over
the organisations tested:
  false alarms   at most 5% per client per round on clean runs
  detection      at least 80% of attacks caught within 3 rounds at a 20% attack share
  beats local    detection at 20% at least 0.05 above the client's own signal
It has to pass on both datasets.

Usage:
    python scripts/ch2_alarm.py --dataset unsw
    python scripts/ch2_alarm.py --dataset ton
    python scripts/ch2_alarm.py --dataset unsw --quick      # wiring check, writes no results
"""

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import config as cfg
from aura import dataset as ds, fl_round
from aura.fl_server import dc_fltrust_aggregate
from aura.models import AttackHead, FlowAutoencoder
import ch2_conditions as c
import ch2_cross_dataset as x

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("ch2_alarm")

ROUNDS, ONSET, HISTORY, WINDOW = 16, 11, 5, 3      # ONSET is 1-based; WINDOW rounds count as caught in time
BURDENS = [0.05, 0.10, 0.20, 0.50]
SIGNALS = ["ch2", "local", "ch1"]
CALIB_SEEDS = [c.CALIB_SEED, c.CALIB_SEED + 1, c.CALIB_SEED + 2]


def pools_of(data):
    """Each organisation's training flows. The attack-free organisation keeps its benign flows only."""
    bank = ds.ORG_NAMES.index("bank")
    pools = []
    for o in range(len(ds.ORG_NAMES)):
        keep = data.train["o"] == o
        if o == bank:
            keep &= data.train["y"] == 0
        pools.append(dict(x=data.train["x"][keep], y=data.train["y"][keep]))
    return pools


def federate(data, pools, attack_x, target, burden, seed, taus):
    """Returns rounds x clients arrays of every signal."""
    torch.manual_seed(cfg.SEED + seed)
    rng = np.random.default_rng(cfg.SEED + seed)
    g = copy.deepcopy(data.g0)
    g_ae, g_head = fl_round.snapshot(g.autoencoder), fl_round.snapshot(g.attack_head)
    root_benign = torch.from_numpy(data.root["x"][data.root["y"] == 0])
    root_attack = torch.from_numpy(data.root["x"][data.root["y"] == 1])
    ae, head = FlowAutoencoder(), AttackHead()
    out = {s: np.full((ROUNDS, len(pools)), np.nan) for s in SIGNALS}
    excluded = np.zeros((ROUNDS, len(pools)))

    saved = cfg.CH2_ANTI_ALIGN_THRESHOLD
    cfg.CH2_ANTI_ALIGN_THRESHOLD = taus["ch2"] if taus else -2.0
    try:
        for rnd in range(ROUNDS):
            root_ae, root_head = fl_round.root_reference(g_ae, g_head, root_benign, root_attack)
            ae_ds, head_ds = [], []
            for i, pool in enumerate(pools):
                pick = rng.integers(len(pool["y"]), size=c.CLIENT_FLOWS)
                flows, y = pool["x"][pick].copy(), pool["y"][pick]
                if i == target and rnd + 1 >= ONSET and burden > 0:
                    benign = np.flatnonzero(y == 0)
                    swap = benign[:int(len(flows) * burden)]
                    flows[swap] = attack_x[rng.integers(len(attack_x), size=len(swap))]
                u = fl_round.client_update(ae, head, g_ae, g_head, torch.from_numpy(flows))
                ae_ds.append(u.ae_delta)
                head_ds.append(u.head_delta)
                out["local"][rnd, i] = u.n_high_mse / max(u.n_benign + u.n_high_mse, 1)
            agg_ae, agg_head, _, _, _, flags = dc_fltrust_aggregate(
                ae_ds, head_ds, root_ae, root_head, [rnd + 1] * len(pools), ch2_warmup_rounds=0,
                current_round=rnd + 1, ch1_threshold=taus["ch1"] if taus else -1.0)
            out["ch2"][rnd] = c._ch2_raw(head_ds, root_head)
            out["ch1"][rnd] = c._ch2_raw(ae_ds, root_ae)
            excluded[rnd] = flags
            g_ae, g_head = fl_round.apply_delta(g_ae, agg_ae), fl_round.apply_delta(g_head, agg_head)
    finally:
        cfg.CH2_ANTI_ALIGN_THRESHOLD = saved
    return out, excluded


def change_scores(series):
    """|score - mean of the previous HISTORY rounds| in standard deviations, rounds x clients."""
    z = np.full(series.shape, np.nan)
    for t in range(HISTORY, len(series)):
        past = series[t - HISTORY:t]
        z[t] = np.abs(series[t] - np.nanmean(past, axis=0)) / np.maximum(np.nanstd(past, axis=0), 1e-6)
    return z


def run(data, seeds, targets, burdens):
    pools = pools_of(data)
    attack_x = data.train["x"][data.train["y"] == 1]
    shares = [float(p["y"].mean()) for p in pools]
    log.info(f"attack share by org {dict(zip(ds.ORG_NAMES, np.round(shares, 3)))}; flows {[len(p['y']) for p in pools]}")
    t0 = time.time()

    first, _ = federate(data, pools, attack_x, 0, 0.0, CALIB_SEEDS[0], None)
    taus = {"ch1": float(np.percentile(first["ch1"], 5)), "ch2": float(np.nanpercentile(first["ch2"], 5))}
    calib = [federate(data, pools, attack_x, 0, 0.0, s, taus)[0] for s in CALIB_SEEDS[:len(seeds)]]
    limits = {s: float(np.nanpercentile(np.concatenate([change_scores(r[s]).ravel() for r in calib]), 95)) for s in SIGNALS}
    log.info(f"taus {taus}; limits {limits} ({time.time() - t0:.0f}s)")

    clean = [federate(data, pools, attack_x, 0, 0.0, s, taus)[0] for s in seeds]
    false_alarm = {s: [float(np.nanmean(change_scores(r[s])[HISTORY:] > limits[s])) for r in clean] for s in SIGNALS}
    fixed_clean = [float(np.nanmean(r["ch2"][HISTORY:] > cfg.CH2_UNDER_ATTACK_THRESHOLD)) for r in clean]
    res = {"targets": {ds.ORG_NAMES[t]: round(shares[t], 3) for t in targets}, "taus": taus, "limits": limits,
           "clean_means": {s: {ds.ORG_NAMES[i]: round(float(np.nanmean([r[s][:, i] for r in clean])), 4) for i in range(len(pools))} for s in SIGNALS},
           "false_alarm": {s: {"mean": float(np.mean(v)), "std": float(np.std(v))} for s, v in false_alarm.items()},
           "fixed_rule_clean_fire_rate": float(np.mean(fixed_clean)), "runs": {}}

    start, end = ONSET - 1, ONSET - 1 + WINDOW
    for t in targets:
        for b in burdens:
            rows = []
            for seed in seeds:
                r, excl = federate(data, pools, attack_x, t, b, seed, taus)
                row = {"seed": seed, "fixed": bool(np.nan_to_num(r["ch2"][start:end, t]) .max() > cfg.CH2_UNDER_ATTACK_THRESHOLD),
                       "excluded_after_onset": float(excl[start:, t].mean())}
                others = [i for i in range(len(pools)) if i != t]
                for s in SIGNALS:
                    z = change_scores(r[s])
                    row[s] = bool(np.nan_to_num(z[start:end, t]).max() > limits[s])
                    row[f"{s}_first_round"] = bool(np.nan_to_num(z[start, t]) > limits[s])
                    row[f"{s}_others_alarm"] = float(np.nanmean(z[start:, others] > limits[s]))
                    row[f"{s}_shift"] = float(np.nanmean(r[s][start:end, t]) - np.nanmean(r[s][start - HISTORY:start, t]))
                rows.append(row)
            res["runs"][f"{ds.ORG_NAMES[t]}|{b}"] = rows
            log.info(f"{ds.ORG_NAMES[t]} burden={b}: {rows} ({time.time() - t0:.0f}s)")
    return summarise(res, burdens)


def summarise(res, burdens):
    res["detection"] = {}
    for b in burdens:
        rows = [r for k, v in res["runs"].items() if k.endswith(f"|{b}") for r in v]
        res["detection"][str(b)] = {s: float(np.mean([r[s] for r in rows])) for s in SIGNALS + ["fixed"]}
    d = res["detection"].get("0.2")
    if d:
        res["verdict"] = {"false_alarm_ok": res["false_alarm"]["ch2"]["mean"] <= 0.05, "detection_ok": d["ch2"] >= 0.80,
                          "beats_local": d["ch2"] - d["local"] >= 0.05}
        res["verdict"]["pass"] = all(res["verdict"].values())
    return res


def main():
    global ROUNDS, ONSET
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["unsw", "ton"], required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        ROUNDS, ONSET, args.seeds = 8, 7, 1
        c.QUICK = True
    torch.set_num_threads(4)
    if args.dataset == "ton":
        x.build_sample()
        data = x.ToNData(keep_dups=True)
        cfg.CH2_MSE_SPLIT_THRESHOLD = data.split_threshold
        out_file = ROOT / "results" / "ton_keep_dups" / "ch2_alarm.json"
    else:
        data = c.Data()
        out_file = ROOT / "results" / "ch2_alarm.json"

    bank = ds.ORG_NAMES.index("bank")
    share = [float(k["y"].mean()) for k in data.clients]
    attacked = sorted((i for i in range(len(share)) if i != bank), key=lambda i: share[i])
    targets = [bank, attacked[0], attacked[-1]]
    res = run(data, list(range(args.seeds)), targets[:1] if args.quick else targets, [0.2] if args.quick else BURDENS)
    log.info(json.dumps({k: res.get(k) for k in ("false_alarm", "fixed_rule_clean_fire_rate", "detection", "verdict")}))
    if not args.quick:
        out_file.write_text(json.dumps(res, indent=1), encoding="utf-8")
    log.info("DONE")


if __name__ == "__main__":
    main()
