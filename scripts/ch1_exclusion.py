#!/usr/bin/env python3
"""
scripts/ch1_exclusion.py — does channel 1 cost the organisations it excludes?
=============================================================================
With the P99 routing cut, DC-FLTrust drops retail's AE update in 32% of clean rounds and
bank's in 87% when bank carries 50% attacks. This scores every organisation separately on
the calibration holdout, to see whether an excluded organisation pays for it, and tests
whether attack flows in its training are the reason it is excluded.

AE only, channel 1 through the real dc_fltrust_aggregate, clients stopped at the export step.

  trust                   DC-FLTrust at the configured channel 1 threshold
  fedavg                  every update counted equally
  trust_no_exclusion      trust weights only, no threshold
  trust_retail_benign     label-oracle: retail trains on its benign flows only
  local                   each organisation trains alone from G0, scored on its own traffic
  trust|bank50, fedavg|bank50   bank with 50% attack traffic

Criteria, fixed before running (means over seeds, retail, trust against fedavg):
  exclusion costs retail if its FPR is >= 0.3 points higher or its recall >= 0.03 lower
  attack flows explain the exclusion if trust_retail_benign excludes retail in <= 10% of rounds

Scored on the calibration holdout. The test split stays sealed.

Usage:
    python scripts/ch1_exclusion.py
    python scripts/ch1_exclusion.py --quick      # wiring check, writes no results
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
from aura.local_training import route_flows, train_ae_pass
from aura.models import FlowAutoencoder
import ch2_conditions as c
from ae_drift_diagnostic import local_ae_delta
from ae_routing_cut import client_flows

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("ch1_exclusion")

ROUNDS = 20
NAMES = ds.ORG_NAMES
RETAIL = NAMES.index("retail")
ARMS = {"trust": dict(defence="trust"), "fedavg": dict(defence="fedavg"),
        "trust_no_exclusion": dict(defence="trust", tau=-1.0),
        "trust_retail_benign": dict(defence="trust", benign_only=[RETAIL]),
        "local": dict(defence="local"),
        "trust|bank50": dict(defence="trust", burden=0.5), "fedavg|bank50": dict(defence="fedavg", burden=0.5)}
NO_HEAD = {"w": torch.zeros(4)}


def per_org(mse_ev, mse_th, data, orgs=None):
    """Recall and FPR per destination organisation at the pooled 1% FPR threshold."""
    thr = np.percentile(mse_th, 99)
    y, o = data.ev["y"] == 1, data.ev["o"]
    out = {}
    for i, n in enumerate(NAMES):
        if orgs is not None and i not in orgs:
            continue
        a, b = y & (o == i), ~y & (o == i)
        out[n] = {"recall": float((mse_ev[a] > thr).mean()) if a.any() else None,
                  "fpr": float((mse_ev[b] > thr).mean()) if b.any() else None}
    out["pooled"] = {"recall": float((mse_ev[y] > thr).mean()), "fpr": float((mse_ev[~y] > thr).mean())}
    return out


def score(state, data):
    ae = FlowAutoencoder()
    ae.load_state_dict(state)
    ae.eval()
    return c.encode(ae, data.ev["x"])[1], c.encode(ae, data.thr["x"])[1]


def root_ae_delta(g_ae, root_benign):
    """The AE half of fl_round.root_reference."""
    ae = FlowAutoencoder()
    ae.load_state_dict(g_ae)
    benign, _, _ = route_flows(ae, root_benign, cfg.CH2_MSE_SPLIT_THRESHOLD)
    export, _, _ = train_ae_pass(ae, torch.optim.Adam(ae.parameters(), lr=cfg.FLTRUST_SERVER_LR), benign,
                                 cfg.AE_BATCH_SIZE, cycle_to_horizon=True)
    return {k: export[k] - g_ae[k] for k in g_ae}


def federate(data, pool_x, arm, seed, rounds):
    torch.manual_seed(cfg.SEED + seed)
    g0 = fl_round.snapshot(copy.deepcopy(data.g0).autoencoder)
    flows = client_flows(data, pool_x, arm.get("burden", 0.0), seed, 10 ** 9)
    for i in arm.get("benign_only", []):
        x, y = flows[i]
        flows[i] = (x[y == 0], y[y == 0])
    ae = FlowAutoencoder()

    if arm["defence"] == "local":
        res = {}
        for i, (x, y) in enumerate(flows):
            g = g0
            for _ in range(rounds):
                d, _ = local_ae_delta(ae, g, x, y, True, cfg.AE_LEARNING_RATE)
                g = fl_round.apply_delta(g, d)
            res.update({k: v for k, v in per_org(*score(g, data), data, orgs={i}).items() if k != "pooled"})
        return {"orgs": res}

    root_benign = torch.from_numpy(data.root["x"][data.root["y"] == 0])
    tau = arm.get("tau", cfg.FLTRUST_CH1_THRESHOLD)
    g, excluded, ch1_all, attack_share = g0, [], [], []
    for _ in range(rounds):
        out = [local_ae_delta(ae, g, x, y, True, cfg.AE_LEARNING_RATE) for x, y in flows]
        deltas = [d for d, _ in out]
        attack_share.append([s["attacks_trained"] / max(s["trained"], 1) for _, s in out])
        if arm["defence"] == "fedavg":
            agg = c._mean_delta(deltas, g)
        else:
            agg, _, ch1, _, _, flags = dc_fltrust_aggregate(
                deltas, [NO_HEAD] * len(deltas), root_ae_delta(g, root_benign), NO_HEAD,
                client_round_counts=[0] * len(deltas), ch2_warmup_rounds=cfg.CH2_WARMUP_ROUNDS,
                current_round=1, ch1_threshold=tau)
            excluded.append(flags)
            ch1_all.append(ch1)
        g = fl_round.apply_delta(g, agg)
    res = {"orgs": per_org(*score(g, data), data), "attacks_in_training": dict(zip(NAMES, np.mean(attack_share, axis=0).tolist()))}
    if excluded:
        res["excluded"] = dict(zip(NAMES, np.mean(excluded, axis=0).tolist()))
        res["ch1_mean"] = dict(zip(NAMES, np.mean(ch1_all, axis=0).tolist()))
    return res


def mean_runs(rows):
    def avg(path):
        vals = [r for r in rows]
        for p in path:
            vals = [v[p] for v in vals]
        return None if vals[0] is None else {"mean": float(np.mean(vals)), "sd": float(np.std(vals))}
    out = {"orgs": {n: {m: avg(["orgs", n, m]) for m in rows[0]["orgs"][n]} for n in rows[0]["orgs"]}}
    for key in ("attacks_in_training", "excluded", "ch1_mean"):
        if key in rows[0]:
            out[key] = {n: float(np.mean([r[key][n] for r in rows])) for n in NAMES}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rounds = 2 if args.quick else ROUNDS
    if args.quick:
        args.seeds = 1
    c.CLIENT_FLOWS = 10 ** 9
    torch.set_num_threads(4)
    data = c.Data()
    pool_x = data.train["x"][data.train["y"] == 1]
    counts = {n: {"attacks": int(((data.ev["o"] == i) & (data.ev["y"] == 1)).sum()), "benign": int(((data.ev["o"] == i) & (data.ev["y"] == 0)).sum())}
              for i, n in enumerate(NAMES)}
    log.info(f"routing cut {cfg.CH2_MSE_SPLIT_THRESHOLD:.5f}; ch1 threshold {cfg.FLTRUST_CH1_THRESHOLD}; holdout per org {counts}")
    g0 = fl_round.snapshot(copy.deepcopy(data.g0).autoencoder)
    res = {"rounds": rounds, "seeds": args.seeds, "routing_cut": cfg.CH2_MSE_SPLIT_THRESHOLD, "ch1_threshold": cfg.FLTRUST_CH1_THRESHOLD,
           "holdout_counts": counts, "g0": per_org(*score(g0, data), data), "arms": {}}
    log.info(f"G0 {res['g0']}")
    t0 = time.time()
    for name, arm in ARMS.items():
        row = res["arms"][name] = mean_runs([federate(data, pool_x, arm, s, rounds) for s in range(args.seeds)])
        brief = {n: (None if v["recall"] is None else round(v["recall"]["mean"], 3), None if v["fpr"] is None else round(100 * v["fpr"]["mean"], 2))
                 for n, v in row["orgs"].items()}
        log.info(f"{name}: (recall, FPR %) {brief}; excluded {({n: round(v, 2) for n, v in row.get('excluded', {}).items()})}; "
                 f"attacks in training {({n: round(v, 3) for n, v in row.get('attacks_in_training', {}).items()})} ({time.time() - t0:.0f}s)")
        if not args.quick:
            (ROOT / "results" / "ch1_exclusion.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log.info("DONE")


if __name__ == "__main__":
    main()
