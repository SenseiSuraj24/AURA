#!/usr/bin/env python3
"""
scripts/final_federation.py — the frozen configuration on the full training data
================================================================================
Option A aggregation through fl_round.aggregate, with G0 from `train.py --train-fraction 1.0`
and every organisation training on all of its flows.

Trust thresholds are the 5th percentile of honest scores on a separate seed, channel 2 counted
only after warmup, when the head is actually training. Copy them into config.py afterwards.

Scored on the calibration holdout. The test split stays sealed until the benchmark is finished.

Usage:
    python scripts/final_federation.py
    python scripts/final_federation.py --quick      # wiring check on the current artefacts, writes no results
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import config as cfg
from aura import dataset as ds, fl_round
from aura.models import AttackHead, FlowAutoencoder
import ch2_conditions as c

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("final_federation")

ROUNDS = 20
ARMS = [("trust", "none", 0.0), ("trust", "sign_flip", 0.0), ("trust", "head_flip", 0.0),
        ("trust", "none", 0.5), ("fedavg", "none", 0.0), ("fedavg", "sign_flip", 0.0)]


def detection(ae, head, data):
    """AE alone, head alone and AE + head at thresholds from benign calibration flows."""
    y = data.ev["y"] == 1
    z_ev, mse_ev = c.encode(ae, data.ev["x"])
    z_th, mse_th = c.encode(ae, data.thr["x"])
    s_ev, s_th = c.scores(head, z_ev), c.scores(head, z_th)
    ae1 = mse_ev > np.percentile(mse_th, 99)
    out = {"ae_auroc": float(roc_auc_score(y, mse_ev)), "ae_recall_1pct": float(ae1[y].mean()), "ae_fpr_1pct": float(ae1[~y].mean()),
           "ae_recall_2pct": float((mse_ev > np.percentile(mse_th, 98))[y].mean()),
           "head_auroc": float(roc_auc_score(y, s_ev)), "head_recall_1pct": float((s_ev > np.percentile(s_th, 99))[y].mean())}
    union = ae1 | (s_ev > np.percentile(s_th, 99))
    fpr = float(union[~y].mean())
    out.update(union_recall=float(union[y].mean()), union_fpr=fpr,
               ae_recall_same_fpr=float((mse_ev > np.percentile(mse_th, 100 * (1 - min(fpr, 0.5))))[y].mean()))
    out["ae_recall_by_class"] = {n: float(ae1[y & (data.ev["c"] == k)].mean())
                                 for k, n in enumerate(data.class_names) if (y & (data.ev["c"] == k)).any()}
    return out


def federate(data, pool_x, defence, byz, burden, seed, rounds):
    torch.manual_seed(cfg.SEED + seed)
    rng = np.random.default_rng(cfg.SEED + seed)
    g = copy.deepcopy(data.g0)
    g_ae, g_head = fl_round.snapshot(g.autoencoder), fl_round.snapshot(g.attack_head)
    bank, hospital = ds.ORG_NAMES.index("bank"), ds.ORG_NAMES.index("hospital")
    flows = []
    for i, k in enumerate(data.clients):
        x = k["x"].copy()
        if i == bank and burden > 0:
            n = int(len(x) * burden)
            x[:n] = pool_x[rng.integers(len(pool_x), size=n)]
        flows.append(torch.from_numpy(x))
    root_benign = torch.from_numpy(data.root["x"][data.root["y"] == 0])
    root_attack = torch.from_numpy(data.root["x"][data.root["y"] == 1])
    ae, head = FlowAutoencoder(), AttackHead()
    ch1_all, ch2_all, ae_flag, head_flag = [], [], [], []
    for rnd in range(1, rounds + 1):
        root_ae, root_head = fl_round.root_reference(g_ae, g_head, root_benign, root_attack)
        ae_ds, head_ds = [], []
        for i, x in enumerate(flows):
            u = fl_round.client_update(ae, head, g_ae, g_head, x)
            ae_d, head_d = u.ae_delta, u.head_delta
            if i == hospital and byz == "sign_flip":
                ae_d = {k: -v for k, v in ae_d.items()}
            elif i == hospital and byz == "head_flip" and head_d is not None:
                head_d = {k: -v for k, v in head_d.items()}
            ae_ds.append(ae_d)
            head_ds.append(head_d)
        ch1, ch2 = c._ch2_raw(ae_ds, root_ae), c._ch2_raw(head_ds, root_head)
        if defence == "fedavg":
            agg_ae = c._mean_delta(ae_ds, g_ae)
            agg_head = c._mean_delta(head_ds, g_head) if rnd >= cfg.CH2_WARMUP_ROUNDS else None
        else:
            agg_ae, agg_head, *_ = fl_round.aggregate(ae_ds, head_ds, root_ae, root_head, rnd)
        g_ae, g_head = fl_round.apply_delta(g_ae, agg_ae), fl_round.apply_delta(g_head, agg_head)
        ch1_all.append(ch1)
        ch2_all.append(ch2)
        ae_flag.append([s <= cfg.FLTRUST_CH1_THRESHOLD for s in np.maximum(ch1, 0)])
        head_flag.append([s < cfg.CH2_ANTI_ALIGN_THRESHOLD for s in ch2] if rnd >= cfg.CH2_WARMUP_ROUNDS else [False] * len(flows))

    ae.load_state_dict(g_ae)
    head.load_state_dict(g_head)
    ae.eval()
    head.eval()
    res = detection(ae, head, data)
    if defence != "fedavg":
        names = ds.ORG_NAMES
        late = cfg.CH2_WARMUP_ROUNDS - 1
        res["ae_excluded"] = {n: float(np.mean(ae_flag, axis=0)[i]) for i, n in enumerate(names)}
        res["head_excluded"] = {n: float(np.mean(head_flag[late:], axis=0)[i]) for i, n in enumerate(names)}
    return res, np.array(ch1_all), np.array(ch2_all)


def mean_rows(rows):
    out = {}
    for key, val in rows[0].items():
        out[key] = {k: float(np.mean([r[key][k] for r in rows])) for k in val} if isinstance(val, dict) else float(np.mean([r[key] for r in rows]))
    out["ae_recall_1pct_std"] = float(np.std([r["ae_recall_1pct"] for r in rows]))
    out["union_recall_std"] = float(np.std([r["union_recall"] for r in rows]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rounds, arms = ROUNDS, ARMS
    if args.quick:
        rounds, arms, args.seeds = cfg.CH2_WARMUP_ROUNDS + 1, ARMS[:2], 1
        c.CLIENT_FLOWS = 8000
    else:
        c.CLIENT_FLOWS = 10 ** 9                  # every organisation uses all of its flows
    torch.set_num_threads(4)
    data = c.Data()
    pool_x = data.train["x"][data.train["y"] == 1]
    log.info(f"train fraction {data.split.meta['train_fraction']}; MSE split {cfg.CH2_MSE_SPLIT_THRESHOLD:.5f}; "
             f"clients {[(len(k['y']), int(k['y'].sum())) for k in data.clients]}; eval {len(data.ev['y'])}")
    t0 = time.time()
    g0_ae, g0_head = data.g0.autoencoder.eval(), data.g0.attack_head.eval()
    res = {"train_fraction": data.split.meta["train_fraction"], "rounds": rounds, "seeds": args.seeds,
           "mse_split": cfg.CH2_MSE_SPLIT_THRESHOLD, "g0": detection(g0_ae, g0_head, data), "runs": {}}
    log.info(f"G0 {res['g0']}")

    # Honest scores, with neither filter acting
    cfg.FLTRUST_CH1_THRESHOLD, cfg.CH2_ANTI_ALIGN_THRESHOLD = -1.0, -2.0
    _, ch1, ch2 = federate(data, pool_x, "trust", "none", 0.0, c.CALIB_SEED, rounds)
    tau1 = float(np.percentile(np.maximum(ch1, 0), 5))
    tau2 = float(np.nanpercentile(ch2[cfg.CH2_WARMUP_ROUNDS - 1:], 5))
    cfg.FLTRUST_CH1_THRESHOLD, cfg.CH2_ANTI_ALIGN_THRESHOLD = tau1, tau2
    res["thresholds"] = {"FLTRUST_CH1_THRESHOLD": tau1, "CH2_ANTI_ALIGN_THRESHOLD": tau2}
    log.info(f"thresholds {res['thresholds']} ({time.time() - t0:.0f}s)")

    for defence, byz, burden in arms:
        key = f"{defence}|{byz}|{burden}"
        rows = [federate(data, pool_x, defence, byz, burden, s, rounds)[0] for s in range(args.seeds)]
        res["runs"][key] = mean_rows(rows)
        log.info(f"{key}: {res['runs'][key]} ({time.time() - t0:.0f}s)")
        if not args.quick:
            (ROOT / "results" / "final_federation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log.info("DONE")


if __name__ == "__main__":
    main()
