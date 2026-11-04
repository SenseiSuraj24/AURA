#!/usr/bin/env python3
"""
scripts/ae_routing_cut.py — which routing cut lets the federated AE keep its detection?
=======================================================================================
ae_drift_diagnostic.py showed the P75 cut lowers recall on its own, with no attack flows reaching
the AE. A higher cut keeps the hard benign flows in training but may let more attacks in, so each
cut runs on clean clients and with one organisation under heavy attack. Plain FedAvg, AE only.

The cuts are percentiles of G0's error on benign calibration flows, fixed for the whole run as the
configured split is. The benign P99 error is tracked each round, since a rising tail would explain
why attacks fall below the 1% FPR threshold.

Clean clients run 40 rounds, because the P75 fall came late; attacked ones run 20. Clean P75 is
repeated at 8,000 flows per client, the size of the earlier 10-round runs that held 0.65.

Criteria, fixed before running, for a cut that needs no labels:
  recall >= 0.55 on clean clients at round 40, and >= 0.50 with bank at 80% attacks at round 20.

Scored on the calibration holdout. The test split stays sealed.

Usage:
    python scripts/ae_routing_cut.py
    python scripts/ae_routing_cut.py --quick      # wiring check, writes no results
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
from aura.models import FlowAutoencoder
import ch2_conditions as c
from ae_drift_diagnostic import local_ae_delta
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("ae_routing_cut")

CUTS = [75, 90, 95, 99, None]       # None: nothing routed away
BURDENS = [0.5, 0.8]
CLEAN_ROUNDS, ATTACK_ROUNDS = 40, 20
SMALL_FLOWS = 8000


def metrics(ae, data):
    y = data.ev["y"] == 1
    _, mse_ev = c.encode(ae, data.ev["x"])
    _, mse_th = c.encode(ae, data.thr["x"])
    p99 = np.percentile(mse_th, 99)
    return {"auroc": float(roc_auc_score(y, mse_ev)), "recall_1pct": float((mse_ev > p99)[y].mean()),
            "benign_p99": float(p99), "benign_median": float(np.median(mse_th))}


def client_flows(data, pool_x, burden, seed, max_flows):
    rng = np.random.default_rng(cfg.SEED + seed)
    bank = ds.ORG_NAMES.index("bank")
    out = []
    for i, k in enumerate(data.clients):
        x, y = k["x"][:max_flows].copy(), k["y"][:max_flows].copy()
        if i == bank and burden > 0:
            n = int(len(x) * burden)
            x[:n] = pool_x[rng.integers(len(pool_x), size=n)]
            y[:n] = 1
        out.append((torch.from_numpy(x), torch.from_numpy(y)))
    return out


def federate(data, pool_x, routed, burden, seed, rounds, max_flows):
    torch.manual_seed(cfg.SEED + seed)
    g_ae = fl_round.snapshot(copy.deepcopy(data.g0).autoencoder)
    flows = client_flows(data, pool_x, burden, seed, max_flows)
    ae = FlowAutoencoder()
    ae.load_state_dict(g_ae)
    ae.eval()
    curve, shares = [metrics(ae, data)], []
    for _ in range(rounds):
        out = [local_ae_delta(ae, g_ae, x, y, routed, cfg.AE_LEARNING_RATE) for x, y in flows]
        g_ae = fl_round.apply_delta(g_ae, c._mean_delta([d for d, _ in out], g_ae))
        total = {k: sum(s[k] for _, s in out) for k in out[0][1]}
        shares.append({"attacks_in_training": total["attacks_trained"] / max(total["trained"], 1),
                       "attacks_kept": total["attacks_trained"] / max(total["attacks"], 1),
                       "trained_share": total["trained"] / total["flows"]})
        ae.load_state_dict(g_ae)
        ae.eval()
        curve.append(metrics(ae, data))
    return curve, shares


def summarise(runs):
    table = {k: np.array([[p[k] for p in curve] for curve, _ in runs]) for k in runs[0][0][0]}
    routing = {k: np.array([[s[k] for s in shares] for _, shares in runs]).mean(axis=0) for k in runs[0][1][0]}
    return {"recall_by_round": table["recall_1pct"].mean(axis=0).tolist(), "auroc_by_round": table["auroc"].mean(axis=0).tolist(),
            "benign_p99_by_round": table["benign_p99"].mean(axis=0).tolist(), "benign_median_by_round": table["benign_median"].mean(axis=0).tolist(),
            "final_recall": float(table["recall_1pct"][:, -1].mean()), "final_recall_std": float(table["recall_1pct"][:, -1].std()),
            "final_auroc": float(table["auroc"][:, -1].mean()),
            "attacks_in_training_by_round": routing["attacks_in_training"].tolist(), "attacks_kept_by_round": routing["attacks_kept"].tolist(),
            "trained_share_by_round": routing["trained_share"].tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    clean_rounds, attack_rounds = (2, 2) if args.quick else (CLEAN_ROUNDS, ATTACK_ROUNDS)
    if args.quick:
        args.seeds = 1
    c.CLIENT_FLOWS = 10 ** 9
    torch.set_num_threads(4)
    data = c.Data()
    pool_x = data.train["x"][data.train["y"] == 1]
    g0 = copy.deepcopy(data.g0).autoencoder.eval()
    _, mse_thr = c.encode(g0, data.thr["x"])
    values = {p: float(np.percentile(mse_thr, p)) for p in CUTS if p is not None}
    log.info(f"cuts {values}; configured split {cfg.CH2_MSE_SPLIT_THRESHOLD:.5f}")

    plan = [(p, 0.0, clean_rounds, 10 ** 9) for p in CUTS]
    plan += [(p, b, attack_rounds, 10 ** 9) for b in BURDENS for p in CUTS]
    plan += [(75, 0.0, attack_rounds, SMALL_FLOWS)]
    res = {"seeds": args.seeds, "train_fraction": data.split.meta["train_fraction"], "cuts": {str(k): v for k, v in values.items()},
           "configured_split": cfg.CH2_MSE_SPLIT_THRESHOLD, "runs": {}}
    t0 = time.time()
    for p, burden, rounds, max_flows in plan:
        if p is not None:
            cfg.CH2_MSE_SPLIT_THRESHOLD = values[p]
        runs = [federate(data, pool_x, p is not None, burden, s, rounds, max_flows) for s in range(args.seeds)]
        key = f"{'none' if p is None else f'p{p}'}|burden{burden}|{'small' if max_flows == SMALL_FLOWS else 'full'}"
        row = res["runs"][key] = summarise(runs)
        marks = [r for r in (0, 5, 10, 20, 30, 40) if r <= rounds]
        log.info(f"{key}: recall {row['final_recall']:.3f} (sd {row['final_recall_std']:.3f}), auroc {row['final_auroc']:.3f}; "
                 f"rounds {marks} {[round(row['recall_by_round'][r], 3) for r in marks]}; "
                 f"benign P99 {[round(row['benign_p99_by_round'][r], 4) for r in marks]}; "
                 f"attacks in training {row['attacks_in_training_by_round'][0]:.4f} -> {row['attacks_in_training_by_round'][-1]:.4f} ({time.time() - t0:.0f}s)")
        if not args.quick:
            (ROOT / "results" / "ae_routing_cut.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log.info("DONE")


if __name__ == "__main__":
    main()
