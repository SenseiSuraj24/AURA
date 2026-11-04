#!/usr/bin/env python3
"""
scripts/ae_drift_diagnostic.py — why does federated AE training lower detection?
================================================================================
final_federation.py took AE recall at 1% FPR from 0.625 (G0) to 0.16-0.20 with no attacker.
Each arm changes one thing about what the clients train on, under plain FedAvg so that trust
scores play no part. Recall is tracked every round to show when the fall happens.

  as_is           every client flow, routed by MSE (should reproduce FedAvg's 0.161)
  benign_routed   label-oracle: benign flows only, still routed by MSE
  benign_all      label-oracle: benign flows only, nothing routed away
  mixed_unrouted  every client flow, nothing routed away
  central         benign flows of all clients pooled into one client, routed
  benign_lr_1e-4  benign_routed at a tenth of the client learning rate

Clients stop at the export step, since nothing after FL_EXPORT_HORIZON reaches the aggregate.

Criteria, fixed before running, on benign_routed recall at the last round:
  >= 0.55  contamination is the main cause;  <= 0.25  it is not;  between  partial cause.

Scored on the calibration holdout. The test split stays sealed.

Usage:
    python scripts/ae_drift_diagnostic.py
    python scripts/ae_drift_diagnostic.py --quick      # wiring check, writes no results
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
from aura import fl_round
from aura.local_training import route_flows, train_ae_pass
from aura.models import FlowAutoencoder
import ch2_conditions as c

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("ae_drift")

ROUNDS = 20
ARMS = {"as_is": dict(benign_only=False, routed=True),
        "benign_routed": dict(benign_only=True, routed=True),
        "benign_all": dict(benign_only=True, routed=False),
        "mixed_unrouted": dict(benign_only=False, routed=False),
        "central": dict(benign_only=True, routed=True, pooled=True),
        "benign_lr_1e-4": dict(benign_only=True, routed=True, lr=1e-4)}


def ae_metrics(ae, data):
    y = data.ev["y"] == 1
    _, mse_ev = c.encode(ae, data.ev["x"])
    _, mse_th = c.encode(ae, data.thr["x"])
    return {"auroc": float(roc_auc_score(y, mse_ev)),
            "recall_1pct": float((mse_ev > np.percentile(mse_th, 99))[y].mean())}


def client_flows(data, benign_only, pooled):
    """Flows and labels per client. Labels only score the routing; training never sees them."""
    sets = [(k["x"][k["y"] == 0], k["y"][k["y"] == 0]) if benign_only else (k["x"], k["y"]) for k in data.clients]
    if pooled:
        sets = [(np.concatenate([s[0] for s in sets]), np.concatenate([s[1] for s in sets]))]
    return [(torch.from_numpy(x), torch.from_numpy(y)) for x, y in sets]


def local_ae_delta(ae, g_ae, x, y, routed, lr):
    """The AE half of fl_round.client_update, stopped at the export step."""
    ae.load_state_dict(g_ae)
    if routed:
        ae.eval()
        with torch.no_grad():
            recon, _ = ae(x)
            keep = torch.nn.functional.mse_loss(recon, x, reduction="none").mean(dim=1) < cfg.CH2_MSE_SPLIT_THRESHOLD
        train_x, _, _ = route_flows(ae, x, cfg.CH2_MSE_SPLIT_THRESHOLD)
    else:
        keep = torch.ones(len(x), dtype=torch.bool)
        train_x = x
    ae.train()
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    export, _, steps = train_ae_pass(ae, opt, train_x, cfg.AE_BATCH_SIZE, cycle_to_horizon=True)
    assert steps == cfg.FL_EXPORT_HORIZON, f"client stopped after {steps} steps"
    stats = {"trained": int(keep.sum()), "attacks_trained": int((y[keep] == 1).sum()), "attacks": int((y == 1).sum()), "flows": len(x)}
    return {k: export[k] - g_ae[k] for k in g_ae}, stats


def federate(data, arm, seed, rounds):
    torch.manual_seed(cfg.SEED + seed)
    g_ae = fl_round.snapshot(copy.deepcopy(data.g0).autoencoder)
    flows = client_flows(data, arm["benign_only"], arm.get("pooled", False))
    lr = arm.get("lr", cfg.AE_LEARNING_RATE)
    ae = FlowAutoencoder()
    ae.load_state_dict(g_ae)
    ae.eval()
    curve, shares = [ae_metrics(ae, data)], []
    for _ in range(rounds):
        out = [local_ae_delta(ae, g_ae, x, y, arm["routed"], lr) for x, y in flows]
        g_ae = fl_round.apply_delta(g_ae, c._mean_delta([d for d, _ in out], g_ae))
        total = {k: sum(s[k] for _, s in out) for k in out[0][1]}
        shares.append({"trained_share": total["trained"] / total["flows"],
                       "attacks_in_training": total["attacks_trained"] / max(total["trained"], 1),
                       "attacks_kept": total["attacks_trained"] / max(total["attacks"], 1)})
        ae.load_state_dict(g_ae)
        ae.eval()
        curve.append(ae_metrics(ae, data))
    return curve, shares


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rounds, arms = ROUNDS, ARMS
    if args.quick:
        rounds, args.seeds = 2, 1
    c.CLIENT_FLOWS = 10 ** 9
    torch.set_num_threads(4)
    data = c.Data()
    t0 = time.time()
    res = {"rounds": rounds, "seeds": args.seeds, "train_fraction": data.split.meta["train_fraction"], "arms": {}}
    for name, arm in arms.items():
        runs = [federate(data, arm, s, rounds) for s in range(args.seeds)]
        recall = np.array([[p["recall_1pct"] for p in curve] for curve, _ in runs])
        auroc = np.array([[p["auroc"] for p in curve] for curve, _ in runs])
        first = runs[0][1]
        res["arms"][name] = {"recall_by_round": recall.mean(axis=0).tolist(), "auroc_by_round": auroc.mean(axis=0).tolist(),
                             "final_recall": float(recall[:, -1].mean()), "final_recall_std": float(recall[:, -1].std()),
                             "final_auroc": float(auroc[:, -1].mean()), "routing_first_round": first[0], "routing_last_round": first[-1]}
        log.info(f"{name}: recall {res['arms'][name]['final_recall']:.3f} (sd {res['arms'][name]['final_recall_std']:.3f}), "
                 f"auroc {res['arms'][name]['final_auroc']:.3f}; rounds 0,1,2,5,10,20 "
                 f"{[round(float(recall.mean(axis=0)[min(r, rounds)]), 3) for r in (0, 1, 2, 5, 10, 20)]}; "
                 f"routing {first[0]} -> {first[-1]} ({time.time() - t0:.0f}s)")
        if not args.quick:
            (ROOT / "results" / "ae_drift.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log.info("DONE")


if __name__ == "__main__":
    main()
