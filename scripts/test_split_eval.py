#!/usr/bin/env python3
"""
scripts/test_split_eval.py — the one scoring of the sealed test split
=====================================================================
Every result so far was scored on half of the calibration split. This scores the frozen
configuration on the test split, once: G0 trained on the full pool, the P99 routing cut and the
calibrated trust thresholds, 20 rounds, 3 seeds. The list of runs was fixed before running:

  G0                          the starting model, no federation
  fedavg|none at P75 and P99  the routing cut on clean clients
  5 defences x 4 attacks      DC-FLTrust, vanilla FLTrust, FedAvg, median, Krum against no attack,
                              one x10 scaled flip, two poisoners, two adaptive attackers at cosine 0

The 1% FPR threshold comes from all benign calibration flows; the FPR it reaches on the test
split is reported alongside recall. Refuses to run if results/test_split.json exists, so the test
split is scored only once. --quick checks the wiring on the calibration holdout and never reads
the test split.

Usage:
    python scripts/test_split_eval.py
    python scripts/test_split_eval.py --quick
"""

import argparse
import copy
import json
import logging
import subprocess
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
from aura.models import FlowAutoencoder
import ch2_conditions as c
import wave3_byzantine as w3

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("test_split")

OUT = ROOT / "results" / "test_split.json"
ROUNDS = 20
ATTACKS = [("none", 0), ("scaled_flip", 1), ("poison", 2), ("adaptive@0.0", 2)]
DEFENCES = ["trust", "fltrust", "fedavg", "median", "krum"]


def score(state, data):
    ae = FlowAutoencoder()
    ae.load_state_dict(state)
    ae.eval()
    y = data.ev["y"] == 1
    _, mse_ev = c.encode(ae, data.ev["x"])
    _, mse_th = c.encode(ae, data.thr["x"])
    t1, t2 = np.percentile(mse_th, 99), np.percentile(mse_th, 98)
    hit = mse_ev > t1
    out = {"auroc": float(roc_auc_score(y, mse_ev)), "recall_1pct": float(hit[y].mean()), "fpr_at_1pct": float(hit[~y].mean()),
           "recall_2pct": float((mse_ev > t2)[y].mean()), "fpr_at_2pct": float((mse_ev > t2)[~y].mean())}
    out["recall_by_class"] = {n: float(hit[y & (data.ev["c"] == k)].mean()) for k, n in enumerate(data.class_names)
                              if (y & (data.ev["c"] == k)).any()}
    out["by_org"] = {n: {"recall": float(hit[y & (data.ev["o"] == i)].mean()) if (y & (data.ev["o"] == i)).any() else None,
                         "fpr": float(hit[~y & (data.ev["o"] == i)].mean())} for i, n in enumerate(ds.ORG_NAMES)}
    return out


def mean_rows(rows):
    out = {}
    for k, v in rows[0].items():
        if isinstance(v, dict):
            out[k] = mean_rows([r[k] for r in rows])
        elif v is None:
            out[k] = None
        else:
            out[k] = float(np.mean([r[k] for r in rows]))
            if k in ("recall_1pct", "auroc"):
                out[k + "_std"] = float(np.std([r[k] for r in rows]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if not args.quick and OUT.exists():
        sys.exit(f"{OUT} exists: the test split has already been scored")
    rounds, seeds, attacks, defences = (2, 1, ATTACKS[:2], ["trust", "fedavg"]) if args.quick else (ROUNDS, 3, ATTACKS, DEFENCES)

    c.CLIENT_FLOWS = 10 ** 9
    torch.set_num_threads(4)
    data = c.Data()
    calib = data.fetch(data.split.flows["calib"])
    data.thr = {k: v[calib["y"] == 0] for k, v in calib.items()}
    if not args.quick:
        data.ev = data.fetch(data.split.flows["test"])
    pool_x = data.train["x"][data.train["y"] == 1]
    p75 = float(json.loads((cfg.LOGS_DIR / "calibration_results.json").read_text(encoding="utf-8"))["p75"])
    p99 = cfg.CH2_MSE_SPLIT_THRESHOLD
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "aura", "scripts", "config.py"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    res = {"split": "calib (wiring check)" if args.quick else "test", "commit": commit, "uncommitted_code": dirty,
           "split_hash": data.split.meta["hash"], "train_fraction": data.split.meta["train_fraction"],
           "routing_cut_p99": p99, "routing_cut_p75": p75, "ch1_threshold": cfg.FLTRUST_CH1_THRESHOLD,
           "rounds": rounds, "seeds": seeds,
           "n_eval": {"attacks": int((data.ev["y"] == 1).sum()), "benign": int((data.ev["y"] == 0).sum())},
           "n_threshold_benign": int(len(data.thr["y"])), "runs": {}}
    log.info(f"scoring on {res['split']}: {res['n_eval']}; threshold from {res['n_threshold_benign']} benign calibration flows; "
             f"commit {commit[:8]}{' (uncommitted changes)' if dirty else ''}")

    res["g0"] = score(fl_round.snapshot(copy.deepcopy(data.g0).autoencoder), data)
    log.info(f"G0: recall {res['g0']['recall_1pct']:.3f}, FPR {res['g0']['fpr_at_1pct']:.4f}, AUROC {res['g0']['auroc']:.3f}")

    plan = [("fedavg", "none", 0, p75, "p75")] + [(d, k, n, p99, "p99") for k, n in attacks for d in defences]
    t0 = time.time()
    for defence, kind, n_att, cut, tag in plan:
        cfg.CH2_MSE_SPLIT_THRESHOLD = cut
        rows = []
        for s in range(seeds):
            extra, g = w3.federate(data, pool_x, defence, kind, n_att, s, rounds, return_state=True)
            rows.append({**score(g, data), **{k: v for k, v in extra.items() if "excluded" in k}})
        key = f"{defence}|{kind}|{n_att}|{tag}"
        row = res["runs"][key] = mean_rows(rows)
        log.info(f"{key}: recall {row['recall_1pct']:.3f} (sd {row['recall_1pct_std']:.3f}), FPR {row['fpr_at_1pct']:.4f}, "
                 f"AUROC {row['auroc']:.3f}, attackers excluded {row.get('attackers_excluded')} ({time.time() - t0:.0f}s)")
    cfg.CH2_MSE_SPLIT_THRESHOLD = p99
    if not args.quick:
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    log.info("DONE")


if __name__ == "__main__":
    main()
