#!/usr/bin/env python3
"""
scripts/cut_quantile_rule.py — is "never train below the alarm quantile" a rule?
================================================================================
ae_routing_cut.py found that frozen cuts at P75, P90 and P95 collapse recall at 1% FPR over
40 rounds while P99, the alarm quantile, holds. Proposed mechanism: flows excluded above the
alarm quantile are already above the threshold, so their growing error cannot move it, but
flows in the band between the cut and the alarm quantile can grow across it and push it up.

  quantile   frozen cuts at P97, P98, P99.5 and P99.9 of G0's benign error, clean clients,
             40 rounds. The mechanism predicts a break just below P99 and stability above it.
  recompute  each client re-estimates its cut every round as a quantile of its own flows'
             error under the current global model (trimmed-loss style), at P75 and P95,
             clean over 40 rounds and with bank at 80% attacks over 20 rounds.
  bands      mean error per G0-quantile band of benign calibration flows each round
             ([0, P75), [P75, P95), [P95, P99), [P99, max]), frozen P75 and P99, 40 rounds.

Predictions, fixed before running: P97 and P98 end below 0.40 recall at round 40 while P99.5
and P99.9 end at or above 0.55; under the P75 cut the untrained bands' error grows far more
than the trained band's, and under P99 no band grows much.

Plain FedAvg, AE only, clients stopped at the export step, calibration holdout. The test split
stays sealed.

Usage:
    python scripts/cut_quantile_rule.py
    python scripts/cut_quantile_rule.py --quick      # wiring check, writes no results
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
from aura import fl_round
from aura.models import FlowAutoencoder
import ch2_conditions as c
from ae_drift_diagnostic import local_ae_delta
from ae_routing_cut import client_flows, metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("cut_rule")

BAND_EDGES = [75, 95, 99]


def own_quantile_cut(ae, g, x, q):
    ae.load_state_dict(g)
    ae.eval()
    with torch.no_grad():
        recon, _ = ae(x)
        mse = ((recon - x) ** 2).mean(dim=1)
    return float(torch.quantile(mse, q / 100.0)) if len(mse) <= 2 ** 24 else float(np.percentile(mse.numpy(), q))


def federate(data, pool_x, mode, q, burden, seed, rounds, frozen_cuts, bands):
    torch.manual_seed(cfg.SEED + seed)
    g = fl_round.snapshot(copy.deepcopy(data.g0).autoencoder)
    flows = client_flows(data, pool_x, burden, seed, 10 ** 9)
    ae = FlowAutoencoder()
    ae.load_state_dict(g)
    ae.eval()
    curve, band_curve = [metrics(ae, data)], []
    _, g0_mse = c.encode(ae, data.thr["x"])
    edges = np.percentile(g0_mse, BAND_EDGES)
    which = np.digitize(g0_mse, edges)

    def band_means(model):
        _, m = c.encode(model, data.thr["x"])
        return [float(m[which == b].mean()) for b in range(len(BAND_EDGES) + 1)]

    if bands:
        band_curve.append(band_means(ae))
    for _ in range(rounds):
        deltas = []
        for x, y in flows:
            cfg.CH2_MSE_SPLIT_THRESHOLD = frozen_cuts[q] if mode == "frozen" else own_quantile_cut(ae, g, x, q)
            deltas.append(local_ae_delta(ae, g, x, y, True, cfg.AE_LEARNING_RATE)[0])
        g = fl_round.apply_delta(g, c._mean_delta(deltas, g))
        ae.load_state_dict(g)
        ae.eval()
        curve.append(metrics(ae, data))
        if bands:
            band_curve.append(band_means(ae))
    return curve, band_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--long", action="store_true", help="P97, P98 and P99 over 80 rounds")
    args = ap.parse_args()
    clean, attacked = (2, 2) if args.quick else (40, 20)
    if args.quick:
        args.seeds = 1
    c.CLIENT_FLOWS = 10 ** 9
    torch.set_num_threads(4)
    data = c.Data()
    pool_x = data.train["x"][data.train["y"] == 1]
    g0 = copy.deepcopy(data.g0).autoencoder.eval()
    _, mse_thr = c.encode(g0, data.thr["x"])
    qs = [75, 95, 97, 98, 99, 99.5, 99.9]
    frozen_cuts = {q: float(np.percentile(mse_thr, q)) for q in qs}
    saved_cut = cfg.CH2_MSE_SPLIT_THRESHOLD
    log.info(f"frozen cuts {frozen_cuts}")

    out = ROOT / "results" / "cut_quantile_rule.json"
    if args.long:
        # P97 lost one seed between rounds 30 and 40: does every cut below P99 collapse, only later?
        plan = [("long", "frozen", q, 0.0, 2 if args.quick else 80, True) for q in (97, 98, 99)]
        out = ROOT / "results" / "cut_quantile_rule_long.json"
    else:
        plan = [("quantile", "frozen", q, 0.0, clean, False) for q in (97, 98, 99.5, 99.9)]
        plan += [("recompute", "recompute", q, 0.0, clean, False) for q in (75, 95)]
        plan += [("recompute", "recompute", q, 0.8, attacked, False) for q in (75, 95)]
        plan += [("bands", "frozen", q, 0.0, clean, True) for q in (75, 99)]
    res = {"seeds": args.seeds, "frozen_cuts": {str(k): v for k, v in frozen_cuts.items()}, "band_edges_pct": BAND_EDGES, "runs": {}}
    t0 = time.time()
    try:
        for part, mode, q, burden, rounds, bands in plan:
            runs = [federate(data, pool_x, mode, q, burden, s, rounds, frozen_cuts, bands) for s in range(args.seeds)]
            recall = np.array([[p["recall_1pct"] for p in cv] for cv, _ in runs])
            row = {"recall_by_round": recall.mean(0).tolist(), "final_recall": float(recall[:, -1].mean()), "final_recall_std": float(recall[:, -1].std()),
                   "auroc_by_round": np.array([[p["auroc"] for p in cv] for cv, _ in runs]).mean(0).tolist(),
                   "benign_p99_by_round": np.array([[p["benign_p99"] for p in cv] for cv, _ in runs]).mean(0).tolist()}
            if bands:
                row["band_mse_by_round"] = np.array([b for _, b in runs]).mean(0).tolist()
            key = f"{part}|{mode}|p{q}|burden{burden}"
            res["runs"][key] = row
            marks = [r for r in (0, 5, 10, 20, 30, 40, 60, 80) if r <= rounds]
            extra = f"; bands r0 {np.round(row['band_mse_by_round'][0], 4).tolist()} -> r{rounds} {np.round(row['band_mse_by_round'][-1], 4).tolist()}" if bands else ""
            log.info(f"{key}: recall {row['final_recall']:.3f} (sd {row['final_recall_std']:.3f}); rounds {marks} "
                     f"{[round(row['recall_by_round'][r], 3) for r in marks]}; benign P99 {round(row['benign_p99_by_round'][-1], 4)}{extra} ({time.time() - t0:.0f}s)")
            if not args.quick:
                out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    finally:
        cfg.CH2_MSE_SPLIT_THRESHOLD = saved_cut
    log.info("DONE")


if __name__ == "__main__":
    main()
