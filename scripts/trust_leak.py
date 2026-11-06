#!/usr/bin/env python3
"""
scripts/trust_leak.py — does an organisation's FLTrust score reveal that it is under attack?
============================================================================================
ch1_exclusion.py found that FLTrust gives low trust to the organisations that carry the most
attack traffic, because attack flows that fall under the routing cut end up in their AE
training. If that is a dose-response, the server learns something about each organisation's
incident status from model updates alone. That is either a privacy leak or a free alarm,
depending on who you ask.

The bank, which has no attacks in its own data, gets a share b of its flows replaced with
training-pool attack flows, b in {0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8}. Everyone else is honest and
unchanged. Vanilla FLTrust (ae_only_fltrust_aggregate), AE only, P99 routing cut, 20 rounds,
3 seeds. Each round records every client's raw cosine with the server's root update (before the
ReLU) and, for the bank, the local signal it already has: the share of its own flows above the
routing cut, and the share of attacks in what it trained on.

Server-side detector: the bank's cosine minus the median cosine of the other four that round
(so a round where everyone drifts does not count). Its AUROC against the same seed's b = 0
rounds says how well the server can tell an attacked bank from a clean one.

Predictions, fixed before running: the bank's mean cosine falls monotonically with b (Spearman
rho <= -0.9 over the seven levels); the server-side AUROC is >= 0.8 by b = 0.2; the local
high-error share separates at least as well as the server signal at every b.

Calibration holdout only. The test split stays sealed.

Usage:
    python scripts/trust_leak.py
    python scripts/trust_leak.py --quick      # wiring check, writes no results
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
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import config as cfg
from aura import dataset as ds, fl_round
from aura.fl_server import ae_only_fltrust_aggregate
from aura.models import FlowAutoencoder
import ch2_conditions as c
from ae_drift_diagnostic import local_ae_delta
from ae_routing_cut import client_flows
from ch1_exclusion import root_ae_delta

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("trust_leak")

BURDENS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
BANK = ds.ORG_NAMES.index("bank")


def flat(d):
    return torch.cat([v.flatten().float() for v in d.values()])


def federate(data, pool_x, burden, seed, rounds):
    torch.manual_seed(cfg.SEED + seed)
    g = fl_round.snapshot(copy.deepcopy(data.g0).autoencoder)
    flows = client_flows(data, pool_x, burden, seed, 10 ** 9)
    root_benign = torch.from_numpy(data.root["x"][data.root["y"] == 0])
    ae = FlowAutoencoder()
    cos, high, att_trained = [], [], []
    for _ in range(rounds):
        out = [local_ae_delta(ae, g, x, y, True, cfg.AE_LEARNING_RATE) for x, y in flows]
        deltas = [d for d, _ in out]
        root = root_ae_delta(g, root_benign)
        r = flat(root)
        cos.append([float(F.cosine_similarity(flat(d).unsqueeze(0), r.unsqueeze(0))) for d in deltas])
        s = out[BANK][1]
        high.append(1.0 - s["trained"] / max(s["flows"], 1))
        att_trained.append(s["attacks_trained"] / max(s["trained"], 1))
        agg, _ = ae_only_fltrust_aggregate(deltas, root)
        g = fl_round.apply_delta(g, agg)
    cos = np.array(cos)
    others = np.delete(cos, BANK, axis=1)
    return {"cos": cos.tolist(), "relative": (cos[:, BANK] - np.median(others, axis=1)).tolist(),
            "high_share": high, "attack_share_trained": att_trained}


def auroc(clean, attacked):
    """How well a LOWER value flags the attacked bank."""
    y = np.r_[np.zeros(len(clean)), np.ones(len(attacked))]
    return float(roc_auc_score(y, -np.r_[clean, attacked]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    burdens = BURDENS
    if args.quick:
        args.seeds, args.rounds, burdens = 1, 2, [0.0, 0.4]
    c.CLIENT_FLOWS = 10 ** 9
    torch.set_num_threads(4)
    data = c.Data()
    pool_x = data.train["x"][data.train["y"] == 1]
    out = ROOT / "results" / "trust_leak.json"
    res = {"rounds": args.rounds, "seeds": args.seeds, "routing_cut": cfg.CH2_MSE_SPLIT_THRESHOLD,
           "orgs": list(ds.ORG_NAMES), "runs": {}}
    t0 = time.time()
    for b in burdens:
        res["runs"][str(b)] = [federate(data, pool_x, b, s, args.rounds) for s in range(args.seeds)]
        rows = res["runs"][str(b)]
        log.info(f"b={b}: bank cos {np.mean([np.mean(np.array(r['cos'])[:, BANK]) for r in rows]):.3f}; "
                 f"relative {np.mean([np.mean(r['relative']) for r in rows]):+.3f}; "
                 f"high share {np.mean([np.mean(r['high_share']) for r in rows]):.3f}; "
                 f"attacks in training {np.mean([np.mean(r['attack_share_trained']) for r in rows]):.3f} ({time.time() - t0:.0f}s)")
        if not args.quick:
            out.write_text(json.dumps(res, indent=1), encoding="utf-8")

    clean = res["runs"]["0.0"]
    summary = {}
    for b in burdens[1:]:
        rows = res["runs"][str(b)]
        summary[str(b)] = {
            "bank_cos_mean": float(np.mean([np.mean(np.array(r["cos"])[:, BANK]) for r in rows])),
            "server_auroc": float(np.mean([auroc(cl["relative"], r["relative"]) for cl, r in zip(clean, rows)])),
            "local_auroc": float(np.mean([auroc(-np.array(cl["high_share"]), -np.array(r["high_share"])) for cl, r in zip(clean, rows)])),
        }
    means = [float(np.mean([np.mean(np.array(r["cos"])[:, BANK]) for r in res["runs"][str(b)]])) for b in burdens]
    res["summary"] = summary
    res["bank_cos_by_burden"] = dict(zip(map(str, burdens), means))
    res["spearman_rho"] = float(spearmanr(burdens, means).correlation) if len(burdens) > 2 else None
    log.info(f"summary {json.dumps(summary)}; spearman {res['spearman_rho']}")
    if not args.quick:
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    log.info("DONE")


if __name__ == "__main__":
    main()
