#!/usr/bin/env python3
"""
scripts/wave3_byzantine.py — does DC-FLTrust protect the AE against attackers?
==============================================================================
In final_federation.py one sign-flipper out of five barely moved FedAvg (0.584 to 0.583),
so there was nothing for trust to block. These attacks are meant to hurt:

  sign_flip       the attacker sends minus its honest update
  scaled_flip     minus ten times its honest update
  poison          trains with the honest recipe on attack flows, so the AE learns to
                  reconstruct attacks (data poisoning)
  adaptive        the poison direction mixed into the honest update so its cosine with the
                  honest update is 0.6, at the honest norm: aims to look trustworthy while
                  still pushing towards reconstructing attacks

One attacker (hospital) or two (hospital, university) out of five organisations. The attackers
do not contribute honest updates. Defences: DC-FLTrust channel 1 (the configured threshold),
FedAvg, coordinate-wise median, multi-Krum keeping 3 of 5, and vanilla FLTrust
(ae_only_fltrust_aggregate: trust weights with no exclusion threshold; "excluded" there means
a weight of zero). AE only, P99 routing cut, clients
stopped at the export step.

--sweep repeats the adaptive attack with two attackers at lower cosines (more poison, less
cover), since at 0.6 it barely moved FedAvg and so tested nothing.

--rootaware gives two attackers the server's root update (worst case: they hold the root data)
and sets their cosine WITH THE ROOT to 0.6 down to 0.02, the rest poison. FLTrust weights by that
cosine, so this is the attack it is weakest against. Prediction, fixed before running: FLTrust
loses more than 0.03 recall at some cosine in the sweep; a cosine just above zero buys weight
while carrying almost pure poison.

--krum-control runs Krum correctly configured for one attacker (f=1, so k = n - f - 2 = 2
neighbours) against one attacker, next to the old k = 1 rule. Prediction: f=1 Krum protects
against every one-attacker attack that hurts FedAvg.

Criterion, fixed before running: a defence protects against an attack if its recall at 1% FPR
stays within 0.03 of its own clean recall, for an attack that lowers FedAvg's by more than 0.05.

Scored on the calibration holdout. The test split stays sealed.

Usage:
    python scripts/wave3_byzantine.py
    python scripts/wave3_byzantine.py --sweep      # adaptive attack at lower cosines
    python scripts/wave3_byzantine.py --quick      # wiring check, writes no results
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import config as cfg
from aura import dataset as ds, fl_round
from aura.fl_server import ae_only_fltrust_aggregate, dc_fltrust_aggregate, krum_select
from aura.local_training import train_ae_pass
from aura.models import FlowAutoencoder
import ch2_conditions as c
from ae_drift_diagnostic import local_ae_delta
from ae_routing_cut import client_flows
from ch1_exclusion import NO_HEAD, root_ae_delta

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("wave3")

ROUNDS = 20
NAMES = ds.ORG_NAMES
ATTACKERS = {1: [NAMES.index("hospital")], 2: [NAMES.index("hospital"), NAMES.index("university")]}
ATTACKS = [("none", 0), ("sign_flip", 1), ("sign_flip", 2), ("scaled_flip", 1),
           ("poison", 1), ("poison", 2), ("adaptive", 1), ("adaptive", 2)]
DEFENCES = ["trust", "fedavg", "median", "krum", "fltrust"]
ADAPTIVE_COSINE = 0.6
SWEEP_COSINES = [0.4, 0.2, 0.1, 0.05, 0.0]
# Cosine with the root update. Honest clients sit around 0.02-0.4 in the logs, so even 0.05 keeps
# a positive FLTrust weight while leaving most of the update free for poison.
ROOTAWARE_COSINES = [0.6, 0.3, 0.1, 0.05, 0.02]
POISON_FLOWS = 20000


def flat(d):
    return torch.cat([v.flatten() for v in d.values()])


def unflat(v, like):
    out, i = {}, 0
    for k, t in like.items():
        out[k] = v[i:i + t.numel()].view_as(t).clone()
        i += t.numel()
    return out


def poison_delta(ae, g, pool_x, rng):
    """The honest recipe on attack flows, with nothing routed away."""
    x = torch.from_numpy(pool_x[rng.integers(len(pool_x), size=POISON_FLOWS)])
    ae.load_state_dict(g)
    ae.train()
    export, _, _ = train_ae_pass(ae, torch.optim.Adam(ae.parameters(), lr=cfg.AE_LEARNING_RATE), x,
                                 cfg.AE_BATCH_SIZE, cycle_to_horizon=True)
    return {k: export[k] - g[k] for k in g}


def attack(kind, honest, ae, g, pool_x, rng, root=None):
    if kind == "sign_flip":
        return {k: -v for k, v in honest.items()}
    if kind == "scaled_flip":
        return {k: -10 * v for k, v in honest.items()}
    p = poison_delta(ae, g, pool_x, rng)
    if kind == "poison":
        return p
    h, pv = flat(honest), flat(p)
    # rootaware knows the server's root update, so it sets its cosine with the very
    # vector FLTrust scores against; adaptive only has its own honest update to aim at.
    u = flat(root) / flat(root).norm() if kind.startswith("rootaware") else h / h.norm()
    orth = pv - (pv @ u) * u
    cos = float(kind.split("@")[1]) if "@" in kind else ADAPTIVE_COSINE
    mixed = cos * u + np.sqrt(1 - cos ** 2) * orth / orth.norm()
    return unflat(mixed * h.norm(), honest)


def aggregate(defence, deltas, g, root):
    if defence == "fedavg":
        return c._mean_delta(deltas, g), None
    if defence == "median":
        return {k: torch.stack([d[k] for d in deltas]).median(0).values for k in deltas[0]}, None
    if defence == "fltrust":
        agg, scores = ae_only_fltrust_aggregate(deltas, root)
        return agg, [s == 0 for s in scores]
    if defence in ("krum", "krum_f1"):
        keep = krum_select([[v.numpy() for v in d.values()] for d in deltas], num_to_select=3,
                           f=1 if defence == "krum_f1" else None)
        return c._mean_delta([deltas[i] for i in keep], g), [i not in keep for i in range(len(deltas))]
    agg, _, ch1, _, _, flags = dc_fltrust_aggregate(
        deltas, [NO_HEAD] * len(deltas), root, NO_HEAD, client_round_counts=[0] * len(deltas),
        ch2_warmup_rounds=cfg.CH2_WARMUP_ROUNDS, current_round=1, ch1_threshold=cfg.FLTRUST_CH1_THRESHOLD)
    return agg, flags


def federate(data, pool_x, defence, kind, n_att, seed, rounds, return_state=False):
    torch.manual_seed(cfg.SEED + seed)
    rng = np.random.default_rng(cfg.SEED + seed)
    g = fl_round.snapshot(copy.deepcopy(data.g0).autoencoder)
    flows = client_flows(data, pool_x, 0.0, seed, 10 ** 9)
    attackers = ATTACKERS.get(n_att, []) if kind != "none" else []
    root_benign = torch.from_numpy(data.root["x"][data.root["y"] == 0])
    ae = FlowAutoencoder()
    excluded = []
    for _ in range(rounds):
        # Only the root-aware attack computes the root first; every other attack keeps the
        # order the committed results (test_split.json included) were produced in.
        root = root_ae_delta(g, root_benign) if kind.startswith("rootaware") else None
        deltas = []
        for i, (x, y) in enumerate(flows):
            d, _ = local_ae_delta(ae, g, x, y, True, cfg.AE_LEARNING_RATE)
            deltas.append(attack(kind, d, ae, g, pool_x, rng, root) if i in attackers else d)
        if root is None and defence in ("trust", "fltrust"):
            root = root_ae_delta(g, root_benign)
        agg, flags = aggregate(defence, deltas, g, root)
        if flags is not None:
            excluded.append(flags)
        g = fl_round.apply_delta(g, agg)
    ae.load_state_dict(g)
    ae.eval()
    y = data.ev["y"] == 1
    _, mse_ev = c.encode(ae, data.ev["x"])
    _, mse_th = c.encode(ae, data.thr["x"])
    res = {"recall_1pct": float((mse_ev > np.percentile(mse_th, 99))[y].mean()), "auroc": float(roc_auc_score(y, mse_ev))}
    if excluded:
        e = np.mean(excluded, axis=0)
        honest = [i for i in range(len(flows)) if i not in attackers]
        res["attackers_excluded"] = float(e[attackers].mean()) if attackers else None
        res["honest_excluded"] = float(e[honest].mean())
    return (res, g) if return_state else res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--defences", help="comma-separated subset, e.g. fltrust")
    ap.add_argument("--rootaware", action="store_true", help="attackers who know the server's root update")
    ap.add_argument("--krum-control", action="store_true", help="Krum with f=1 against one attacker")
    args = ap.parse_args()
    rounds, attacks, defences = ROUNDS, ATTACKS, DEFENCES
    if args.quick:
        rounds, args.seeds, attacks = 2, 1, [("none", 0), ("scaled_flip", 1), ("adaptive", 2)]
        if args.rootaware:
            attacks = [("rootaware@0.1", 2)]
    out_path = ROOT / "results" / "wave3_byzantine.json"
    if args.sweep:
        attacks, defences = [(f"adaptive@{x}", 2) for x in SWEEP_COSINES], ["trust", "fedavg", "median"]
        out_path = ROOT / "results" / "wave3_adaptive_sweep.json"
    if args.rootaware:
        if not args.quick:
            attacks = [(f"rootaware@{x}", 2) for x in ROOTAWARE_COSINES]
        defences = ["fltrust", "trust", "fedavg", "median"]
        out_path = ROOT / "results" / "wave3_rootaware.json"
    if args.krum_control:
        attacks = [("none", 0), ("sign_flip", 1), ("scaled_flip", 1), ("poison", 1), ("adaptive@0.0", 1)]
        defences = ["krum_f1", "krum"]
        out_path = ROOT / "results" / "wave3_krum_control.json"
    if args.defences:
        defences = args.defences.split(",")
    c.CLIENT_FLOWS = 10 ** 9
    torch.set_num_threads(4)
    data = c.Data()
    pool_x = data.train["x"][data.train["y"] == 1]
    log.info(f"routing cut {cfg.CH2_MSE_SPLIT_THRESHOLD:.5f}; ch1 threshold {cfg.FLTRUST_CH1_THRESHOLD}; attackers {ATTACKERS}")
    res = {"rounds": rounds, "seeds": args.seeds, "routing_cut": cfg.CH2_MSE_SPLIT_THRESHOLD,
           "ch1_threshold": cfg.FLTRUST_CH1_THRESHOLD, "adaptive_cosine": ADAPTIVE_COSINE, "runs": {}}
    if not args.quick and out_path.exists():
        done = json.loads(out_path.read_text(encoding="utf-8"))
        if all(done.get(k) == res[k] for k in ("rounds", "seeds", "routing_cut", "ch1_threshold", "adaptive_cosine")):
            res["runs"] = done["runs"]
            log.info(f"resuming after {len(res['runs'])} finished runs")
    t0 = time.time()
    for kind, n_att in attacks:
        for defence in defences:
            if f"{defence}|{kind}|{n_att}" in res["runs"]:
                continue
            rows = [federate(data, pool_x, defence, kind, n_att, s, rounds) for s in range(args.seeds)]
            key = f"{defence}|{kind}|{n_att}"
            row = res["runs"][key] = {k: (None if rows[0][k] is None else float(np.mean([r[k] for r in rows]))) for k in rows[0]}
            row["recall_1pct_std"] = float(np.std([r["recall_1pct"] for r in rows]))
            log.info(f"{key}: {({k: (None if v is None else round(v, 3)) for k, v in row.items()})} ({time.time() - t0:.0f}s)")
            if not args.quick:
                out_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    log.info("DONE")


if __name__ == "__main__":
    main()
