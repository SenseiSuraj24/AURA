#!/usr/bin/env python3
"""
scripts/ch2_zeroday.py — can channel 2 tell the server that an organisation is seeing a NEW attack type?
=======================================================================================================
Channel 2 is the cosine between a client's AttackHead update and the server's root head update.
Both heads train towards "output 1" on flows above the routing cut, so the update's direction is
set by where those flows sit in the AE latent space. The root head trains on the server's known
attack examples. So channel 2 asks: do the flows this organisation flags look like attacks the
server already knows? ch2_alarm.py asked a different question ("is it under attack at all?"),
which the client's own high-error share answers better. This asks "is the attack new?".

Design. For each attack class X with enough examples in the server's root set:
  known    the attack-free bank gets a share BURDEN of its flows replaced with training-pool
           flows of class X; the server's root attack set is complete.
  unseen   the same injection, same seeds; class X is removed from the server's root attack set,
           so X is a zero-day from the server's point of view.
  clean    no injection, root without X (does removing X alone move the bank's score?).
The only difference between known and unseen is whether the server knows X. Production option A
aggregation with no exclusion on either channel, so the federation itself does not depend on the
scores being measured. The score is the bank's channel 2 minus the median of the other four that
round (as in trust_leak.py), so a root change that moves everyone does not count.

Predictions, fixed before running:
  1. Pooled over classes, seeds and rounds, the bank's relative channel 2 is lower under unseen
     than under known: AUROC >= 0.8 for "unseen" with lower score = more novel.
  2. The per-class mean difference (unseen - known) is negative for at least 2/3 of the classes.
  3. Removing X with no injection moves the bank's relative channel 2 by less than half the
     unseen-known gap (the effect needs the traffic, not just the root change).
If 1 fails, channel 2 is not a zero-day signal on UNSW at this burden.

Calibration split only; the test split stays sealed. UNSW only for now.

Usage:
    python scripts/ch2_zeroday.py
    python scripts/ch2_zeroday.py --quick      # wiring check, writes no results
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
from aura.fl_server import dc_fltrust_aggregate
from aura.models import AttackHead, FlowAutoencoder
import ch2_conditions as c
from ch2_alarm import pools_of

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("ch2_zeroday")

ROUNDS, BURDEN, MIN_ROOT, MIN_POOL = 10, 0.2, 20, 200
BANK = ds.ORG_NAMES.index("bank")


def federate(data, pools, inject_x, root_attack, seed, rounds):
    """Raw channel 2 per round and client, with the bank's flows partly replaced by inject_x."""
    torch.manual_seed(cfg.SEED + seed)
    rng = np.random.default_rng(cfg.SEED + seed)
    g = copy.deepcopy(data.g0)
    g_ae, g_head = fl_round.snapshot(g.autoencoder), fl_round.snapshot(g.attack_head)
    root_benign = torch.from_numpy(data.root["x"][data.root["y"] == 0])
    root_attack = torch.from_numpy(root_attack)
    ae, head = FlowAutoencoder(), AttackHead()
    ch2 = np.full((rounds, len(pools)), np.nan)
    high = np.zeros(rounds)
    saved = cfg.CH2_ANTI_ALIGN_THRESHOLD
    cfg.CH2_ANTI_ALIGN_THRESHOLD = -2.0
    try:
        for rnd in range(rounds):
            root_ae, root_head = fl_round.root_reference(g_ae, g_head, root_benign, root_attack)
            ae_ds, head_ds = [], []
            for i, pool in enumerate(pools):
                pick = rng.integers(len(pool["y"]), size=c.CLIENT_FLOWS)
                flows = pool["x"][pick].copy()
                if i == BANK and inject_x is not None:
                    n = int(len(flows) * BURDEN)
                    flows[:n] = inject_x[rng.integers(len(inject_x), size=n)]
                u = fl_round.client_update(ae, head, g_ae, g_head, torch.from_numpy(flows))
                ae_ds.append(u.ae_delta)
                head_ds.append(u.head_delta)
                if i == BANK:
                    high[rnd] = u.n_high_mse / max(u.n_benign + u.n_high_mse, 1)
            agg_ae, agg_head, *_ = dc_fltrust_aggregate(
                ae_ds, head_ds, root_ae, root_head, [rnd + 1] * len(pools), ch2_warmup_rounds=0,
                current_round=rnd + 1, ch1_threshold=-1.0)
            ch2[rnd] = c._ch2_raw(head_ds, root_head)
            g_ae, g_head = fl_round.apply_delta(g_ae, agg_ae), fl_round.apply_delta(g_head, agg_head)
    finally:
        cfg.CH2_ANTI_ALIGN_THRESHOLD = saved
    rel = ch2[:, BANK] - np.nanmedian(np.delete(ch2, BANK, axis=1), axis=1)
    return {"ch2": ch2.tolist(), "relative": rel.tolist(), "bank_high_share": high.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rounds = ROUNDS
    if args.quick:
        rounds, args.seeds = 2, 1
    torch.set_num_threads(4)
    data = c.Data()
    pools = pools_of(data)
    train_att = (data.train["y"] == 1)
    root_att = data.root["y"] == 1
    counts = {data.class_names[k]: (int((data.root["c"][root_att] == k).sum()), int((data.train["c"][train_att] == k).sum()))
              for k in np.unique(data.train["c"][train_att])}
    log.info(f"class: (root flows, training-pool flows) {counts}")
    classes = [k for k in np.unique(data.train["c"][train_att])
               if (data.root["c"][root_att] == k).sum() >= MIN_ROOT and (data.train["c"][train_att] == k).sum() >= MIN_POOL]
    if args.quick:
        classes = classes[:1]
    log.info(f"testing {[data.class_names[k] for k in classes]}")

    full_root = data.root["x"][root_att]
    out = ROOT / "results" / "ch2_zeroday.json"
    res = {"rounds": rounds, "seeds": args.seeds, "burden": BURDEN, "root_and_pool_counts": counts, "classes": {}}
    t0 = time.time()
    base = [federate(data, pools, None, full_root, s, rounds) for s in range(args.seeds)]
    res["clean_full_root"] = float(np.nanmean([r["relative"] for r in base]))
    log.info(f"clean bank, full root: relative ch2 {res['clean_full_root']:+.3f} ({time.time() - t0:.0f}s)")
    for k in classes:
        name = data.class_names[k]
        inject = data.train["x"][train_att & (data.train["c"] == k)]
        root_wo = data.root["x"][root_att & (data.root["c"] != k)]
        runs = {cond: [federate(data, pools, inj, root, s, rounds) for s in range(args.seeds)]
                for cond, inj, root in (("known", inject, full_root), ("unseen", inject, root_wo), ("clean", None, root_wo))}
        row = {cond: float(np.nanmean([r["relative"] for r in rs])) for cond, rs in runs.items()}
        row["gap"] = row["unseen"] - row["known"]
        row["high_share"] = float(np.mean([r["bank_high_share"] for r in runs["known"]]))
        res["classes"][name] = {"summary": row, "runs": runs}
        log.info(f"{name}: known {row['known']:+.3f} unseen {row['unseen']:+.3f} clean {row['clean']:+.3f} "
                 f"gap {row['gap']:+.3f}; bank high share {row['high_share']:.3f} ({time.time() - t0:.0f}s)")
        if not args.quick:
            out.write_text(json.dumps(res, indent=1), encoding="utf-8")

    known = np.concatenate([np.ravel([r["relative"] for r in v["runs"]["known"]]) for v in res["classes"].values()])
    unseen = np.concatenate([np.ravel([r["relative"] for r in v["runs"]["unseen"]]) for v in res["classes"].values()])
    ok = ~np.isnan(known) & ~np.isnan(unseen)
    y = np.r_[np.zeros(ok.sum()), np.ones(ok.sum())]
    auroc = float(roc_auc_score(y, -np.r_[known[ok], unseen[ok]])) if ok.sum() > 1 else None
    gaps = [v["summary"]["gap"] for v in res["classes"].values()]
    root_only = [abs(v["summary"]["clean"] - res["clean_full_root"]) for v in res["classes"].values()]
    res["verdict"] = {
        "p3_root_change_alone_small": bool(gaps) and all(r < abs(g) / 2 for r, g in zip(root_only, gaps)),
        "auroc_unseen_vs_known": auroc,
        "p1_auroc_ok": auroc is not None and auroc >= 0.8,
        "share_classes_negative_gap": float(np.mean([g < 0 for g in gaps])) if gaps else None,
        "p2_ok": bool(gaps) and np.mean([g < 0 for g in gaps]) >= 2 / 3,
    }
    log.info(f"verdict {res['verdict']}")
    if not args.quick:
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    log.info("DONE")


if __name__ == "__main__":
    main()
