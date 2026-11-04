#!/usr/bin/env python3
"""
scripts/ch2_cross_dataset.py — repeat the channel 2 conditions on NF-ToN-IoT-v3
===============================================================================
Reads a seeded 10% sample of the archive in dataset/ton/ and reuses the experiment code of
ch2_conditions.py, writing to results/ton/. UNSW results are not overwritten.

Fixed before any ToN result was seen:
  shortcuts repeat    a single feature reaches AUROC >= 0.95, or LightGBM reaches recall >= 0.95
                      on classes it never trained on
  transfer works      trained on one dataset, the other scores AUROC >= 0.90 AND recall >= 0.50
                      at 1% FPR (threshold taken from the target's own benign flows, which is
                      generous: a real deployment would not have it on day one)
  head adds detection on all features, AE + labelled head beats the AE alone by >= 0.05 recall
                      at the same false-positive rate
  channel 2 inverts   adding channel 2 raises the attack-free organisation's exclusion rate
                      by >= 0.20 over channel 1 alone

Steps: audit, cross, e2, e6, e8, e5, e7. Run one at a time; the machine is short of memory.

--keep-dups: removing training rows repeated in other splits took the ToN training attack share from
39% to 74% and left the AE at AUROC 0.51, so results/ton/ does not test the AE or routing. This flag
keeps the realistic mix. Repeated vectors then cross splits, so the audit also scores unseen vectors.

Usage:
    python scripts/ch2_cross_dataset.py
    python scripts/ch2_cross_dataset.py --keep-dups
    python scripts/ch2_cross_dataset.py --steps audit,cross
    python scripts/ch2_cross_dataset.py --quick      # wiring check, writes no results
"""

import argparse
import json
import logging
import sys
import time
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import config as cfg
from aura import dataset as ds
from aura.models import AttackHead
from baselines import fit_ae, fit_head
import ch2_conditions as c

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("ch2_cross_dataset")

RESULTS = ROOT / "results" / "ton"
SAMPLE_FILE = cfg.CACHE_DIR / "ton_sample_10pct.npz"
SAMPLE_RATE = 0.10
FEATURE_COLUMNS = list(ds._TABLE_FEATURE_COLUMNS)
MAX_TRAIN, MAX_AE, MAX_EVAL, MAX_THR, MAX_ROOT_ATTACKS = 500_000, 300_000, 100_000, 30_000, 5_000


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def build_sample():
    """Stream the CSV out of the archive and keep a seeded random sample of rows."""
    if SAMPLE_FILE.exists():
        return
    archive = next((cfg.CSV_DIR / "ton").glob("*.zip"))
    rng = np.random.default_rng(cfg.SEED)
    parts, total = [], 0
    t0 = time.time()
    use = FEATURE_COLUMNS + ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "Label", "Attack"]
    with zipfile.ZipFile(archive) as z:
        inner = next(n for n in z.namelist() if n.endswith("NF-ToN-IoT-v3.csv"))
        with z.open(inner) as f:
            for chunk in pd.read_csv(f, usecols=use, dtype={k: np.float32 for k in FEATURE_COLUMNS}, chunksize=1_000_000):
                total += len(chunk)
                parts.append(chunk[rng.random(len(chunk)) < SAMPLE_RATE])
                log.info(f"[sample] read {total:,} rows ({time.time() - t0:.0f}s)")
    df = pd.concat(parts, ignore_index=True)
    del parts
    df.sort_values(["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"], kind="stable", inplace=True, ignore_index=True)
    raw = df[FEATURE_COLUMNS].to_numpy(np.float32)
    raw[~np.isfinite(raw)] = 0.0
    attack = df.Attack.fillna("Benign").astype(str).str.strip()
    attack[df.Label.values == 0] = "Benign"
    names = ["Benign"] + sorted(set(attack) - {"Benign"})
    np.savez(SAMPLE_FILE, raw=raw, y=df.Label.to_numpy(np.int8), c=attack.map({n: i for i, n in enumerate(names)}).to_numpy(np.int8),
             o=np.array([ds._org_of(ip) for ip in df.IPV4_DST_ADDR], np.int8), names=np.array(names), total=total,
             hosts=len(set(df.IPV4_SRC_ADDR) | set(df.IPV4_DST_ADDR)))
    log.info(f"[sample] kept {len(df):,} of {total:,} flows ({time.time() - t0:.0f}s)")


class ToNData:
    """The fields ch2_conditions.py expects from its Data class, built from the ToN sample."""

    def __init__(self, keep_dups=False):
        s = np.load(SAMPLE_FILE, allow_pickle=False)
        self.raw, self.y, self.c, self.o = s["raw"], s["y"], s["c"], s["o"].copy()
        self.class_names = [str(n) for n in s["names"]]
        self.total, self.hosts = int(s["total"]), int(s["hosts"])
        self.table_feature_names = list(cfg.FEATURE_NAMES)
        rng = np.random.default_rng(cfg.SEED)

        # Blocks of neighbouring flows, as in the UNSW split, with no feature vector crossing splits
        blocks = rng.choice(3, size=len(self.y) // 1000 + 1, p=[.7, .1, .2])[np.arange(len(self.y)) // 1000]
        keys = pd.util.hash_pandas_object(pd.DataFrame(self.raw), index=False).to_numpy()
        self.keys = keys
        self.duplicate_rate = float(1 - len(np.unique(keys)) / len(keys))
        # 92% of benign rows repeat, so removing crossings strips ordinary benign traffic from training
        kept_calib = np.zeros(len(keys), bool) if keep_dups else np.isin(keys, keys[blocks == 2])
        kept_root = np.zeros(len(keys), bool) if keep_dups else np.isin(keys, keys[blocks == 1])
        idx = {"calib": np.flatnonzero(blocks == 2),
               "root": np.flatnonzero((blocks == 1) & ~kept_calib),
               "train": np.flatnonzero((blocks == 0) & ~kept_calib & ~kept_root)}
        del kept_calib, kept_root

        # Same situation as UNSW: one organisation sees no attacks. It takes the name "bank".
        counts = np.bincount(self.o[self.y == 1], minlength=len(ds.ORG_NAMES))
        quietest, bank = int(counts.argmin()), ds.ORG_NAMES.index("bank")
        self.o = np.where(self.o == quietest, bank, np.where(self.o == bank, quietest, self.o)).astype(np.int8)
        self.attacks_by_org = {ds.ORG_NAMES[i]: int(n) for i, n in enumerate(np.bincount(self.o[self.y == 1], minlength=len(ds.ORG_NAMES)))}

        train = rng.permutation(idx["train"])[:MAX_TRAIN]
        benign_all = idx["train"][self.y[idx["train"]] == 0]
        benign = rng.permutation(benign_all)[:MAX_AE]
        self.prep = ds.FlowPreprocessor()
        fit_rows = rng.permutation(benign)[:200_000]
        inlier = self.prep.fit(self.raw[fit_rows], FEATURE_COLUMNS)
        self.if_baseline = self.prep.sanitiser

        self.train = self.fetch(train, self.prep)
        self.clients = []
        for o in range(len(ds.ORG_NAMES)):
            pick = rng.permutation(np.flatnonzero(self.train["o"] == o))[:c.CLIENT_FLOWS]
            client = {k: v[pick] for k, v in self.train.items()}
            if o == bank:
                client = {k: v[client["y"] == 0] for k, v in client.items()}
            self.clients.append(client)
        root = self.fetch(idx["root"], self.prep)
        r_b = rng.permutation(np.flatnonzero(root["y"] == 0))[:cfg.FLTRUST_ROOT_SAMPLES]
        r_a = rng.permutation(np.flatnonzero(root["y"] == 1))[:MAX_ROOT_ATTACKS]
        self.root = {k: v[np.r_[r_b, r_a]] for k, v in root.items()}
        calib = rng.permutation(idx["calib"])
        half = rng.random(len(calib)) < 0.5
        self.thr_idx = calib[half & (self.y[calib] == 0)][:MAX_THR]
        self.ev_idx = calib[~half][:MAX_EVAL]
        self.thr = self.fetch(self.thr_idx, self.prep)
        self.ev = self.fetch(self.ev_idx, self.prep)
        self.train_idx, self.benign_idx = train, benign
        self.train_benign_x = np.asarray(self.prep.transform(self.raw[benign], FEATURE_COLUMNS), np.float32)

        # A starting model made the way G0 is: AE on sanitised benign flows, untrained head
        keep = self.prep.is_inlier(self.raw[benign], FEATURE_COLUMNS)
        epochs = 2 if c.QUICK else 15
        self.g0 = types.SimpleNamespace(autoencoder=fit_ae(self.train_benign_x[keep], epochs, cfg.SEED), attack_head=AttackHead())
        _, thr_mse = c.encode(self.g0.autoencoder, self.thr["x"])
        self.split_threshold = float(np.percentile(thr_mse, 75))
        log.info(f"ToN: {len(self.y):,} sampled of {self.total:,}; duplicates {self.duplicate_rate:.3f}; attacks by org {self.attacks_by_org}; "
                 f"clients {[(len(x['y']), int(x['y'].sum())) for x in self.clients]}; MSE split {self.split_threshold:.5f} "
                 f"(fit on {int(inlier.sum()):,} inliers)")

    def fetch(self, idx, prep):
        return dict(x=np.asarray(prep.transform(self.raw[idx], FEATURE_COLUMNS), np.float32), y=self.y[idx], c=self.c[idx], o=self.o[idx])


# ─────────────────────────────────────────────────────────────────────────────
# Audit: do the UNSW shortcuts repeat?
# ─────────────────────────────────────────────────────────────────────────────

def run_audit(td):
    sample = np.random.default_rng(cfg.SEED).permutation(td.train_idx)[:200_000]
    y = td.y[sample]
    rows = []
    for j, name in enumerate(FEATURE_COLUMNS):
        col = td.raw[sample, j]
        a = roc_auc_score(y, col) if col.min() < col.max() else 0.5
        rows.append({"feature": name, "auroc": round(float(max(a, 1 - a)), 4)})
    rows.sort(key=lambda r: -r["auroc"])
    # IsolationForest baseline, on the same log-scaled features it was fitted on
    logc = td.prep._log_continuous(td.prep._select(td.raw[td.ev_idx], FEATURE_COLUMNS))
    logt = td.prep._log_continuous(td.prep._select(td.raw[td.thr_idx], FEATURE_COLUMNS))
    if_ev, if_thr = -td.if_baseline.score_samples(logc), -td.if_baseline.score_samples(logt)
    y_ev = td.ev["y"] == 1
    _, mse_ev = c.encode(td.g0.autoencoder, td.ev["x"])
    _, mse_thr = c.encode(td.g0.autoencoder, td.thr["x"])
    # The same detector on flows whose exact vector was not in its training data
    new_ev = ~np.isin(td.keys[td.ev_idx], td.keys[td.benign_idx])
    new_thr = ~np.isin(td.keys[td.thr_idx], td.keys[td.benign_idx])
    return {"flows_in_archive": td.total, "flows_sampled": len(td.y), "hosts_in_sample": td.hosts,
            "attack_rate": round(float(td.y.mean()), 4), "duplicate_rate": round(td.duplicate_rate, 4),
            "train_attack_share": round(float(td.train["y"].mean()), 4),
            "ae_unseen_vectors": {"benign": int((new_ev & ~y_ev).sum()), "attacks": int((new_ev & y_ev).sum()),
                                  "auroc": float(roc_auc_score(y_ev[new_ev], mse_ev[new_ev])),
                                  "recall_1pct": float((mse_ev[new_ev & y_ev] > np.percentile(mse_thr[new_thr], 99)).mean()),
                                  "fpr_1pct": float((mse_ev[new_ev & ~y_ev] > np.percentile(mse_thr[new_thr], 99)).mean())},
            "class_counts": {n: int((td.c == i).sum()) for i, n in enumerate(td.class_names)},
            "attacks_by_org": td.attacks_by_org, "single_feature_top": rows[:15],
            "isolation_forest": {"auroc": float(roc_auc_score(y_ev, if_ev)), "recall_1pct": float((if_ev > np.percentile(if_thr, 99))[y_ev].mean())},
            "ae": {"auroc": float(roc_auc_score(y_ev, mse_ev)), "recall_1pct": float((mse_ev > np.percentile(mse_thr, 99))[y_ev].mean()),
                   "recall_2pct": float((mse_ev > np.percentile(mse_thr, 98))[y_ev].mean())}}


# ─────────────────────────────────────────────────────────────────────────────
# Cross-dataset transfer
# ─────────────────────────────────────────────────────────────────────────────

def score_block(score_ev, score_thr, y, classes, names):
    pred = score_ev > np.percentile(score_thr, 99)
    per_class = {names[k]: round(float(pred[y & (classes == k)].mean()), 3) for k in np.unique(classes[y])}
    return {"auroc": round(float(roc_auc_score(y, score_ev)), 4), "recall_1pct": round(float(pred[y].mean()), 4),
            "fpr": round(float(pred[~y].mean()), 4), "per_class": per_class}


def transfer(train, benign_x, ae_fixed, ev, thr, names, seeds):
    """Train on one dataset, score the other, in the UNSW preprocessor's feature scale."""
    from lightgbm import LGBMClassifier
    y = ev["y"] == 1
    out = {"lightgbm": [], "ae": [], "head_labelled": []}
    lgb = LGBMClassifier(n_estimators=100, n_jobs=4, verbose=-1, random_state=cfg.SEED).fit(train["x"], train["y"])
    out["lightgbm"] = score_block(lgb.predict_proba(ev["x"])[:, 1], lgb.predict_proba(thr["x"])[:, 1], y, ev["c"], names)
    ae_rows, head_rows = [], []
    for s in range(seeds):
        ae = ae_fixed if ae_fixed is not None and s == 0 else fit_ae(benign_x, 2 if c.QUICK else 15, cfg.SEED + s)
        tr = c.with_encoding(ae, train)
        e, t = c.with_encoding(ae, ev), c.with_encoding(ae, thr)
        ae_rows.append(score_block(e["mse"], t["mse"], y, ev["c"], names))
        h = fit_head(tr["z"], tr["y"].astype(np.float32), 2 if c.QUICK else c.SUPERVISED_EPOCHS, cfg.SEED + s)
        head_rows.append(score_block(c.scores(h, e["z"]), c.scores(h, t["z"]), y, ev["c"], names))
    for key, rows in (("ae", ae_rows), ("head_labelled", head_rows)):
        out[key] = {**c.mean_of(rows), "recall_std": c.std_of(rows, "recall_1pct"), "per_class_seed0": rows[0]["per_class"]}
    return out


def run_cross(td, seeds):
    ud = c.Data()
    prep_u = ds.load_preprocessor()
    t_train = td.fetch(td.train_idx, prep_u)
    t_ev, t_thr = td.fetch(td.ev_idx, prep_u), td.fetch(td.thr_idx, prep_u)
    t_benign = np.asarray(prep_u.transform(td.raw[td.benign_idx], FEATURE_COLUMNS), np.float32)
    rng = np.random.default_rng(cfg.SEED)
    u_pick = rng.permutation(len(ud.train["y"]))[:MAX_TRAIN]
    u_train = {k: v[u_pick] for k, v in ud.train.items()}
    res = {"unsw_to_ton": transfer(u_train, ud.train_benign_x, ud.g0.autoencoder.eval(), t_ev, t_thr, td.class_names, seeds),
           "ton_to_unsw": transfer(t_train, t_benign, None, ud.ev, ud.thr, ud.class_names, seeds)}
    log.info(f"[cross] {json.dumps({d: {m: {k: v[k] for k in ('auroc', 'recall_1pct')} for m, v in r.items()} for d, r in res.items()})}")
    return res


# ─────────────────────────────────────────────────────────────────────────────

def write(name, res):
    if not c.QUICK:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="audit,cross,e2,e6,e8,e5,e7")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--keep-dups", action="store_true", help="do not remove training rows repeated in other splits; writes to results/ton_keep_dups")
    args = ap.parse_args()
    if args.quick:
        c.QUICK, c.ROUNDS, args.seeds = True, 2, 1
    torch.set_num_threads(4)
    global RESULTS
    if args.keep_dups:
        RESULTS = ROOT / "results" / "ton_keep_dups"
    RESULTS.mkdir(parents=True, exist_ok=True)
    c.RESULTS = RESULTS       # keeps the UNSW result files untouched
    build_sample()
    td = ToNData(keep_dups=args.keep_dups)
    cfg.CH2_MSE_SPLIT_THRESHOLD = td.split_threshold
    for step in args.steps.split(","):
        t0 = time.time()
        if step == "audit":
            res = run_audit(td)
            write("audit", res)
        elif step == "cross":
            res = run_cross(td, args.seeds)
            write("cross", res)
        else:
            if step == "e6" and c.QUICK:
                continue      # needs the feature ranking a full e2 run saves
            res = {"e2": c.run_e2, "e5": c.run_e5, "e6": c.run_e6, "e7": c.run_e7, "e8": c.run_e8}[step](td, args.seeds)
        log.info(f"=== {step} finished in {time.time() - t0:.0f}s ===")
        if c.QUICK:
            log.info(json.dumps(res, default=str)[:500])
    log.info("DONE")


if __name__ == "__main__":
    main()
