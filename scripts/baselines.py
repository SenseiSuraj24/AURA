#!/usr/bin/env python3
"""
scripts/baselines.py — Wave 1.5 baselines and go/no-go
=======================================================

What can AURA honestly claim on NF-UNSW-NB15-v3? Every number here is
measured on the canonical, de-duplicated TEST flows; every threshold comes
from the disjoint CALIBRATION flows; nothing is written to saved_models/.

  1. Shortcuts      TTL rule (reference) and the strongest single features
                    in the TTL-free feature space.
  2. Detectors      IsolationForest, supervised LightGBM, and the autoencoder
                    trained centrally / locally per org / federated (FedAvg,
                    honest clients) — one shared training recipe — with TTL
                    features off (primary) and on (secondary).
  3. Zero-day       leave-one-attack-class-out for the supervised model.
  4. Federation     non-IID class hold-out: each attacked org hides some
                    attack classes from its own training data; an AttackHead
                    on AE latents is trained locally / FedAvg / centrally and
                    scored on each org's HIDDEN classes.

Metrics: AUROC and PR-AUC on test flows; recall and test FPR at the
threshold where calibration benign FPR = 1% (P99 of calibration scores).

Usage:
    python scripts/baselines.py                  # full run (TTL off + on)
    python scripts/baselines.py --quick          # fewer epochs/seeds, smoke run
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
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg
from aura import dataset as ds
from aura.models import AttackHead, FlowAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("baselines")
RESULTS_DIR = ROOT / "results"


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

# Traffic-generator fingerprint (measured 2026-10): benign flows have a minimum
# IP packet length of 52 bytes (TCP with options) in 98.4% of cases; 91.4% of
# attack flows — in every class — have <= 40 (no TCP options, IXIA generator).
GENERATOR_SHORTCUTS = ["MIN_IP_PKT_LEN", "SHORTEST_FLOW_PKT"]


def make_preprocessor(use_ttl: bool, table, split, drop=()) -> ds.FlowPreprocessor:
    """Fit a FlowPreprocessor on training benign flows with TTL on/off and columns dropped."""
    if use_ttl == cfg.USE_TTL_FEATURES and not drop:
        return ds.load_preprocessor()
    prep = ds.FlowPreprocessor()
    base = [c for c in cfg.CONTINUOUS_FEATURES if c not in cfg.TTL_FEATURES and c not in drop]
    prep.continuous = base + (cfg.TTL_FEATURES if use_ttl else [])
    prep.raw_columns = prep.continuous + cfg.TCP_FLAG_FEATURES + list(cfg.CATEGORICAL_FEATURES)
    prep.feature_names = [c.lower() for c in prep.continuous] + [
        n for n in cfg.FEATURE_NAMES if n not in {c.lower() for c in cfg.CONTINUOUS_FEATURES}]
    prep.use_ttl = use_ttl
    benign = split.benign_flows("train", table)
    prep.fit(table.raw[benign], table.raw_columns)
    return prep


class Features:
    """Transformed feature rows for the flow sets used here (in memory)."""

    def __init__(self, prep, table, split):
        X = ds.feature_matrix(prep, table)
        self.names = prep.feature_names
        self.table = table
        benign_train = split.benign_flows("train", table)
        inlier = prep.is_inlier(table.raw[benign_train], table.raw_columns)
        self.sets = {
            "train":        split.flows["train"],
            "train_benign": benign_train[inlier],      # same IF-cleaned rule as train.py
            "calib_benign": split.benign_flows("calib", table),
            "test":         split.flows["test"],
        }
        self.X = {k: np.asarray(X[v], np.float32) for k, v in self.sets.items()}
        lab = np.asarray(table.labels)
        cls = np.asarray(table.attack_class)
        org = np.asarray(table.dst_org)
        self.y = {k: lab[v] for k, v in self.sets.items()}
        self.cls = {k: cls[v] for k, v in self.sets.items()}
        self.org = {k: org[v] for k, v in self.sets.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(score_test, y_test, cls_test, score_calib_benign, class_names, mask=None):
    """AUROC / PR-AUC, and recall + FPR at the calibration-P99 threshold."""
    if mask is not None:
        score_test, y_test, cls_test = score_test[mask], y_test[mask], cls_test[mask]
    thr = float(np.percentile(score_calib_benign, 99))
    pred = score_test > thr
    out = {"n_test": int(len(y_test)), "n_attack": int(y_test.sum())}
    if 0 < y_test.sum() < len(y_test):
        out["auroc"] = round(float(roc_auc_score(y_test, score_test)), 4)
        out["pr_auc"] = round(float(average_precision_score(y_test, score_test)), 4)
    out["recall@calibFPR1%"] = round(float(pred[y_test == 1].mean()), 4) if y_test.sum() else None
    out["test_fpr"] = round(float(pred[y_test == 0].mean()), 4)
    out["per_class_recall"] = {
        class_names[c]: round(float(pred[cls_test == c].mean()), 3)
        for c in np.unique(cls_test[y_test == 1])
    }
    return out


def summarise(runs: list) -> dict:
    """Mean ± std of scalar metrics across seeds; per-class recall averaged."""
    keys = [k for k in ("auroc", "pr_auc", "recall@calibFPR1%", "test_fpr")
            if runs[0].get(k) is not None]          # undefined e.g. for an org with no attacks
    out = {k: {"mean": round(float(np.mean([r[k] for r in runs])), 4),
               "std": round(float(np.std([r[k] for r in runs])), 4)} for k in keys}
    out["per_class_recall"] = {c: round(float(np.mean([r["per_class_recall"].get(c, np.nan)
                                                       for r in runs])), 3)
                               for c in runs[0]["per_class_recall"]}
    out["n_seeds"] = len(runs)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Models (in-memory only)
# ─────────────────────────────────────────────────────────────────────────────

def fit_ae(X, epochs, seed, init=None, lr=cfg.AE_LEARNING_RATE):
    """Same recipe as train.py (Adam, cosine LR, grad clip 1.0, batch 256), no checkpoints."""
    torch.manual_seed(seed)
    ae = FlowAutoencoder(feature_dim=X.shape[1]) if init is None else copy.deepcopy(init)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    data = torch.from_numpy(X)
    g = torch.Generator().manual_seed(seed)
    ae.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(data), generator=g).split(cfg.AE_BATCH_SIZE):
            batch = data[idx]
            opt.zero_grad()
            x_hat, _ = ae(batch)
            F.mse_loss(x_hat, batch).backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            opt.step()
        sched.step()
    return ae.eval()


def fedavg(models, weights):
    avg = copy.deepcopy(models[0])
    w = np.asarray(weights, np.float64) / np.sum(weights)
    with torch.no_grad():
        for name, p in avg.state_dict().items():
            p.copy_(sum(float(wi) * m.state_dict()[name] for wi, m in zip(w, models)))
    return avg


def ae_score(ae, X, bs=65536):
    out = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            out.append(ae.anomaly_score(torch.from_numpy(X[s:s + bs])).numpy())
    return np.concatenate(out)


def ae_latent(ae, X, bs=65536):
    out = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            out.append(ae.encode(torch.from_numpy(X[s:s + bs])).numpy())
    return np.concatenate(out)


def fit_head(Z, y, epochs, seed, init=None):
    torch.manual_seed(seed)
    head = AttackHead(Z.shape[1]) if init is None else copy.deepcopy(init)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    Zt, yt = torch.from_numpy(Z), torch.from_numpy(y.astype(np.float32))
    pos = max(float(y.mean()), 1e-6)
    w_pos, w_neg = 0.5 / pos, 0.5 / max(1 - pos, 1e-6)        # class-balanced BCE
    g = torch.Generator().manual_seed(seed)
    head.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(Zt), generator=g).split(512):
            opt.zero_grad()
            p = head(Zt[idx]).squeeze(-1)
            w = torch.where(yt[idx] > 0, w_pos, w_neg)
            F.binary_cross_entropy(p, yt[idx], weight=w).backward()
            opt.step()
    return head.eval()


def head_score(head, Z):
    with torch.no_grad():
        return head(torch.from_numpy(Z)).squeeze(-1).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Experiments
# ─────────────────────────────────────────────────────────────────────────────

def shortcut_report(fx: Features, table) -> dict:
    y = fx.y["test"]
    ttl = table.columns(["MAX_TTL"])[fx.sets["test"], 0]
    a = roc_auc_score(y, ttl)
    single = {}
    for j, name in enumerate(fx.names):
        col = fx.X["test"][:, j]
        if col.min() == col.max():
            continue
        a_j = roc_auc_score(y, col)
        single[name] = round(float(max(a_j, 1 - a_j)), 4)
    top = dict(sorted(single.items(), key=lambda kv: -kv[1])[:8])
    return {"max_ttl_rule_auroc": round(float(max(a, 1 - a)), 4),
            "top_single_features_auroc": top}


def detector_baselines(fx: Features, class_names, args, ttl_tag, ae_variants=True) -> dict:
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import IsolationForest
    res = {}
    ev = lambda s_test, s_cal, mask=None: evaluate(
        s_test, fx.y["test"], fx.cls["test"], s_cal, class_names, mask)

    t0 = time.time()
    rng = np.random.default_rng(cfg.SEED)
    sub = rng.choice(len(fx.X["train_benign"]), min(200_000, len(fx.X["train_benign"])), replace=False)
    iso = IsolationForest(n_estimators=200, random_state=cfg.SEED, n_jobs=-1).fit(fx.X["train_benign"][sub])
    res["isolation_forest"] = ev(-iso.score_samples(fx.X["test"]), -iso.score_samples(fx.X["calib_benign"]))
    log.info(f"[{ttl_tag}] IsolationForest done ({time.time()-t0:.0f}s)")

    t0 = time.time()
    gb = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                        random_state=cfg.SEED, verbose=-1, n_jobs=-1)
    gb.fit(fx.X["train"], fx.y["train"])
    res["lightgbm_supervised"] = ev(gb.predict_proba(fx.X["test"])[:, 1],
                                    gb.predict_proba(fx.X["calib_benign"])[:, 1])
    log.info(f"[{ttl_tag}] LightGBM done ({time.time()-t0:.0f}s)")

    # Autoencoder: central / local per org / FedAvg — one recipe, several seeds.
    if not ae_variants:
        runs = []
        for seed in range(args.seeds):
            ae_c = fit_ae(fx.X["train_benign"], args.ae_epochs, cfg.SEED + seed)
            runs.append(ev(ae_score(ae_c, fx.X["test"]), ae_score(ae_c, fx.X["calib_benign"])))
        res["ae_central"] = summarise(runs)
        return res
    central, local, fed = [], [], []
    orgs = [o for o in range(len(ds.ORG_NAMES))
            if (fx.org["train_benign"] == o).sum() > 1000]
    for seed in range(args.seeds):
        t0 = time.time()
        ae_c = fit_ae(fx.X["train_benign"], args.ae_epochs, cfg.SEED + seed)
        central.append(ev(ae_score(ae_c, fx.X["test"]), ae_score(ae_c, fx.X["calib_benign"])))

        # Local-only: each org trains on its own benign flows, calibrates on its
        # own calibration flows and is scored on its own test flows.
        per_org = {}
        for o in orgs:
            Xo = fx.X["train_benign"][fx.org["train_benign"] == o]
            ae_o = fit_ae(Xo, args.ae_epochs, cfg.SEED + seed)
            cal = fx.X["calib_benign"][fx.org["calib_benign"] == o]
            mask = fx.org["test"] == o
            per_org[ds.ORG_NAMES[o]] = ev(ae_score(ae_o, fx.X["test"]), ae_score(ae_o, cal), mask)
        local.append(per_org)

        # FedAvg over honest orgs: rounds × local epochs = the central epoch budget.
        rounds = max(1, args.ae_epochs // args.local_epochs)
        torch.manual_seed(cfg.SEED + seed)
        glob = FlowAutoencoder(feature_dim=fx.X["train"].shape[1])
        for r in range(rounds):
            models, sizes = [], []
            for o in orgs:
                Xo = fx.X["train_benign"][fx.org["train_benign"] == o]
                models.append(fit_ae(Xo, args.local_epochs, cfg.SEED + seed * 100 + r, init=glob))
                sizes.append(len(Xo))
            glob = fedavg(models, sizes).eval()
        fed.append(ev(ae_score(glob, fx.X["test"]), ae_score(glob, fx.X["calib_benign"])))
        log.info(f"[{ttl_tag}] AE central/local/FedAvg seed {seed} done ({time.time()-t0:.0f}s)")

    res["ae_central"] = summarise(central)
    res["ae_fedavg"] = summarise(fed)
    res["ae_local_per_org"] = {o: summarise([run[o] for run in local]) for o in local[0]}
    return res


def zero_day_loco(fx: Features, class_names) -> dict:
    """Supervised LightGBM trained WITHOUT class c: how much of c does it still catch?"""
    from lightgbm import LGBMClassifier
    out = {}
    for c in np.unique(fx.cls["test"][fx.y["test"] == 1]):
        keep = fx.cls["train"] != c
        gb = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                            random_state=cfg.SEED, verbose=-1, n_jobs=-1)
        gb.fit(fx.X["train"][keep], fx.y["train"][keep])
        s_test = gb.predict_proba(fx.X["test"])[:, 1]
        thr = float(np.percentile(gb.predict_proba(fx.X["calib_benign"])[:, 1], 99))
        m = fx.cls["test"] == c
        benign = fx.y["test"] == 0
        both = m | benign
        out[class_names[c]] = {
            "n_test": int(m.sum()),
            "recall_unseen@calibFPR1%": round(float((s_test[m] > thr).mean()), 3),
            "auroc_unseen_vs_benign": round(float(roc_auc_score(fx.y["test"][both], s_test[both])), 4),
        }
        log.info(f"[LOCO] {class_names[c]}: recall={out[class_names[c]]['recall_unseen@calibFPR1%']}")
    return out


def federation_holdout(fx: Features, class_names, args) -> dict:
    """
    Non-IID attack-class hold-out. Each attacked org hides `hide` classes from
    its own training data (every class stays visible to at least one other
    org). AttackHead on central-AE latents, trained local / FedAvg / central.
    Scored per org on its test flows: recall on classes it HID vs classes it saw.
    """
    rng = np.random.default_rng(cfg.SEED)
    attack_classes = [int(c) for c in np.unique(fx.cls["train"][fx.y["train"] == 1])]
    orgs = [o for o in range(len(ds.ORG_NAMES))
            if ((fx.org["train"] == o) & (fx.y["train"] == 1)).sum() > 0]
    hidden = {}
    for o in orgs:
        hidden[o] = set(int(c) for c in rng.choice(attack_classes, size=args.hide, replace=False))
    for c in attack_classes:                  # every class must stay visible somewhere
        if all(c in hidden[o] for o in orgs):
            hidden[orgs[int(rng.integers(len(orgs)))]].discard(c)

    ae = fit_ae(fx.X["train_benign"], args.ae_epochs, cfg.SEED)
    Z = {k: ae_latent(ae, fx.X[k]) for k in ("train", "calib_benign", "test")}

    def org_train(o):
        m = (fx.org["train"] == o) & ~np.isin(fx.cls["train"], list(hidden[o]))
        return Z["train"][m], fx.y["train"][m]

    results = {"hidden_classes": {ds.ORG_NAMES[o]: sorted(class_names[c] for c in hidden[o])
                                  for o in orgs}}
    runs = {"local": [], "fedavg": [], "central": []}
    for seed in range(args.seeds):
        local = {o: fit_head(*org_train(o), args.head_epochs, cfg.SEED + seed) for o in orgs}
        torch.manual_seed(cfg.SEED + seed)
        glob = AttackHead(Z["train"].shape[1])
        for r in range(args.head_epochs):
            parts = [org_train(o) for o in orgs]
            models = [fit_head(z, y, 1, cfg.SEED + seed * 100 + r, init=glob) for z, y in parts]
            glob = fedavg(models, [len(y) for _, y in parts]).eval()
        pooled = [org_train(o) for o in orgs]
        central = fit_head(np.concatenate([z for z, _ in pooled]),
                           np.concatenate([y for _, y in pooled]), args.head_epochs, cfg.SEED + seed)

        for name, model_of in (("local", lambda o: local[o]), ("fedavg", lambda o: glob),
                               ("central", lambda o: central)):
            per_org = {}
            for o in orgs:
                head = model_of(o)
                s_test = head_score(head, Z["test"])
                cal = Z["calib_benign"][fx.org["calib_benign"] == o]
                thr = float(np.percentile(head_score(head, cal), 99))
                mo = fx.org["test"] == o
                hid = mo & np.isin(fx.cls["test"], list(hidden[o]))
                seen = mo & (fx.y["test"] == 1) & ~np.isin(fx.cls["test"], list(hidden[o]))
                ben = mo & (fx.y["test"] == 0)
                per_org[ds.ORG_NAMES[o]] = {
                    "recall_hidden": float((s_test[hid] > thr).mean()) if hid.any() else np.nan,
                    "recall_seen": float((s_test[seen] > thr).mean()) if seen.any() else np.nan,
                    "test_fpr": float((s_test[ben] > thr).mean()),
                    "n_hidden": int(hid.sum()), "n_seen": int(seen.sum()),
                }
            runs[name].append(per_org)
        log.info(f"[holdout] seed {seed} done")

    for name, rs in runs.items():
        agg = {}
        for o in rs[0]:
            agg[o] = {k: round(float(np.nanmean([r[o][k] for r in rs])), 4)
                      for k in ("recall_hidden", "recall_seen", "test_fpr")}
            agg[o]["n_hidden"], agg[o]["n_seen"] = rs[0][o]["n_hidden"], rs[0][o]["n_seen"]
        agg["mean_over_orgs"] = {k: round(float(np.nanmean([agg[o][k] for o in rs[0]])), 4)
                                 for k in ("recall_hidden", "recall_seen", "test_fpr")}
        results[name] = agg
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _m(v):
    if isinstance(v, dict):
        return f"{v['mean']:.3f} ± {v['std']:.3f}"
    return "—" if v is None else f"{v:.3f}"


def write_markdown(res: dict, path: Path) -> None:
    L = ["# Wave 1.5 baselines (NF-UNSW-NB15-v3)", "",
         f"Split hash `{res['split_hash'][:12]}`, train fraction {res['train_data_fraction']}, "
         f"test flows (de-duplicated): {res['n_test_flows']:,}. Thresholds: calibration benign P99.", ""]
    titles = {"ttl_off": "TTL OFF (primary)", "ttl_on": "TTL ON",
              "shortcut_ablated": "TTL OFF + generator shortcuts removed"}
    for tag, title in titles.items():
        if tag not in res:
            continue
        r = res[tag]
        L += [f"## Detectors — {title}", "", f"Feature dim {r['feature_dim']}; dropped: {r['dropped'] or 'none'}", "",
              "| Detector | AUROC | PR-AUC | Recall @ calib FPR 1% | Test FPR |", "|---|---|---|---|---|"]
        for name in ("isolation_forest", "lightgbm_supervised", "ae_central", "ae_fedavg"):
            if name not in r["detectors"]:
                continue
            d = r["detectors"][name]
            L.append(f"| {name} | {_m(d.get('auroc'))} | {_m(d.get('pr_auc'))} | "
                     f"{_m(d.get('recall@calibFPR1%'))} | {_m(d.get('test_fpr'))} |")
        for org, d in r["detectors"].get("ae_local_per_org", {}).items():
            L.append(f"| ae_local ({org}, own test flows) | {_m(d.get('auroc'))} | {_m(d.get('pr_auc'))} | "
                     f"{_m(d.get('recall@calibFPR1%'))} | {_m(d.get('test_fpr'))} |")
        L += ["", f"Shortcuts: MAX_TTL rule AUROC {r['shortcuts']['max_ttl_rule_auroc']}; "
              f"top single features in this feature space: {r['shortcuts']['top_single_features_auroc']}", ""]
    for key, title in (("zero_day_loco", "TTL off"),
                       ("zero_day_loco_ablated", "TTL off + generator shortcuts removed")):
        if key not in res:
            continue
        L += [f"## Zero-day: supervised LightGBM trained without the class ({title})", "",
              "| Held-out class | Test flows | Recall @ calib FPR 1% | AUROC vs benign |", "|---|---|---|---|"]
        for c, d in res[key].items():
            L.append(f"| {c} | {d['n_test']} | {d['recall_unseen@calibFPR1%']:.3f} | {d['auroc_unseen_vs_benign']:.4f} |")
        L.append("")
    if "federation_holdout" in res:
        h = res["federation_holdout"]
        L += ["## Federation value: AttackHead on AE latents, non-IID class hold-out (TTL off)", "",
              f"Hidden classes per org: {h['hidden_classes']}", "",
              "| Training | Recall on HIDDEN classes | Recall on seen classes | Test FPR |", "|---|---|---|---|"]
        for name in ("local", "fedavg", "central"):
            d = h[name]["mean_over_orgs"]
            L.append(f"| {name} | {d['recall_hidden']:.3f} | {d['recall_seen']:.3f} | {d['test_fpr']:.3f} |")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 AE epochs, 1 seed — wiring check")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ae-epochs", type=int, default=20)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--head-epochs", type=int, default=10)
    ap.add_argument("--hide", type=int, default=3, help="attack classes each org hides")
    ap.add_argument("--skip-ttl-on", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.ae_epochs, args.local_epochs, args.head_epochs = 1, 2, 1, 2

    manifest = ds.check_manifest()
    table = ds.load_flow_table()
    split = ds.compute_split(manifest["train_data_fraction"], table)
    names = table.class_names
    res = {"split_hash": split.meta["hash"], "train_data_fraction": manifest["train_data_fraction"],
           "n_test_flows": int(len(split.flows["test"])), "args": vars(args)}

    settings = [("ttl_off", False, ()), ("ttl_on", True, ()),
                ("shortcut_ablated", False, tuple(GENERATOR_SHORTCUTS))]
    if args.skip_ttl_on:
        settings = [s for s in settings if s[0] != "ttl_on"]
    for tag, use_ttl, drop in settings:
        log.info(f"══ Feature setting: {tag} ══")
        prep = make_preprocessor(use_ttl, table, split, drop)
        fx = Features(prep, table, split)
        res[tag] = {"feature_dim": len(prep.feature_names), "dropped": list(drop),
                    "shortcuts": shortcut_report(fx, table),
                    "detectors": detector_baselines(fx, names, args, tag,
                                                    ae_variants=(tag != "shortcut_ablated"))}
        if tag == "ttl_off":
            res["zero_day_loco"] = zero_day_loco(fx, names)
            res["federation_holdout"] = federation_holdout(fx, names, args)
        if tag == "shortcut_ablated":
            res["zero_day_loco_ablated"] = zero_day_loco(fx, names)
        del fx

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / ("baselines_quick.json" if args.quick else "baselines.json")
    out.write_text(json.dumps(res, indent=2, default=float))
    write_markdown(res, out.with_suffix(".md"))
    log.info(f"Wrote {out} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
