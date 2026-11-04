#!/usr/bin/env python3
"""
scripts/ch2_conditions.py — under which conditions does the dual-channel idea work?
===================================================================================
E1  filter under attacks that do real damage (attacker count, attack size, head training,
    label share, root coverage)
E2  data hardness: remove the most separating features, retrain the AE, compare AE and head
E3  attack stealth: blend attack flows towards benign traffic (synthetic control)
E4  knowledge sharing: organisations see different attack classes; local vs federated head
E5  production federation: honest-under-attack vs Byzantine clients, routing on/off,
    channel 1 alone vs both channels
E6  federated head on hard data: does the central head's gain survive federation?
E7  production federation again: undefended FedAvg, and each channel gating only its own model
E8  filter threshold sweep: can it recover from real damage at 5% honest exclusion or less?

Everything is scored on a held-out half of the calibration split. The test split is not
touched. Results go to results/ch2_conditions_<exp>.json.

Usage:
    python scripts/ch2_conditions.py                 # all experiments, one after another
    python scripts/ch2_conditions.py --exp e2,e3
    python scripts/ch2_conditions.py --quick         # wiring check, writes nothing
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
from aura.fl_server import dc_fltrust_aggregate
from aura.models import AttackHead, FlowAutoencoder
from baselines import fit_ae, fit_head

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("ch2_conditions")
RESULTS = ROOT / "results"

CLIENT_FLOWS, ROUNDS, QUICK = 8000, 20, False
CALIB_SEED = 1000
SUPERVISED_EPOCHS = 30


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

class Data:
    """Client, root and evaluation flows as raw model inputs, encoded by a given AE."""

    def __init__(self):
        ds.check_manifest()
        self.table, self.split = ds.load_flow_table(), ds.compute_split()
        self.X = ds.feature_matrix(ds.load_preprocessor(), self.table)
        self.class_names = list(self.table.class_names)
        self.lab, self.cls, self.org = map(np.asarray, (self.table.labels, self.table.attack_class, self.table.dst_org))
        self.g0 = fl_round.load_global_model()
        rng = np.random.default_rng(cfg.SEED)

        self.train = self.fetch(self.split.flows["train"])
        self.clients = []
        for o in range(len(ds.ORG_NAMES)):
            pick = rng.permutation(np.flatnonzero(self.train["o"] == o))[:CLIENT_FLOWS]
            self.clients.append({k: v[pick] for k, v in self.train.items()})
        root = self.fetch(self.split.flows["root"])
        benign = rng.permutation(np.flatnonzero(root["y"] == 0))[:cfg.FLTRUST_ROOT_SAMPLES]
        keep = np.r_[benign, np.flatnonzero(root["y"] == 1)]
        self.root = {k: v[keep] for k, v in root.items()}
        calib = self.fetch(self.split.flows["calib"])
        half = rng.random(len(calib["y"])) < 0.5
        self.thr = {k: v[half & (calib["y"] == 0)] for k, v in calib.items()}
        self.ev = {k: v[~half] for k, v in calib.items()}
        benign = self.split.benign_flows("train", self.table)
        self.train_benign_x = np.asarray(self.X[benign], np.float32)

    def fetch(self, idx):
        return dict(x=np.asarray(self.X[idx], np.float32), y=self.lab[idx], c=self.cls[idx], o=self.org[idx])


def encode(ae, x, bs=65536):
    zs, mses = [], []
    with torch.no_grad():
        for s in range(0, len(x), bs):
            b = torch.from_numpy(x[s:s + bs])
            r, z = ae(b)
            zs.append(z.numpy())
            mses.append(((r - b) ** 2).mean(1).numpy())
    return np.concatenate(zs), np.concatenate(mses)


def with_encoding(ae, d, cols=None):
    x = d["x"] if cols is None else d["x"][:, cols]
    z, mse = encode(ae, x)
    return {**d, "z": z, "mse": mse}


def blended(d, alpha, benign_pool, rng):
    """Move attack flows towards random benign flows: alpha=1 is the real attack."""
    x = d["x"].copy()
    m = d["y"] == 1
    if alpha < 1 and m.any():
        x[m] = alpha * x[m] + (1 - alpha) * benign_pool[rng.integers(len(benign_pool), size=int(m.sum()))]
    return {**d, "x": x}


# ─────────────────────────────────────────────────────────────────────────────
# Head recipes
# ─────────────────────────────────────────────────────────────────────────────

def client_labels(c, labels, seed):
    high = c["mse"] >= cfg.CH2_MSE_SPLIT_THRESHOLD
    if labels == "pseudo":
        return np.ones(len(high), bool), high.astype(np.float32)
    share = int(labels.split("_")[1]) / 100
    known = (c["y"] == 1) & (np.random.default_rng(seed).random(len(high)) < share)
    use = known | ~high
    return use, known[use].astype(np.float32)


def train_delta(head, z, y, recipe):
    """recipe 'weak': 20 full-batch steps. 'strong': 3 mini-batch epochs (about 90 steps at 8k flows)."""
    h = copy.deepcopy(head).train()
    opt = torch.optim.Adam(h.parameters(), lr=1e-3)
    zt, yt = torch.from_numpy(z), torch.from_numpy(y.astype(np.float32))
    pos = float(y.mean())
    w = None
    if 0 < pos < 1:
        w = torch.where(yt > 0, torch.tensor(.5 / pos), torch.tensor(.5 / (1 - pos)))
    if recipe == "weak":
        batches = [torch.arange(len(zt))] * 20
    else:
        batches = [b for _ in range(3) for b in torch.randperm(len(zt)).split(256)]
    for b in batches:
        opt.zero_grad()
        F.binary_cross_entropy(h(zt[b]).squeeze(-1), yt[b], weight=None if w is None else w[b]).backward()
        opt.step()
    return torch.cat([(a - b).flatten() for a, b in zip(h.state_dict().values(), head.state_dict().values())])


def add_delta(head, delta):
    h = copy.deepcopy(head)
    i = 0
    with torch.no_grad():
        for p in h.parameters():
            p.add_(delta[i:i + p.numel()].view_as(p))
            i += p.numel()
    return h


def scores(head, z):
    with torch.no_grad():
        return head(torch.from_numpy(z)).squeeze(-1).numpy()


def detector_metrics(score_ev, score_thr, ev, ae_mse_ev, ae_mse_thr):
    """Head metrics at 1% FPR, and AE + head against the AE alone at the same false-positive rate."""
    y = ev["y"] == 1
    pred = score_ev > np.percentile(score_thr, 99)
    ae_pred = ae_mse_ev > np.percentile(ae_mse_thr, 99)
    union = pred | ae_pred
    union_fpr = float(union[~y].mean())
    matched = np.percentile(ae_mse_thr, 100 * (1 - min(union_fpr, 0.5)))
    return {"auroc": float(roc_auc_score(y, score_ev)), "recall": float(pred[y].mean()),
            "union_recall": float(union[y].mean()), "union_fpr": union_fpr,
            "ae_recall_same_fpr": float((ae_mse_ev > matched)[y].mean()),
            "ae_misses_caught": float(pred[y & ~ae_pred].mean()) if (y & ~ae_pred).any() else 0.0}


def mean_of(rows):
    return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0] if isinstance(rows[0][k], (int, float))}


def std_of(rows, key):
    return round(float(np.std([r[key] for r in rows])), 4)


# ─────────────────────────────────────────────────────────────────────────────
# E1: filter under attacks that do real damage
# ─────────────────────────────────────────────────────────────────────────────

def e1_federate(clients, root, ev, thr_z, labels, recipe, scoring, attack, n_attackers, seed, tau, drop_root_target=False):
    torch.manual_seed(cfg.SEED + seed)
    head = AttackHead()
    attacked = sorted((i for i, c in enumerate(clients) if c["y"].sum() > 0), key=lambda i: -clients[i]["y"].sum())
    attackers = attacked[:n_attackers] if attack != "none" else []
    pooled = np.concatenate([clients[i]["c"][clients[i]["y"] == 1] for i in attacked])
    target = int(np.bincount(pooled).argmax())
    if attackers:
        target = int(np.bincount(np.concatenate([clients[i]["c"][clients[i]["y"] == 1] for i in attackers])).argmax())

    r_keep = np.ones(len(root["y"]), bool)
    if drop_root_target:
        r_keep = ~((root["y"] == 1) & (root["c"] == target))
    rz, ry = root["z"][r_keep], root["y"][r_keep].astype(np.float32)
    groups = {int(g): rz[(root["c"][r_keep] == g) & (ry == 1)] for g in np.unique(root["c"][r_keep][ry == 1])}
    groups[-1] = rz[ry == 0]

    honest_s, attacker_s, honest_x, attacker_x = [], [], [], []
    bank_x = []
    for _ in range(ROUNDS):
        root_d = train_delta(head, rz, ry, recipe)
        deltas = []
        for i, c in enumerate(clients):
            use, y = client_labels(c, labels, seed + i)
            if i in attackers:
                y = y.copy()
                is_attack = c["y"][use] == 1
                y[is_attack if attack == "suppress_all" else (is_attack & (c["c"][use] == target))] = 0
            deltas.append(train_delta(head, c["z"][use], y, recipe))

        if scoring == "cosine":
            s = np.array([float(F.cosine_similarity(d.unsqueeze(0), root_d.unsqueeze(0))) for d in deltas])
        elif scoring == "root_loss":
            L = np.zeros((len(deltas), len(groups)))
            for i, d in enumerate(deltas):
                h = add_delta(head, d)
                for j, (g, z) in enumerate(groups.items()):
                    p = torch.from_numpy(scores(h, z)).clamp(1e-6, 1 - 1e-6)
                    L[i, j] = float(F.binary_cross_entropy(p, torch.zeros_like(p) if g == -1 else torch.ones_like(p)))
            s = -(L - np.median(L, axis=0)).max(axis=1)
        else:
            s = np.ones(len(deltas))

        accept = np.ones(len(deltas), bool) if (tau is None or scoring == "none") else s >= tau
        bank_x.append(not accept[ds.ORG_NAMES.index("bank")])
        for i in range(len(deltas)):
            (attacker_s if i in attackers else honest_s).append(float(s[i]))
            (attacker_x if i in attackers else honest_x).append(not accept[i])
        w = np.where(accept, np.maximum(s, 0) if scoring == "cosine" else 1.0, 0.0)
        if w.sum() > 0:
            w = w / w.sum()
            total = torch.zeros_like(root_d)
            for wi, d in zip(w, deltas):
                if wi > 0 and d.norm() > 0:
                    total += float(wi) * d * (root_d.norm() / d.norm())
            head = add_delta(head, total)

    s_ev, y = scores(head, ev["z"]), ev["y"] == 1
    pred = s_ev > np.percentile(scores(head, thr_z), 99)
    out = {"auroc": float(roc_auc_score(y, s_ev)), "recall": float(pred[y].mean()),
           "target_recall": float(pred[y & (ev["c"] == target)].mean()),
           "honest_excluded": float(np.mean(honest_x)), "bank_excluded": float(np.mean(bank_x))}
    if attackers:
        out["attacker_excluded"] = float(np.mean(attacker_x))
    return out, honest_s


def run_e1(data, seeds):
    ae = data.g0.autoencoder.eval()
    clients = [with_encoding(ae, c) for c in data.clients]
    root, ev = with_encoding(ae, data.root), with_encoding(ae, data.ev)
    thr_z = encode(ae, data.thr["x"])[0]
    labels_set = ["pseudo", "labelled_10", "labelled_100"] if not QUICK else ["labelled_100"]
    recipes = ["weak", "strong"] if not QUICK else ["strong"]
    res = {"grid": {}, "root_coverage": {}, "tau": {}}
    t0 = time.time()
    for lb in labels_set:
        for rc in recipes:
            for sc in ["cosine", "root_loss"]:
                _, hon = e1_federate(clients, root, ev, thr_z, lb, rc, sc, "none", 0, CALIB_SEED, None)
                res["tau"][f"{lb}|{rc}|{sc}"] = float(np.percentile(hon, 5))
            clean = {sc: [e1_federate(clients, root, ev, thr_z, lb, rc, sc, "none", 0, s, res["tau"].get(f"{lb}|{rc}|{sc}"))[0]
                          for s in range(seeds)] for sc in ["none", "cosine", "root_loss"]}
            for n in (1, 2):
                for at in ("suppress_main", "suppress_all"):
                    for sc in ["none", "cosine", "root_loss"]:
                        runs = [e1_federate(clients, root, ev, thr_z, lb, rc, sc, at, n, s, res["tau"].get(f"{lb}|{rc}|{sc}"))[0]
                                for s in range(seeds)]
                        m = mean_of(runs)
                        m["recall_std"], m["target_recall_std"] = std_of(runs, "recall"), std_of(runs, "target_recall")
                        m["clean_recall"], m["clean_honest_excluded"] = mean_of(clean[sc])["recall"], mean_of(clean[sc])["honest_excluded"]
                        res["grid"][f"{lb}|{rc}|{n}|{at}|{sc}"] = m
                log.info(f"[E1] {lb}|{rc}|{n} attackers ({time.time() - t0:.0f}s)")
            save("e1", res)
    # Root data missing the attacked class: strongest case that has damage
    lb, rc = labels_set[-1], recipes[-1]
    for sc in ["cosine", "root_loss"]:
        runs = [e1_federate(clients, root, ev, thr_z, lb, rc, sc, "suppress_main", 2, s, res["tau"][f"{lb}|{rc}|{sc}"], drop_root_target=True)[0]
                for s in range(seeds)]
        res["root_coverage"][sc] = mean_of(runs)
    save("e1", res)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# E2: data hardness
# ─────────────────────────────────────────────────────────────────────────────

def run_e2(data, seeds):
    from lightgbm import LGBMClassifier
    names = data.table_feature_names
    y_tr = data.train["y"]
    sample = np.random.default_rng(cfg.SEED).permutation(len(y_tr))[:100_000]
    auc = []
    for j in range(data.train["x"].shape[1]):
        col = data.train["x"][sample, j]
        a = roc_auc_score(y_tr[sample], col) if col.min() < col.max() else 0.5
        auc.append(max(a, 1 - a))
    order = np.argsort(auc)[::-1]         # ranked on training flows, never evaluation flows
    res = {"ranking": [{"feature": names[j], "auroc": round(float(auc[j]), 4)} for j in order[:30]], "by_k": {}}
    t0 = time.time()
    for k in ([0, 2, 5, 10, 20, 30] if not QUICK else [0, 5]):
        cols = np.sort(order[k:])
        rows, ae_rows = [], []
        for s in range(seeds):
            ae = fit_ae(data.train_benign_x[:, cols], 2 if QUICK else 15, cfg.SEED + s)
            tr = with_encoding(ae, data.train, cols)
            ev, th = with_encoding(ae, data.ev, cols), with_encoding(ae, data.thr, cols)
            y = ev["y"] == 1
            ae_rows.append({"ae_auroc": float(roc_auc_score(y, ev["mse"])),
                            "ae_recall_1pct": float((ev["mse"] > np.percentile(th["mse"], 99))[y].mean()),
                            "ae_recall_2pct": float((ev["mse"] > np.percentile(th["mse"], 98))[y].mean())})
            high = tr["mse"] >= np.percentile(th["mse"], 75)
            for name, t in (("labelled", tr["y"].astype(np.float32)), ("pseudo", high.astype(np.float32))):
                h = fit_head(tr["z"], t, 2 if QUICK else SUPERVISED_EPOCHS, cfg.SEED + s)
                m = detector_metrics(scores(h, ev["z"]), scores(h, th["z"]), ev, ev["mse"], th["mse"])
                rows.append({"name": name, **m})
        entry = {"ae": mean_of(ae_rows), "head_labelled": mean_of([r for r in rows if r["name"] == "labelled"]),
                 "head_pseudo": mean_of([r for r in rows if r["name"] == "pseudo"])}

        lgb = LGBMClassifier(n_estimators=100, n_jobs=4, verbose=-1, random_state=cfg.SEED)
        lgb.fit(data.train["x"][:, cols], data.train["y"])
        p, pt = lgb.predict_proba(data.ev["x"][:, cols])[:, 1], lgb.predict_proba(data.thr["x"][:, cols])[:, 1]
        y = data.ev["y"] == 1
        entry["lightgbm"] = {"auroc": float(roc_auc_score(y, p)), "recall_1pct": float((p > np.percentile(pt, 99))[y].mean())}
        loco = {}
        for c in np.unique(data.train["c"][data.train["y"] == 1]):
            keep = data.train["c"] != c
            m = LGBMClassifier(n_estimators=100, n_jobs=4, verbose=-1, random_state=cfg.SEED)
            m.fit(data.train["x"][keep][:, cols], data.train["y"][keep])
            pe = m.predict_proba(data.ev["x"][:, cols])[:, 1]
            th_c = np.percentile(m.predict_proba(data.thr["x"][:, cols])[:, 1], 99)
            loco[data.class_names[c]] = round(float((pe[y & (data.ev["c"] == c)] > th_c).mean()), 3)
        entry["lightgbm_unseen_class"] = {"mean": round(float(np.mean(list(loco.values()))), 3),
                                          "min": round(float(np.min(list(loco.values()))), 3), "per_class": loco}
        entry["removed"] = [names[j] for j in order[:k]]
        res["by_k"][str(k)] = entry
        log.info(f"[E2] k={k}: {json.dumps({a: entry[a] for a in ('ae', 'lightgbm')})} ({time.time() - t0:.0f}s)")
        save("e2", res)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# E3: attack stealth (synthetic control)
# ─────────────────────────────────────────────────────────────────────────────

def run_e3(data, seeds):
    from lightgbm import LGBMClassifier
    ae = data.g0.autoencoder.eval()
    res = {}
    th = with_encoding(ae, data.thr)
    t0 = time.time()
    for alpha in ([1.0, 0.5, 0.25, 0.1, 0.05] if not QUICK else [1.0, 0.1]):
        rng = np.random.default_rng(cfg.SEED)
        tr = with_encoding(ae, blended(data.train, alpha, data.train_benign_x, rng))
        ev = with_encoding(ae, blended(data.ev, alpha, data.train_benign_x, rng))
        y = ev["y"] == 1
        entry = {"ae": {"auroc": float(roc_auc_score(y, ev["mse"])),
                        "recall_1pct": float((ev["mse"] > np.percentile(th["mse"], 99))[y].mean()),
                        "recall_2pct": float((ev["mse"] > np.percentile(th["mse"], 98))[y].mean())}}
        high = tr["mse"] >= cfg.CH2_MSE_SPLIT_THRESHOLD
        for name, t in (("labelled", tr["y"].astype(np.float32)), ("pseudo", high.astype(np.float32))):
            rows = []
            for s in range(seeds):
                h = fit_head(tr["z"], t, 2 if QUICK else SUPERVISED_EPOCHS, cfg.SEED + s)
                rows.append(detector_metrics(scores(h, ev["z"]), scores(h, th["z"]), ev, ev["mse"], th["mse"]))
            entry[f"head_{name}"] = mean_of(rows)
        lgb = LGBMClassifier(n_estimators=100, n_jobs=4, verbose=-1, random_state=cfg.SEED).fit(tr["x"], tr["y"])
        p, pt = lgb.predict_proba(ev["x"])[:, 1], lgb.predict_proba(data.thr["x"])[:, 1]
        entry["lightgbm"] = {"auroc": float(roc_auc_score(y, p)), "recall_1pct": float((p > np.percentile(pt, 99))[y].mean())}
        res[str(alpha)] = entry
        log.info(f"[E3] alpha={alpha}: AE {entry['ae']} ({time.time() - t0:.0f}s)")
        save("e3", res)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# E4: sharing attack knowledge across organisations
# ─────────────────────────────────────────────────────────────────────────────

def run_e4(data, seeds):
    ae = data.g0.autoencoder.eval()
    attacked = [i for i, c in enumerate(data.clients) if c["y"].sum() > 0]
    classes = list(np.unique(np.concatenate([data.clients[i]["c"][data.clients[i]["y"] == 1] for i in attacked])))
    own = {i: {c for n, c in enumerate(classes) if n % len(attacked) == a} for a, i in enumerate(attacked)}
    res = {"seen_classes": {ds.ORG_NAMES[i]: sorted(data.class_names[c] for c in own[i]) for i in attacked}, "by_alpha": {}}
    th = with_encoding(ae, data.thr)
    t0 = time.time()
    for alpha in ([1.0, 0.25, 0.1] if not QUICK else [1.0]):
        rng = np.random.default_rng(cfg.SEED)
        ev = with_encoding(ae, blended(data.ev, alpha, data.train_benign_x, rng))
        clients = []
        for i, c in enumerate(data.clients):
            keep = np.ones(len(c["y"]), bool)
            if i in own:
                keep = (c["y"] == 0) | np.isin(c["c"], list(own[i]))
            else:
                keep = c["y"] == 0
            clients.append(with_encoding(ae, blended({k: v[keep] for k, v in c.items()}, alpha, data.train_benign_x, rng)))
        root = with_encoding(ae, blended(data.root, alpha, data.train_benign_x, rng))
        y = ev["y"] == 1
        entry = {"ae_recall_unseen_by_org": {}}
        for labels in (["pseudo", "labelled_100"] if not QUICK else ["labelled_100"]):
            arms = {"local": [], "fedavg": [], "cosine_filter": []}
            exclusions = []
            for s in range(seeds):
                # Federated heads with and without the cosine filter
                for arm, use_filter in (("fedavg", False), ("cosine_filter", True)):
                    torch.manual_seed(cfg.SEED + s)
                    head = AttackHead()
                    for _ in range(ROUNDS):
                        rd = train_delta(head, root["z"], root["y"].astype(np.float32), "strong")
                        ds_ = [train_delta(head, c["z"][u], t, "strong") for c in clients for u, t in [client_labels(c, labels, s)]]
                        cs = np.array([float(F.cosine_similarity(d.unsqueeze(0), rd.unsqueeze(0))) for d in ds_])
                        keep_ = (cs >= cfg.CH2_ANTI_ALIGN_THRESHOLD) if use_filter else np.ones(len(ds_), bool)
                        if use_filter:
                            exclusions.append(float((~keep_).mean()))
                        w = (np.maximum(cs, 0) * keep_) if use_filter else keep_.astype(float)
                        if w.sum() > 0:
                            w = w / w.sum()
                            head = add_delta(head, sum(float(wi) * d * (rd.norm() / d.norm()) for wi, d in zip(w, ds_) if wi > 0 and d.norm() > 0))
                    arms[arm].append(unseen_recall(head, ev, th, own, y))
                local = {}
                for i in own:
                    u, t = client_labels(clients[i], labels, s)
                    local[i] = fit_head(clients[i]["z"][u], t, 2 if QUICK else SUPERVISED_EPOCHS, cfg.SEED + s)
                arms["local"].append(unseen_recall(local, ev, th, own, y))
            entry[labels] = {a: round(float(np.mean(v)), 4) for a, v in arms.items()}
            entry[labels]["honest_excluded_by_filter"] = round(float(np.mean(exclusions)), 4)
        ae_pred = ev["mse"] > np.percentile(th["mse"], 99)
        ae_pred2 = ev["mse"] > np.percentile(th["mse"], 98)
        unseen = [y & ~np.isin(ev["c"], list(own[i])) & (ev["o"] == i) for i in own]
        entry["ae_recall_unseen_1pct"] = round(float(np.mean([ae_pred[m].mean() for m in unseen if m.any()])), 4)
        entry["ae_recall_unseen_2pct"] = round(float(np.mean([ae_pred2[m].mean() for m in unseen if m.any()])), 4)
        entry.pop("ae_recall_unseen_by_org")
        res["by_alpha"][str(alpha)] = entry
        log.info(f"[E4] alpha={alpha}: {entry} ({time.time() - t0:.0f}s)")
        save("e4", res)
    return res


def unseen_recall(head_or_heads, ev, th, own, y):
    """Mean over organisations of recall on attack classes that organisation never saw."""
    vals = []
    for i in own:
        h = head_or_heads[i] if isinstance(head_or_heads, dict) else head_or_heads
        m = y & ~np.isin(ev["c"], list(own[i])) & (ev["o"] == i)
        if m.any():
            vals.append(float((scores(h, ev["z"][m]) > np.percentile(scores(h, th["z"]), 99)).mean()))
    return float(np.mean(vals))


# ─────────────────────────────────────────────────────────────────────────────
# E5: production federation
# ─────────────────────────────────────────────────────────────────────────────

def e5_federate(data, pool_x, routing, defence, burden, byz, seed, ch1_tau, ch2_tau):
    """Real round functions from aura/fl_round.py and the real aggregator."""
    torch.manual_seed(cfg.SEED + seed)
    rng = np.random.default_rng(cfg.SEED + seed)
    g = copy.deepcopy(data.g0)
    g_ae, g_head = fl_round.snapshot(g.autoencoder), fl_round.snapshot(g.attack_head)
    bank, hospital = ds.ORG_NAMES.index("bank"), ds.ORG_NAMES.index("hospital")
    flows = []
    for i, c in enumerate(data.clients):
        x = c["x"].copy()
        if i == bank and burden > 0:              # honest organisation under a real attack
            n = int(len(x) * burden)
            x[:n] = pool_x[rng.integers(len(pool_x), size=n)]
        flows.append(torch.from_numpy(x))
    root_benign = torch.from_numpy(data.root["x"][data.root["y"] == 0])
    root_attack = torch.from_numpy(data.root["x"][data.root["y"] == 1])

    saved = cfg.CH2_MSE_SPLIT_THRESHOLD, cfg.FLTRUST_CH1_THRESHOLD, cfg.CH2_ANTI_ALIGN_THRESHOLD
    if not routing:
        cfg.CH2_MSE_SPLIT_THRESHOLD = float("inf")
    cfg.FLTRUST_CH1_THRESHOLD = ch1_tau if ch1_tau is not None else -1.0
    cfg.CH2_ANTI_ALIGN_THRESHOLD = ch2_tau if ch2_tau is not None else -2.0
    warmup = 0 if defence == "both" else 10 ** 6
    ch1_all, ch2_all, excl_all, head_excl_all = [], [], [], []
    try:
        for rnd in range(1, ROUNDS + 1):
            ae, head = FlowAutoencoder(), AttackHead()
            root_ae, root_head = fl_round.root_reference(g_ae, g_head, root_benign, root_attack)
            ae_ds, head_ds = [], []
            for i, x in enumerate(flows):
                u = fl_round.client_update(ae, head, g_ae, g_head, x)
                ae_d, head_d = u.ae_delta, u.head_delta
                if i == hospital and byz == "sign_flip":
                    ae_d = {k: -v for k, v in ae_d.items()}
                elif i == hospital and byz == "noise":
                    ae_d = {k: torch.randn_like(v) * v.std() for k, v in ae_d.items()}
                elif i == hospital and byz == "head_flip" and head_d is not None:
                    head_d = {k: -v for k, v in head_d.items()}
                ae_ds.append(ae_d)
                head_ds.append(head_d)
            if defence == "fedavg":
                agg_ae, agg_head = _mean_delta(ae_ds, g_ae), _mean_delta(head_ds, g_head)
                ch1, excl, head_excl = [0.0] * len(flows), [False] * len(flows), [False] * len(flows)
            elif defence == "separate":
                # Channel 1 decides the AE update only; channel 2 decides the head update only
                agg_ae, _, ch1, _, _, excl = dc_fltrust_aggregate(
                    ae_ds, head_ds, root_ae, root_head, [rnd] * len(flows),
                    ch2_warmup_rounds=10 ** 6, current_round=rnd, ch1_threshold=cfg.FLTRUST_CH1_THRESHOLD)
                _, agg_head, _, _, _, head_excl = dc_fltrust_aggregate(
                    ae_ds, head_ds, root_ae, root_head, [rnd] * len(flows),
                    ch2_warmup_rounds=0, current_round=rnd, ch1_threshold=-1.0)
            else:
                agg_ae, agg_head, ch1, _, _, excl = dc_fltrust_aggregate(
                    ae_ds, head_ds, root_ae, root_head, [rnd] * len(flows),
                    ch2_warmup_rounds=warmup, current_round=rnd, ch1_threshold=cfg.FLTRUST_CH1_THRESHOLD,
                    joint_exclusion=True)     # the rule E5 measured
                head_excl = excl
            g_ae, g_head = fl_round.apply_delta(g_ae, agg_ae), fl_round.apply_delta(g_head, agg_head)
            ch1_all.append(ch1)
            ch2_all.append(_ch2_raw(head_ds, root_head))
            excl_all.append(excl)
            head_excl_all.append(head_excl)
    finally:
        cfg.CH2_MSE_SPLIT_THRESHOLD, cfg.FLTRUST_CH1_THRESHOLD, cfg.CH2_ANTI_ALIGN_THRESHOLD = saved

    ae = FlowAutoencoder()
    ae.load_state_dict(g_ae)
    ae.eval()
    _, mse_ev = encode(ae, data.ev["x"])
    _, mse_th = encode(ae, data.thr["x"])
    y = data.ev["y"] == 1
    excl = np.array(excl_all, float)
    head_excl = np.array(head_excl_all, float)
    others = [i for i in range(len(flows)) if i not in (bank, hospital)]
    return {"ae_auroc": float(roc_auc_score(y, mse_ev)),
            "ae_recall_1pct": float((mse_ev > np.percentile(mse_th, 99))[y].mean()),
            "ae_recall_2pct": float((mse_ev > np.percentile(mse_th, 98))[y].mean()),
            "under_attack_excluded": float(excl[:, bank].mean()), "byzantine_excluded": float(excl[:, hospital].mean()),
            "others_excluded": float(excl[:, others].mean()),
            "bank_head_excluded": float(head_excl[:, bank].mean()), "byzantine_head_excluded": float(head_excl[:, hospital].mean()),
            "others_head_excluded": float(head_excl[:, others].mean()),
            "ch1_under_attack": float(np.mean([r[bank] for r in ch1_all])),
            "ch1_others": float(np.mean([[r[i] for i in others] for r in ch1_all])),
            "ch2_under_attack": float(np.nanmean([r[bank] for r in ch2_all])),
            "ch2_others": float(np.nanmean([[r[i] for i in others] for r in ch2_all]))}, ch1_all, ch2_all


def _mean_delta(deltas, like):
    present = [d for d in deltas if d is not None]
    if not present:          # with routing off no client trains a head
        return {k: torch.zeros_like(v) for k, v in like.items()}
    return {k: torch.stack([d[k] for d in present]).mean(0) for k in present[0]}


def _ch2_raw(head_ds, root_head):
    root = torch.cat([v.flatten() for v in root_head.values()])
    out = []
    for d in head_ds:
        if d is None:
            out.append(np.nan)
        else:
            f = torch.cat([v.flatten() for v in d.values()])
            out.append(float(F.cosine_similarity(f.unsqueeze(0), root.unsqueeze(0))))
    return out


def run_e5(data, seeds):
    global ROUNDS
    saved_rounds, ROUNDS = ROUNDS, (2 if QUICK else 10)
    pool_x = data.train["x"][data.train["y"] == 1]
    res = {"tau": {}, "runs": {}}
    arms = [("routing_on", True, "ch1"), ("routing_on", True, "both"), ("routing_off", False, "ch1")]
    t0 = time.time()
    for tag, routing, defence in arms:
        _, ch1, ch2 = e5_federate(data, pool_x, routing, defence, 0.0, "none", CALIB_SEED, None, None)
        ch1_tau = float(np.percentile(np.ravel(ch1), 5))
        vals = np.ravel(ch2)
        ch2_tau = float(np.nanpercentile(vals, 5)) if defence == "both" and np.isfinite(vals).any() else None
        key = f"{tag}|{defence}"
        res["tau"][key] = {"ch1": ch1_tau, "ch2": ch2_tau}
        for burden in ([0.0, 0.5, 0.8] if not QUICK else [0.5]):
            for byz in (["none", "sign_flip", "noise"] if not QUICK else ["sign_flip"]):
                runs = [e5_federate(data, pool_x, routing, defence, burden, byz, s, ch1_tau, ch2_tau)[0] for s in range(seeds)]
                res["runs"][f"{key}|{burden}|{byz}"] = {**mean_of(runs), "ae_recall_1pct_std": std_of(runs, "ae_recall_1pct")}
                log.info(f"[E5] {key} burden={burden} byz={byz}: {res['runs'][f'{key}|{burden}|{byz}']} ({time.time() - t0:.0f}s)")
                save("e5", res)
    ROUNDS = saved_rounds
    return res


# ─────────────────────────────────────────────────────────────────────────────
# E6: federated head on hard data
# ─────────────────────────────────────────────────────────────────────────────

def unseen_union(head_or_heads, ev, th, own, y):
    """Per organisation, on classes it never saw: head alone, AE + head, and the AE at the same FPR."""
    rows = []
    ae_pred = ev["mse"] > np.percentile(th["mse"], 99)
    for i in own:
        h = head_or_heads[i] if isinstance(head_or_heads, dict) else head_or_heads
        m = y & ~np.isin(ev["c"], list(own[i])) & (ev["o"] == i)
        if not m.any():
            continue
        pred = scores(h, ev["z"]) > np.percentile(scores(h, th["z"]), 99)
        union = pred | ae_pred
        fpr = float(union[~y].mean())
        matched = ev["mse"] > np.percentile(th["mse"], 100 * (1 - min(fpr, 0.5)))
        rows.append({"head": float(pred[m].mean()), "union": float(union[m].mean()),
                     "ae_same_fpr": float(matched[m].mean()), "union_fpr": fpr})
    return mean_of(rows)


def run_e6(data, seeds):
    e2 = json.loads((RESULTS / "ch2_conditions_e2.json").read_text(encoding="utf-8"))
    names = data.table_feature_names
    attacked = [i for i, c in enumerate(data.clients) if c["y"].sum() > 0]
    classes = list(np.unique(np.concatenate([data.clients[i]["c"][data.clients[i]["y"] == 1] for i in attacked])))
    own = {i: {c for n, c in enumerate(classes) if n % len(attacked) == a} for a, i in enumerate(attacked)}
    res = {}
    t0 = time.time()
    for k in ([0, 10] if not QUICK else [10]):
        drop = {names.index(n) for n in e2["by_k"][str(k)]["removed"]}
        cols = np.array([j for j in range(len(names)) if j not in drop])
        arms = {}
        for s in range(seeds):
            ae = fit_ae(data.train_benign_x[:, cols], 2 if QUICK else 15, cfg.SEED + s)
            ev, th = with_encoding(ae, data.ev, cols), with_encoding(ae, data.thr, cols)
            root = with_encoding(ae, data.root, cols)
            y = ev["y"] == 1
            clients = []
            for i, c in enumerate(data.clients):
                keep = (c["y"] == 0) | np.isin(c["c"], list(own[i])) if i in own else c["y"] == 0
                clients.append(with_encoding(ae, {key: v[keep] for key, v in c.items()}, cols))
            for labels in (["labelled_100", "labelled_10", "pseudo"] if not QUICK else ["labelled_100"]):
                for arm, use_filter in (("fedavg", False), ("cosine_filter", True)):
                    torch.manual_seed(cfg.SEED + s)
                    head = AttackHead()
                    excluded = []
                    for _ in range(ROUNDS):
                        rd = train_delta(head, root["z"], root["y"].astype(np.float32), "strong")
                        ds_ = [train_delta(head, c["z"][u], t, "strong") for i, c in enumerate(clients) for u, t in [client_labels(c, labels, s + i)]]
                        cs = np.array([float(F.cosine_similarity(d.unsqueeze(0), rd.unsqueeze(0))) for d in ds_])
                        keep_ = (cs >= cfg.CH2_ANTI_ALIGN_THRESHOLD) if use_filter else np.ones(len(ds_), bool)
                        excluded.append(float((~keep_).mean()))
                        w = (np.maximum(cs, 0) * keep_) if use_filter else keep_.astype(float)
                        if w.sum() > 0:
                            w = w / w.sum()
                            head = add_delta(head, sum(float(wi) * d * (rd.norm() / d.norm()) for wi, d in zip(w, ds_) if wi > 0 and d.norm() > 0))
                    arms.setdefault(f"{labels}|{arm}", []).append({**unseen_union(head, ev, th, own, y), "excluded": float(np.mean(excluded))})
                local = {}
                for i in own:
                    u, t = client_labels(clients[i], labels, s + i)
                    local[i] = fit_head(clients[i]["z"][u], t, 2 if QUICK else SUPERVISED_EPOCHS, cfg.SEED + s)
                arms.setdefault(f"{labels}|local", []).append(unseen_union(local, ev, th, own, y))
        res[str(k)] = {a: {**mean_of(v), "union_std": std_of(v, "union")} for a, v in arms.items()}
        log.info(f"[E6] k={k}: {res[str(k)]} ({time.time() - t0:.0f}s)")
        save("e6", res)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# E7: undefended baseline, and each channel gating only its own model
# ─────────────────────────────────────────────────────────────────────────────

def run_e7(data, seeds):
    global ROUNDS
    saved_rounds, ROUNDS = ROUNDS, (2 if QUICK else 10)
    pool_x = data.train["x"][data.train["y"] == 1]
    res = {"tau": {}, "runs": {}}
    arms = [("routing_on", True, "fedavg"), ("routing_off", False, "fedavg"), ("routing_on", True, "separate")]
    t0 = time.time()
    for tag, routing, defence in arms:
        ch1_tau = ch2_tau = None
        if defence == "separate":
            _, ch1, ch2 = e5_federate(data, pool_x, routing, defence, 0.0, "none", CALIB_SEED, None, None)
            ch1_tau, ch2_tau = float(np.percentile(np.ravel(ch1), 5)), float(np.nanpercentile(np.ravel(ch2), 5))
        key = f"{tag}|{defence}"
        res["tau"][key] = {"ch1": ch1_tau, "ch2": ch2_tau}
        for burden in ([0.0, 0.5, 0.8] if not QUICK else [0.5]):
            for byz in (["none", "sign_flip", "head_flip"] if not QUICK else ["head_flip"]):
                runs = [e5_federate(data, pool_x, routing, defence, burden, byz, s, ch1_tau, ch2_tau)[0] for s in range(seeds)]
                res["runs"][f"{key}|{burden}|{byz}"] = {**mean_of(runs), "ae_recall_1pct_std": std_of(runs, "ae_recall_1pct")}
                log.info(f"[E7] {key} burden={burden} byz={byz}: {res['runs'][f'{key}|{burden}|{byz}']} ({time.time() - t0:.0f}s)")
                save("e7", res)
    ROUNDS = saved_rounds
    return res


# ─────────────────────────────────────────────────────────────────────────────
# E8: filter threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_e8(data, seeds):
    ae = data.g0.autoencoder.eval()
    clients = [with_encoding(ae, c) for c in data.clients]
    root, ev = with_encoding(ae, data.root), with_encoding(ae, data.ev)
    thr_z = encode(ae, data.thr["x"])[0]
    res = {"tau": {}, "runs": {}}
    t0 = time.time()
    for lb in (["labelled_10", "labelled_100"] if not QUICK else ["labelled_100"]):
        for rc in (["weak", "strong"] if not QUICK else ["strong"]):
            _, hon = e1_federate(clients, root, ev, thr_z, lb, rc, "cosine", "none", 0, CALIB_SEED, None)
            for pct in ([0, 1, 2, 5] if not QUICK else [1]):
                tau = float(np.percentile(hon, pct))
                res["tau"][f"{lb}|{rc}|{pct}"] = tau
                for at, n in (("none", 0), ("suppress_all", 1), ("suppress_all", 2)):
                    runs = [e1_federate(clients, root, ev, thr_z, lb, rc, "cosine", at, n, s, tau)[0] for s in range(seeds)]
                    res["runs"][f"{lb}|{rc}|{pct}|{at}|{n}"] = {**mean_of(runs), "recall_std": std_of(runs, "recall")}
                log.info(f"[E8] {lb}|{rc}|P{pct} ({time.time() - t0:.0f}s)")
                save("e8", res)
    return res


# ─────────────────────────────────────────────────────────────────────────────

def save(name, res):
    if not QUICK:
        (RESULTS / f"ch2_conditions_{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def main():
    global ROUNDS, QUICK
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="e2,e3,e4,e1,e5")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        QUICK, ROUNDS, args.seeds = True, 2, 1
    torch.set_num_threads(4)
    data = Data()
    data.table_feature_names = list(cfg.FEATURE_NAMES)
    log.info(f"clients {[int(c['y'].sum()) for c in data.clients]} attacks of {CLIENT_FLOWS}; "
             f"root {len(data.root['y'])}; eval {len(data.ev['y'])} (attacks {int(data.ev['y'].sum())})")
    for name in args.exp.split(","):
        t0 = time.time()
        res = {"e1": run_e1, "e2": run_e2, "e3": run_e3, "e4": run_e4, "e5": run_e5,
               "e6": run_e6, "e7": run_e7, "e8": run_e8}[name](data, args.seeds)
        log.info(f"=== {name} finished in {time.time() - t0:.0f}s ===")
        if QUICK:
            log.info(json.dumps(res)[:600])
    log.info("DONE")


if __name__ == "__main__":
    main()
