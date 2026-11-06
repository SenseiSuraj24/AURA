#!/usr/bin/env python3
"""
scripts/ch2_gate.py — does channel 2 earn its place in DC-FLTrust?
==================================================================
Pass criteria, fixed before any run:
  Filter:   beats no-filter (the plain FLTrust stand-in) against a client relabelling
            an attack class as benign, with at most 5% of honest clients excluded and
            no drop in final AE recall (phase B).
  Detector: the head catches attacks the AE misses, without helping as a filter.
  Neither:  report the negative result.

Phase A trains the AttackHead over federation rounds with the AE frozen at G0.
Thresholds come from clean runs on a separate seed (5th percentile of honest scores).
Scored on a held-out half of the calibration split; the test split is not touched.

Usage:
    python scripts/ch2_gate.py            # phase A → results/ch2_gate.json / .md
    python scripts/ch2_gate.py --quick    # wiring check, writes nothing
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
sys.path.insert(0, str(ROOT))

import config as cfg
from aura import dataset as ds, fl_round
from aura.models import AttackHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("ch2_gate")

ROUNDS, HEAD_STEPS, CLIENT_FLOWS, LR = 20, 20, 8000, 1e-3
LABELS = ["one_class", "pseudo", "labelled_1", "labelled_10", "labelled_100"]
SCORING = ["none", "cosine", "root_loss"]
ATTACKS = ["none", "label_flip", "random_labels", "garbage_data",
           "suppress_class", "suppress_blend_50", "suppress_blend_25"]
HONEST_EXCLUSION_TARGET = 5.0     # percentile of clean honest scores used as the threshold
CALIB_SEED_OFFSET = 1000


class Data:
    def __init__(self):
        ds.check_manifest()
        table, split = ds.load_flow_table(), ds.compute_split()
        prep = ds.load_preprocessor()
        X = ds.feature_matrix(prep, table)
        self.class_names = list(table.class_names)
        lab, cls, org = map(np.asarray, (table.labels, table.attack_class, table.dst_org))
        self.ae = fl_round.load_global_model().autoencoder.eval()
        rng = np.random.default_rng(cfg.SEED)

        def load(idx):
            z, mse = self.encode(np.asarray(X[idx], np.float32))
            return dict(z=z, mse=mse, y=lab[idx], c=cls[idx], o=org[idx])

        train = load(split.flows["train"])
        self.clients = []
        for o in range(len(ds.ORG_NAMES)):
            pick = np.flatnonzero(train["o"] == o)
            pick = rng.permutation(pick)[:CLIENT_FLOWS]
            self.clients.append({k: v[pick] for k, v in train.items()})

        root = load(split.flows["root"])
        benign = rng.permutation(np.flatnonzero(root["y"] == 0))[:cfg.FLTRUST_ROOT_SAMPLES]
        keep = np.r_[benign, np.flatnonzero(root["y"] == 1)]
        self.root = {k: v[keep] for k, v in root.items()}

        calib = load(split.flows["calib"])
        half = rng.random(len(calib["y"])) < 0.5
        self.thr_z = calib["z"][half & (calib["y"] == 0)]
        self.thr_mse = calib["mse"][half & (calib["y"] == 0)]
        self.ev = {k: v[~half] for k, v in calib.items()}
        self.garbage_z, _ = self.encode(rng.random((CLIENT_FLOWS, X.shape[1]), dtype=np.float32))
        log.info(f"clients {[len(c['y']) for c in self.clients]} "
                 f"attacks {[int(c['y'].sum()) for c in self.clients]}; root {len(self.root['y'])} "
                 f"(attacks {int(self.root['y'].sum())}); eval {len(self.ev['y'])} "
                 f"(attacks {int(self.ev['y'].sum())})")

    def encode(self, x, bs=65536):
        zs, mses = [], []
        with torch.no_grad():
            for s in range(0, len(x), bs):
                b = torch.from_numpy(x[s:s + bs])
                r, z = self.ae(b)
                zs.append(z.numpy())
                mses.append(((r - b) ** 2).mean(1).numpy())
        return np.concatenate(zs), np.concatenate(mses)


def client_labels(c, labels, seed):
    """(mask of flows used, targets, per-flow weights or None) an honest client trains on."""
    high = c["mse"] >= cfg.CH2_MSE_SPLIT_THRESHOLD
    if labels == "one_class":                 # production: high-MSE flows, all ones, MSE-weighted
        m = c["mse"][high].astype(np.float32)
        w = (m - m.min()) / (m.max() - m.min() + 1e-8) if len(m) else m
        return high, np.ones(high.sum(), np.float32), w
    if labels == "pseudo":
        return np.ones(len(high), bool), high.astype(np.float32), None
    # An analyst has confirmed this share of attack flows; low-error flows count as benign.
    share = int(labels.split("_")[1]) / 100
    known = (c["y"] == 1) & (np.random.default_rng(seed).random(len(high)) < share)
    use = known | ~high
    return use, known[use].astype(np.float32), None


def balance(y):
    pos = float(y.mean())
    if pos in (0.0, 1.0):
        return None
    return np.where(y > 0, .5 / pos, .5 / (1 - pos)).astype(np.float32)


def train_delta(head, z, y, w=None):
    """Full-batch Adam steps from the current global head; returns the flat update."""
    h = copy.deepcopy(head).train()
    opt = torch.optim.Adam(h.parameters(), lr=LR)
    zt, yt = torch.from_numpy(z), torch.from_numpy(y.astype(np.float32))
    wt = torch.from_numpy(w if w is not None else (balance(y) if balance(y) is not None else np.ones(len(y), np.float32)))
    for _ in range(HEAD_STEPS):
        opt.zero_grad()
        F.binary_cross_entropy(h(zt).squeeze(-1), yt, weight=wt).backward()
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


def head_scores(head, z):
    with torch.no_grad():
        return head(torch.from_numpy(z)).squeeze(-1).numpy()


def attack_update(data, head, c, labels, attack, target, honest, gen, label_seed):
    use, y, w = client_labels(c, labels, label_seed)
    z = c["z"][use]
    if attack == "label_flip":
        return train_delta(head, z, 1 - y)
    if attack == "random_labels":
        return train_delta(head, z, (gen.random(len(y)) < .5).astype(np.float32))
    if attack == "garbage_data":
        return train_delta(head, data.garbage_z, np.ones(len(data.garbage_z), np.float32))
    # Relabel the target class as benign: the attacker knows its own traffic.
    cls_use = c["c"][use]
    y2 = y.copy()
    y2[(cls_use == target) & (c["y"][use] == 1)] = 0
    # one_class has no benign targets of its own unless poisoning adds them
    poison = train_delta(head, z, y2)
    blend = {"suppress_class": 1.0, "suppress_blend_50": 0.5, "suppress_blend_25": 0.25}[attack]
    return blend * poison + (1 - blend) * honest


def root_targets(data, labels):
    r = data.root
    if labels == "one_class":
        m = r["y"] == 1
        return r["z"][m], np.ones(m.sum(), np.float32)
    return r["z"], r["y"].astype(np.float32)


def run_federation(data, labels, scoring, attack, seed, tau, ae_eval):
    """Returns the final metrics and the per-round honest / attacker scores."""
    torch.manual_seed(cfg.SEED + seed)
    gen = np.random.default_rng(cfg.SEED + seed)
    head = AttackHead()
    attacked = [i for i, c in enumerate(data.clients) if c["y"].sum() > 0]
    attacker = attacked[seed % len(attacked)] if attack != "none" else None
    target = None
    if attacker is not None:
        c = data.clients[attacker]
        target = int(np.bincount(c["c"][c["y"] == 1]).argmax())

    rz, ry = root_targets(data, labels)
    groups = {int(g): data.root["z"][(data.root["c"] == g) & (data.root["y"] == 1)]
              for g in np.unique(data.root["c"][data.root["y"] == 1])}
    groups[-1] = data.root["z"][data.root["y"] == 0]

    honest_scores, attacker_scores = [], []
    honest_excl, attacker_excl = [], []
    for _ in range(ROUNDS):
        root_d = train_delta(head, rz, ry)
        deltas = []
        for i, c in enumerate(data.clients):
            use, y, w = client_labels(c, labels, seed + i)
            honest = train_delta(head, c["z"][use], y, w) if use.any() else torch.zeros_like(root_d)
            deltas.append(attack_update(data, head, c, labels, attack, target, honest, gen, seed + i)
                          if i == attacker else honest)

        if scoring == "cosine":
            s = np.array([float(F.cosine_similarity(d.unsqueeze(0), root_d.unsqueeze(0))) for d in deltas])
        elif scoring == "root_loss":
            # Worst per-class loss on root data, relative to the median client this round.
            L = np.zeros((len(deltas), len(groups)))
            for i, d in enumerate(deltas):
                h = add_delta(head, d)
                for j, (g, z) in enumerate(groups.items()):
                    if len(z) == 0:
                        continue
                    p = torch.from_numpy(head_scores(h, z)).clamp(1e-6, 1 - 1e-6)
                    t = torch.zeros_like(p) if g == -1 else torch.ones_like(p)
                    L[i, j] = float(F.binary_cross_entropy(p, t))
            s = -(L - np.median(L, axis=0)).max(axis=1)
        else:
            s = np.ones(len(deltas))

        accept = np.ones(len(deltas), bool) if tau is None else s >= tau
        if scoring == "none":
            accept[:] = True
        for i in range(len(deltas)):
            (attacker_scores if i == attacker else honest_scores).append(float(s[i]))
            (attacker_excl if i == attacker else honest_excl).append(not accept[i])

        weights = np.where(accept, np.maximum(s, 0) if scoring == "cosine" else 1.0, 0.0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
            root_norm = root_d.norm()
            total = torch.zeros_like(root_d)
            for wi, d in zip(weights, deltas):
                if wi > 0 and d.norm() > 0:
                    total += float(wi) * d * (root_norm / d.norm())
            head = add_delta(head, total)

    ev = data.ev
    s_ev = head_scores(head, ev["z"])
    thr = np.percentile(head_scores(head, data.thr_z), 99)
    pred = s_ev > thr
    attack_mask = ev["y"] == 1
    out = {
        "auroc": float(roc_auc_score(ev["y"], s_ev)),
        "recall": float(pred[attack_mask].mean()),
        "fpr": float(pred[~attack_mask].mean()),
        "recall_added_to_ae": float(pred[attack_mask & ~ae_eval["pred"]].mean()),
        "union_recall": float((pred | ae_eval["pred"])[attack_mask].mean()),
        "union_fpr": float((pred | ae_eval["pred"])[~attack_mask].mean()),
        "honest_excluded": float(np.mean(honest_excl)),
    }
    if attacker is not None:
        tm = attack_mask & (ev["c"] == target)
        out.update(target_class=data.class_names[target], target_recall=float(pred[tm].mean()),
                   attacker_excluded=float(np.mean(attacker_excl)), attacker=ds.ORG_NAMES[attacker])
    return out, honest_scores


def summarise(runs):
    keys = [k for k in runs[0] if isinstance(runs[0][k], float)]
    return {k: {"mean": round(float(np.mean([r[k] for r in runs])), 4),
                "std": round(float(np.std([r[k] for r in runs])), 4)} for k in keys}


def write_markdown(res, path):
    ae = res["ae_only"]
    L = ["# Channel 2 gate, phase A", "",
         f"Frozen AE at G0; {ROUNDS} rounds, {HEAD_STEPS} head steps per round, {res['seeds']} seeds, one attacker per run.",
         "Thresholds: 5th percentile of honest scores on clean runs from a separate seed. Scored on a held-out half of the calibration split.", "",
         f"AE alone: AUROC {ae['auroc']:.3f}, recall {ae['recall']:.3f} at 1% FPR, recall {ae['recall_2pct']:.3f} at 2% FPR.", "",
         "## Clean federation: head as a detector", "",
         "| Labels | AUROC | Recall at 1% FPR | Share of AE misses caught | AE + head recall | AE + head FPR |",
         "|---|---|---|---|---|---|"]
    for lb in LABELS:
        r = res["runs"][f"{lb}|none|none"]
        L.append(f"| {lb} | {r['auroc']['mean']:.3f} | {r['recall']['mean']:.3f} ± {r['recall']['std']:.3f} | "
                 f"{r['recall_added_to_ae']['mean']:.3f} | {r['union_recall']['mean']:.3f} | {r['union_fpr']['mean']:.3f} |")
    L += ["", "## Against a client relabelling its main class as benign", "",
          "Lower target recall means more damage. Compare each filter to `none` on the same row group.", "",
          "| Labels | Scoring | Attack | Target recall | Attacker excluded | Honest excluded |", "|---|---|---|---|---|---|"]
    for lb in LABELS:
        for sc in SCORING:
            for at in ["suppress_class", "suppress_blend_50", "suppress_blend_25"]:
                r = res["runs"][f"{lb}|{sc}|{at}"]
                L.append(f"| {lb} | {sc} | {at} | {r['target_recall']['mean']:.3f} ± {r['target_recall']['std']:.3f} | "
                         f"{r['attacker_excluded']['mean']:.2f} | {r['honest_excluded']['mean']:.3f} |")
    L += ["", "## Other attacks: share of attacker rounds excluded", "",
          "| Labels | Scoring | Label flip | Random labels | Noise-trained |", "|---|---|---|---|---|"]
    for lb in LABELS:
        for sc in SCORING[1:]:
            cells = [f"{res['runs'][f'{lb}|{sc}|{a}']['attacker_excluded']['mean']:.2f}"
                     for a in ["label_flip", "random_labels", "garbage_data"]]
            L.append(f"| {lb} | {sc} | " + " | ".join(cells) + " |")
    L += ["", "## Honest exclusion on clean federations", "", "| Labels | Scoring | Honest excluded |", "|---|---|---|"]
    for lb in LABELS:
        for sc in SCORING[1:]:
            L.append(f"| {lb} | {sc} | {res['runs'][f'{lb}|{sc}|none']['honest_excluded']['mean']:.3f} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    global ROUNDS, HEAD_STEPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="2 rounds, 1 seed, few combinations; writes nothing")
    args = ap.parse_args()
    labels, attacks = LABELS, ATTACKS
    if args.quick:
        ROUNDS, HEAD_STEPS, args.seeds = 2, 3, 1
        labels, attacks = ["one_class", "labelled_10"], ["none", "suppress_class", "garbage_data"]

    torch.set_num_threads(4)
    data = Data()
    mse = data.ev["mse"]
    thr1, thr2 = np.percentile(data.thr_mse, 99), np.percentile(data.thr_mse, 98)
    y = data.ev["y"] == 1
    ae_eval = {"pred": mse > thr1}
    res = {"seeds": args.seeds, "ae_only": {
        "auroc": float(roc_auc_score(y, mse)), "recall": float((mse > thr1)[y].mean()),
        "recall_2pct": float((mse > thr2)[y].mean()),
        "per_class": {data.class_names[c]: round(float((mse > thr1)[y & (data.ev["c"] == c)].mean()), 3)
                      for c in np.unique(data.ev["c"][y])}},
        "tau": {}, "runs": {}}
    log.info(f"AE only: {res['ae_only']}")

    t0 = time.time()
    for lb in labels:
        for sc in SCORING[1:]:
            _, honest = run_federation(data, lb, sc, "none", CALIB_SEED_OFFSET, None, ae_eval)
            res["tau"][f"{lb}|{sc}"] = float(np.percentile(honest, HONEST_EXCLUSION_TARGET))
        log.info(f"[tau] {lb}: { {k: round(v, 3) for k, v in res['tau'].items() if k.startswith(lb)} } ({time.time() - t0:.0f}s)")

    out = ROOT / "results" / "ch2_gate.json"
    for lb in labels:
        for sc in SCORING:
            for at in attacks:
                tau = res["tau"].get(f"{lb}|{sc}")
                runs = [run_federation(data, lb, sc, at, s, tau, ae_eval)[0] for s in range(args.seeds)]
                res["runs"][f"{lb}|{sc}|{at}"] = {**summarise(runs), "target_classes": sorted({r.get("target_class", "-") for r in runs})}
            log.info(f"{lb}|{sc} done ({time.time() - t0:.0f}s)")
            if not args.quick:
                out.write_text(json.dumps(res, indent=1), encoding="utf-8")

    if args.quick:
        log.info(json.dumps(res["runs"], indent=0)[:1500])
        return
    write_markdown(res, ROOT / "results" / "ch2_gate.md")
    log.info("DONE")


if __name__ == "__main__":
    main()
