"""
aura/dataset.py — Canonical NF-UNSW-NB15-v3 data layer
=======================================================

Single owner of everything between the CSV and the models. Every training,
calibration, evaluation and federation script goes through this module, so
they all see the same rows, the same feature representation and the same
split.

Pipeline
--------
  1. load_flow_table()   CSV → time-sorted flow table (cached, memory-mapped)
  2. compute_split()     block-stratified test / calibration / root / train pool,
                         TRAIN_DATA_FRACTION subsample of the train pool only,
                         priority de-duplication of flow-level sets
  3. FlowPreprocessor    log1p + MinMax (continuous), flag bits, one-hot codes;
                         IsolationForest baseline sanitiser; fitted on
                         training benign flows only
  4. feature_matrix()    transformed [N, FEATURE_DIM] float32 (memory-mapped)
  5. WindowStore         lazy 60-flow graph windows over the sorted table
  6. manifest            records what a model was trained on; mismatches fail

Design decisions (2026-10 audit, measured on the full dataset)
-------------------------------------------------------------
  * The CSV is not time-ordered (14% backward steps) → sort by flow start.
  * 54% of rows are duplicate feature vectors → a vector may appear in only
    one of test / calibration / root / train, and once within each.
  * Attacks occur in two eras of the timeline → whole time blocks are
    assigned to splits, stratified by era, so every split sees both.
  * Benign organisations are indistinguishable when split by host → orgs are
    assigned by DESTINATION IP (the victim side), giving 4/5 orgs attacks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

import config as cfg

logger = logging.getLogger(__name__)

DATASET_PATH = cfg.CSV_DIR / "NF-UNSW-NB15-v3.csv"
ORG_NAMES = ["hospital", "bank", "university", "isp", "retail"]

# Every column the preprocessor may read, regardless of the TTL switch, so the
# cached table never needs re-parsing when USE_TTL_FEATURES changes.
_TABLE_FEATURE_COLUMNS = (
    [c for c in cfg.CONTINUOUS_FEATURES if c not in cfg.TTL_FEATURES]
    + cfg.TTL_FEATURES + cfg.TCP_FLAG_FEATURES + list(cfg.CATEGORICAL_FEATURES)
)
_SPLIT_FILE = cfg.SPLITS_DIR / "canonical_split.npz"
_PREPROCESSOR_FILE = cfg.MODELS_DIR / "preprocessor.joblib"
MANIFEST_FILE = cfg.MODELS_DIR / "manifest.json"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Flow table
# ─────────────────────────────────────────────────────────────────────────────

def dataset_fingerprint(path: Path = DATASET_PATH) -> str:
    """Cheap, stable identity for the CSV: size + hash of its head and tail."""
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(4 << 20))
        f.seek(max(0, size - (4 << 20)))
        h.update(f.read())
    return h.hexdigest()


@dataclass
class FlowTable:
    raw:          np.ndarray   # [N, len(raw_columns)] float32 raw feature values
    raw_columns:  List[str]
    labels:       np.ndarray   # [N] int8   0 = benign, 1 = attack
    attack_class: np.ndarray   # [N] int8   index into class_names
    class_names:  List[str]
    src_node:     np.ndarray   # [N] int16  index into host_ips
    dst_node:     np.ndarray   # [N] int16
    dst_org:      np.ndarray   # [N] int8   index into ORG_NAMES (by destination IP)
    start_ms:     np.ndarray   # [N] int64
    host_ips:     List[str]
    fingerprint:  str

    @property
    def n_flows(self) -> int:
        return len(self.labels)

    @property
    def n_windows(self) -> int:
        return self.n_flows // cfg.WINDOW_SIZE

    @property
    def n_hosts(self) -> int:
        return len(self.host_ips)

    def columns(self, names: List[str]) -> np.ndarray:
        idx = [self.raw_columns.index(n) for n in names]
        return self.raw[:, idx]


_TABLE: Optional[FlowTable] = None


def _org_of(ip: str) -> int:
    return int(hashlib.md5(str(ip).encode()).hexdigest(), 16) % len(ORG_NAMES)


def load_flow_table(path: Path = DATASET_PATH) -> FlowTable:
    """Return the time-sorted flow table, parsing the CSV only on a cache miss."""
    global _TABLE
    if _TABLE is not None:
        return _TABLE
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    fp = dataset_fingerprint(path)
    cache = cfg.CACHE_DIR / f"flows_v{cfg.PREPROCESSING_VERSION}_{fp[:12]}"
    meta_file = cache / "meta.json"

    if not meta_file.exists():
        t0 = time.time()
        logger.info(f"[dataset] Parsing {path.name} (one-time, cached to {cache}) …")
        dtypes = {c: np.float32 for c in _TABLE_FEATURE_COLUMNS}
        df = pd.read_csv(
            path,
            usecols=_TABLE_FEATURE_COLUMNS + [
                "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
                "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "Label", "Attack",
            ],
            dtype=dtypes,
        )
        # Rows are not time-ordered in the CSV; windows and splits need them to be.
        df.sort_values(["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"],
                       kind="stable", inplace=True, ignore_index=True)

        raw = df[_TABLE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        bad = ~np.isfinite(raw)
        if bad.any():
            logger.info(f"[dataset] Zero-filling {int(bad.sum()):,} non-finite values "
                        f"(column-wise: {dict(zip(_TABLE_FEATURE_COLUMNS, bad.sum(0)))})")
            raw[bad] = 0.0

        host_ips = sorted(set(df.IPV4_SRC_ADDR) | set(df.IPV4_DST_ADDR))
        host_id = {ip: i for i, ip in enumerate(host_ips)}
        attack = df.Attack.fillna("Benign").astype(str).str.strip()
        attack[df.Label.values == 0] = "Benign"
        class_names = ["Benign"] + sorted(set(attack) - {"Benign"})
        class_id = {c: i for i, c in enumerate(class_names)}

        cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / "raw.npy", raw)
        np.save(cache / "labels.npy", df.Label.to_numpy(np.int8))
        np.save(cache / "attack_class.npy", attack.map(class_id).to_numpy(np.int8))
        np.save(cache / "src_node.npy", df.IPV4_SRC_ADDR.map(host_id).to_numpy(np.int16))
        np.save(cache / "dst_node.npy", df.IPV4_DST_ADDR.map(host_id).to_numpy(np.int16))
        np.save(cache / "dst_org.npy",
                np.array([_org_of(ip) for ip in host_ips], np.int8)[
                    df.IPV4_DST_ADDR.map(host_id).to_numpy()])
        np.save(cache / "start_ms.npy", df.FLOW_START_MILLISECONDS.to_numpy(np.int64))
        meta_file.write_text(json.dumps({
            "raw_columns": _TABLE_FEATURE_COLUMNS, "class_names": class_names,
            "host_ips": host_ips, "fingerprint": fp, "n_flows": len(df),
        }, indent=2))
        del df
        logger.info(f"[dataset] Flow table cached in {time.time() - t0:.0f}s.")

    meta = json.loads(meta_file.read_text())
    ld = lambda name: np.load(cache / f"{name}.npy", mmap_mode="r")
    _TABLE = FlowTable(
        raw=ld("raw"), raw_columns=meta["raw_columns"], labels=ld("labels"),
        attack_class=ld("attack_class"), class_names=meta["class_names"],
        src_node=ld("src_node"), dst_node=ld("dst_node"), dst_org=ld("dst_org"),
        start_ms=ld("start_ms"), host_ips=meta["host_ips"], fingerprint=meta["fingerprint"],
    )
    return _TABLE


# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessor
# ─────────────────────────────────────────────────────────────────────────────

class FlowPreprocessor:
    """
    Raw NetFlow columns → model features in [0, 1], ordered as cfg.FEATURE_NAMES.

      continuous : log1p(max(x, 0)), then MinMax over training benign inliers
      TCP flags  : one binary column per bit in cfg.TCP_FLAG_BITS
      categorical: one-hot over cfg.CATEGORICAL_FEATURES vocab + "other"

    fit() also trains the IsolationForest baseline sanitiser on the same rows;
    is_inlier() applies it, so poisoned / mislabelled benign outliers can be
    removed from AE training.
    """

    def __init__(self):
        self.raw_columns = list(cfg.RAW_FEATURE_COLUMNS)
        self.continuous = list(cfg.CONTINUOUS_FEATURES)
        self.feature_names = list(cfg.FEATURE_NAMES)
        self.use_ttl = cfg.USE_TTL_FEATURES
        self.version = cfg.PREPROCESSING_VERSION
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None
        self.sanitiser = None
        self.n_fit_rows = 0

    # Raw arrays are accepted either in self.raw_columns order or as a DataFrame.
    def _select(self, raw, columns: Optional[List[str]] = None) -> np.ndarray:
        if isinstance(raw, pd.DataFrame):
            return raw[self.raw_columns].to_numpy(np.float32)
        columns = columns or self.raw_columns
        if list(columns) == self.raw_columns:
            return np.asarray(raw, dtype=np.float32)
        idx = [list(columns).index(c) for c in self.raw_columns]
        return np.asarray(raw, dtype=np.float32)[:, idx]

    def _log_continuous(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(np.maximum(x[:, : len(self.continuous)], 0.0))

    def fit(self, raw, columns: Optional[List[str]] = None) -> np.ndarray:
        """Fit on benign training rows. Returns the IsolationForest inlier mask."""
        from sklearn.ensemble import IsolationForest
        x = self._select(raw, columns)
        logc = self._log_continuous(x)
        self.sanitiser = IsolationForest(
            n_estimators=100, contamination=cfg.IF_CONTAMINATION,
            random_state=cfg.SEED, n_jobs=-1,
        ).fit(logc)
        inlier = self.sanitiser.predict(logc) == 1
        self.data_min_ = logc[inlier].min(axis=0)
        self.data_max_ = logc[inlier].max(axis=0)
        self.n_fit_rows = int(inlier.sum())
        logger.info(f"[preprocessor] Fitted on {self.n_fit_rows:,} benign inliers "
                    f"({int((~inlier).sum()):,} IsolationForest outliers removed).")
        return inlier

    def is_inlier(self, raw, columns: Optional[List[str]] = None) -> np.ndarray:
        return self.sanitiser.predict(self._log_continuous(self._select(raw, columns))) == 1

    def transform(self, raw, columns: Optional[List[str]] = None) -> np.ndarray:
        if self.data_min_ is None:
            raise RuntimeError("FlowPreprocessor.transform() called before fit().")
        x = self._select(raw, columns)
        n_cont = len(self.continuous)
        rng = np.where(self.data_max_ > self.data_min_, self.data_max_ - self.data_min_, 1.0)
        parts = [np.clip((self._log_continuous(x) - self.data_min_) / rng, 0.0, 1.0)]
        col = n_cont
        for _ in cfg.TCP_FLAG_FEATURES:
            flags = x[:, col].astype(np.int64)
            parts.append(np.stack([(flags & b) > 0 for b in cfg.TCP_FLAG_BITS.values()], 1))
            col += 1
        for vocab in cfg.CATEGORICAL_FEATURES.values():
            v = x[:, col]
            hot = np.stack([v == c for c in vocab], 1)
            parts.append(np.concatenate([hot, ~hot.any(1, keepdims=True)], 1))
            col += 1
        out = np.concatenate([p.astype(np.float32) for p in parts], axis=1)
        assert out.shape[1] == len(self.feature_names)
        return out

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps([self.version, self.feature_names, self.use_ttl]).encode())
        h.update(np.ascontiguousarray(self.data_min_).tobytes())
        h.update(np.ascontiguousarray(self.data_max_).tobytes())
        return h.hexdigest()

    def matches_config(self) -> bool:
        return (self.version == cfg.PREPROCESSING_VERSION
                and self.feature_names == list(cfg.FEATURE_NAMES)
                and self.use_ttl == cfg.USE_TTL_FEATURES)


def feature_matrix(prep: FlowPreprocessor, table: Optional[FlowTable] = None) -> np.ndarray:
    """Transformed features for every flow, cached on disk and memory-mapped."""
    table = table or load_flow_table()
    path = cfg.CACHE_DIR / f"X_{table.fingerprint[:12]}_{prep.fingerprint()[:12]}.npy"
    if not path.exists():
        logger.info(f"[dataset] Transforming {table.n_flows:,} flows → {path.name} …")
        tmp = path.with_suffix(".partial.npy")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                        shape=(table.n_flows, len(prep.feature_names)))
        cols = table.raw_columns
        for s in range(0, table.n_flows, 250_000):
            out[s:s + 250_000] = prep.transform(table.raw[s:s + 250_000], cols)
        out.flush()
        del out
        tmp.replace(path)
    return np.load(path, mmap_mode="r")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Canonical split
# ─────────────────────────────────────────────────────────────────────────────

SPLIT_ORDER = ("test", "calib", "root", "train")   # de-duplication priority


@dataclass
class Split:
    windows: Dict[str, np.ndarray]   # set name → sorted window indices
    flows:   Dict[str, np.ndarray]   # set name → de-duplicated flow indices
    meta:    dict = field(default_factory=dict)

    @property
    def train_pool_windows(self) -> np.ndarray:
        return self.windows["train_pool"]

    def benign_flows(self, name: str, table: Optional[FlowTable] = None) -> np.ndarray:
        table = table or load_flow_table()
        f = self.flows[name]
        return f[table.labels[f] == 0]

    def org_flows(self, name: str, org_idx: int, table: Optional[FlowTable] = None) -> np.ndarray:
        table = table or load_flow_table()
        f = self.flows[name]
        return f[table.dst_org[f] == org_idx]


def _window_flows(windows: np.ndarray) -> np.ndarray:
    w = np.asarray(windows, dtype=np.int64)
    return (w[:, None] * cfg.WINDOW_SIZE + np.arange(cfg.WINDOW_SIZE)).ravel()


def dedup_keys(table: FlowTable) -> np.ndarray:
    """uint64 key per flow over the model's raw feature view + label."""
    cache = cfg.CACHE_DIR / f"dedup_{table.fingerprint[:12]}_ttl{int(cfg.USE_TTL_FEATURES)}.npy"
    if cache.exists():
        return np.load(cache, mmap_mode="r")
    idx = [table.raw_columns.index(c) for c in cfg.RAW_FEATURE_COLUMNS]
    keys = np.empty(table.n_flows, dtype=np.uint64)
    for s in range(0, table.n_flows, 500_000):           # chunked: bounded memory
        view = pd.DataFrame(np.asarray(table.raw[s:s + 500_000])[:, idx])
        view["label"] = np.asarray(table.labels[s:s + 500_000])
        keys[s:s + 500_000] = pd.util.hash_pandas_object(view, index=False).to_numpy(np.uint64)
    np.save(cache, keys)
    return keys


def _split_meta(table: FlowTable, train_fraction: float) -> dict:
    return {
        "dataset": table.fingerprint, "n_windows": table.n_windows,
        "window_size": cfg.WINDOW_SIZE, "block_windows": cfg.SPLIT_BLOCK_WINDOWS,
        "fractions": [cfg.TEST_SPLIT_FRACTION, cfg.CALIB_SPLIT_FRACTION, cfg.ROOT_SPLIT_FRACTION],
        "train_fraction": float(train_fraction), "seed": cfg.SEED,
        "preprocessing_version": cfg.PREPROCESSING_VERSION,
    }


def _assign_windows(table: FlowTable, train_fraction: float) -> Dict[str, np.ndarray]:
    n_win = table.n_windows
    win_attack = np.asarray(table.labels[: n_win * cfg.WINDOW_SIZE]).reshape(n_win, -1).max(1) > 0
    block = np.arange(n_win) // cfg.SPLIT_BLOCK_WINDOWS
    n_blocks = block.max() + 1
    attack_era = np.bincount(block, weights=win_attack, minlength=n_blocks) \
        / np.bincount(block, minlength=n_blocks) >= 0.5

    rng = np.random.default_rng(cfg.SEED)
    shares = {"test": cfg.TEST_SPLIT_FRACTION, "calib": cfg.CALIB_SPLIT_FRACTION,
              "root": cfg.ROOT_SPLIT_FRACTION}
    block_set = np.full(n_blocks, "train_pool", dtype=object)
    for stratum in (True, False):          # attack-era blocks, then benign-era blocks
        ids = rng.permutation(np.flatnonzero(attack_era == stratum))
        start = 0
        for name, share in shares.items():
            k = int(round(share * len(ids)))
            block_set[ids[start:start + k]] = name
            start += k

    windows = {name: np.flatnonzero(block_set[block] == name)
               for name in ("test", "calib", "root", "train_pool")}

    # TRAIN_DATA_FRACTION subsamples the training pool only (stratified by
    # whether a window contains an attack). The other sets never change.
    pool = windows["train_pool"]
    chosen = []
    for has_attack in (True, False):
        cand = pool[win_attack[pool] == has_attack]
        k = int(round(train_fraction * len(cand)))
        chosen.append(rng.choice(cand, size=k, replace=False))
    windows["train"] = np.sort(np.concatenate(chosen))
    return windows


def _dedup_flows(table: FlowTable, windows: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    keys = dedup_keys(table)
    seen = np.empty(0, dtype=np.uint64)
    flows = {}
    for name in SPLIT_ORDER:
        f = _window_flows(windows[name])
        k = keys[f]
        _, first = np.unique(k, return_index=True)        # unique within the set
        f, k = f[first], k[first]
        keep = ~np.isin(k, seen)                            # not in a higher-priority set
        flows[name] = np.sort(f[keep])
        seen = np.concatenate([seen, k[keep]])
    return flows


def compute_split(train_fraction: Optional[float] = None,
                  table: Optional[FlowTable] = None) -> Split:
    """Load the saved canonical split if it matches the config, else recompute it."""
    table = table or load_flow_table()
    train_fraction = cfg.TRAIN_DATA_FRACTION if train_fraction is None else train_fraction
    meta = _split_meta(table, train_fraction)
    key = json.dumps(meta, sort_keys=True)

    saved_meta = None
    if _SPLIT_FILE.exists():
        saved = np.load(_SPLIT_FILE, allow_pickle=False)
        saved_meta = json.loads(str(saved["meta"])) if "meta" in saved else None

    if key in _SPLITS:                       # per-process cache: de-dup costs ~5 s
        split = _SPLITS[key]
        if saved_meta != meta:
            _save_split_windows(split.windows, meta)
        return split

    if saved_meta == meta:
        windows = {k: saved[k] for k in ("test", "calib", "root", "train_pool", "train")}
    else:
        # The file stores int32, and the hash below must match what a later load computes
        windows = {k: v.astype(np.int32) for k, v in _assign_windows(table, train_fraction).items()}
        _save_split_windows(windows, meta)

    split = Split(windows=windows, flows=_dedup_flows(table, windows), meta=meta)
    split.meta["hash"] = hashlib.sha256(
        b"".join(np.ascontiguousarray(windows[k]).tobytes() for k in sorted(windows))
    ).hexdigest()
    verify_split(split, table)
    _SPLITS[key] = split
    return split


_SPLITS: Dict[str, Split] = {}


def _save_split_windows(windows: Dict[str, np.ndarray], meta: dict) -> None:
    cfg.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(_SPLIT_FILE, meta=np.array(json.dumps(meta)),
             **{k: np.asarray(v).astype(np.int32) for k, v in windows.items()})
    logger.info(f"[split] Canonical split written to {_SPLIT_FILE}")


def verify_split(split: Split, table: Optional[FlowTable] = None) -> None:
    """Raise if any two of test / calib / root / train share a window, flow or vector."""
    table = table or load_flow_table()
    keys = dedup_keys(table)
    names = list(SPLIT_ORDER)
    if not np.all(np.isin(split.windows["train"], split.windows["train_pool"])):
        raise RuntimeError("[split] train windows are not a subset of the train pool.")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if np.intersect1d(split.windows[a], split.windows[b]).size:
                raise RuntimeError(f"[split] windows shared between {a} and {b}")
            if np.intersect1d(split.flows[a], split.flows[b]).size:
                raise RuntimeError(f"[split] flows shared between {a} and {b}")
            if np.intersect1d(keys[split.flows[a]], keys[split.flows[b]]).size:
                raise RuntimeError(f"[split] duplicate feature vectors span {a} and {b}")


def split_summary(split: Split, table: Optional[FlowTable] = None) -> dict:
    table = table or load_flow_table()
    out = {}
    for name in SPLIT_ORDER:
        f = split.flows[name]
        lab = np.asarray(table.labels[f])
        out[name] = {
            "windows": int(len(split.windows[name])), "flows_dedup": int(len(f)),
            "attack_flows": int(lab.sum()), "attack_rate": round(float(lab.mean()), 4),
            "attack_classes": int(len(np.unique(np.asarray(table.attack_class[f])[lab == 1]))),
        }
    out["train_pool_windows"] = int(len(split.windows["train_pool"]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Lazy graph windows
# ─────────────────────────────────────────────────────────────────────────────

def build_node_features(edge_feats: np.ndarray, src: np.ndarray, dst: np.ndarray,
                        num_nodes: int) -> np.ndarray:
    """Node feature = mean of incident edge (flow) features."""
    x = np.zeros((num_nodes, edge_feats.shape[1]), dtype=np.float32)
    np.add.at(x, src, edge_feats)
    np.add.at(x, dst, edge_feats)
    deg = np.bincount(src, minlength=num_nodes) + np.bincount(dst, minlength=num_nodes)
    return x / np.maximum(deg, 1).astype(np.float32)[:, None]


class LazyGraph(Mapping):
    """
    A graph window built on access. Behaves like the old graph dict
    (keys: x, edge_index, edge_attr, window_id, ttl_state, active_nodes).
    """

    _KEYS = ("x", "edge_index", "edge_attr", "window_id", "ttl_state", "active_nodes")

    def __init__(self, store: "WindowStore", window_index: int):
        self._store = store
        self.window_index = int(window_index)

    def __getitem__(self, key):
        if key not in self._KEYS:
            raise KeyError(key)
        return self._store._build(self.window_index)[key]

    def __iter__(self):
        return iter(self._KEYS)

    def __len__(self):
        return len(self._KEYS)


class WindowStore(Sequence):
    """Sequence of (LazyGraph, labels) windows over the time-sorted flow table."""

    def __init__(self, prep: FlowPreprocessor, window_indices: Optional[np.ndarray] = None,
                 table: Optional[FlowTable] = None, _X: Optional[np.ndarray] = None):
        self.table = table or load_flow_table()
        self.prep = prep
        self.X = feature_matrix(prep, self.table) if _X is None else _X
        self.window_indices = (np.arange(self.table.n_windows) if window_indices is None
                               else np.asarray(window_indices, dtype=np.int64))
        self._cache_w, self._cache_g = None, None

    def subset(self, window_indices) -> "WindowStore":
        return WindowStore(self.prep, window_indices, self.table, self.X)

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self.subset(self.window_indices[i])
        w = int(self.window_indices[i])
        s = w * cfg.WINDOW_SIZE
        labels = torch.from_numpy(np.asarray(self.table.labels[s:s + cfg.WINDOW_SIZE], np.int64))
        return LazyGraph(self, w), labels

    def _build(self, w: int) -> dict:
        if self._cache_w != w:
            s, e = w * cfg.WINDOW_SIZE, (w + 1) * cfg.WINDOW_SIZE
            feats = np.array(self.X[s:e])
            src = np.asarray(self.table.src_node[s:e], np.int64)
            dst = np.asarray(self.table.dst_node[s:e], np.int64)
            n = self.table.n_hosts
            self._cache_g = {
                "x": torch.from_numpy(build_node_features(feats, src, dst, n)),
                "edge_index": torch.from_numpy(np.stack([src, dst])),
                "edge_attr": torch.from_numpy(feats),
                "window_id": f"{DATASET_PATH.name}:w{w}",
                "ttl_state": {},
                "active_nodes": torch.from_numpy(np.unique(np.concatenate([src, dst]))),
            }
            self._cache_w = w
        return self._cache_g


def window_index_of(item) -> int:
    """Global window index of a (graph, labels) item from any producer."""
    graph = item[0]
    if isinstance(graph, LazyGraph):
        return graph.window_index
    return int(str(graph["window_id"]).rsplit(":w", 1)[1])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Preprocessor persistence + manifest
# ─────────────────────────────────────────────────────────────────────────────

_PREP: Optional[FlowPreprocessor] = None


def fit_preprocessor(split: Split, table: Optional[FlowTable] = None) -> FlowPreprocessor:
    """Fit on de-duplicated benign TRAIN flows and save it (train.py only)."""
    global _PREP
    import joblib
    table = table or load_flow_table()
    benign = split.benign_flows("train", table)
    prep = FlowPreprocessor()
    prep.fit(table.raw[benign], table.raw_columns)
    _PREPROCESSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(prep, _PREPROCESSOR_FILE)
    _PREP = prep
    return prep


def load_preprocessor() -> FlowPreprocessor:
    """The preprocessor train.py fitted. Raises if missing or stale for the config."""
    global _PREP
    if _PREP is not None:
        return _PREP
    import joblib
    if not _PREPROCESSOR_FILE.exists():
        raise FileNotFoundError(
            f"{_PREPROCESSOR_FILE} not found. Run `python train.py` first — every "
            "other script must use the preprocessor the model was trained with.")
    prep = joblib.load(_PREPROCESSOR_FILE)
    if not isinstance(prep, FlowPreprocessor) or not prep.matches_config():
        raise RuntimeError(
            f"{_PREPROCESSOR_FILE} does not match config.py (feature spec / TTL switch / "
            "preprocessing version changed). Re-run `python train.py`.")
    _PREP = prep
    return prep


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cfg.BASE_DIR,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def write_manifest(prep: FlowPreprocessor, split: Split, extra: Optional[dict] = None) -> dict:
    import sklearn
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git_commit(),
        "dataset_fingerprint": load_flow_table().fingerprint,
        "preprocessing_version": cfg.PREPROCESSING_VERSION,
        "use_ttl_features": cfg.USE_TTL_FEATURES,
        "feature_dim": len(prep.feature_names),
        "feature_names": prep.feature_names,
        "preprocessor_fingerprint": prep.fingerprint(),
        "split_hash": split.meta["hash"],
        "train_data_fraction": split.meta["train_fraction"],
        "seed": cfg.SEED,
        "split_summary": split_summary(split),
        "versions": {"torch": torch.__version__, "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    manifest.update(extra or {})
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))
    return manifest


def check_manifest(split: Optional[Split] = None) -> dict:
    """Raise unless the saved model artefacts match the current data + config."""
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"{MANIFEST_FILE} not found. Run `python train.py` first.")
    m = json.loads(MANIFEST_FILE.read_text())
    prep = load_preprocessor()
    problems = []
    if m["preprocessor_fingerprint"] != prep.fingerprint():
        problems.append("preprocessor differs from the one the model was trained with")
    if m["dataset_fingerprint"] != load_flow_table().fingerprint:
        problems.append("dataset file changed")
    if m["use_ttl_features"] != cfg.USE_TTL_FEATURES:
        problems.append("USE_TTL_FEATURES changed")
    split = split or compute_split(m["train_data_fraction"])
    if m["split_hash"] != split.meta["hash"]:
        problems.append("canonical split differs")
    if problems:
        raise RuntimeError("[manifest] Model artefacts are stale: " + "; ".join(problems)
                           + ". Re-run `python train.py` (then calibrate_thresholds.py).")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Convenience accessors
# ─────────────────────────────────────────────────────────────────────────────

def flows_tensor(flow_idx: np.ndarray, prep: Optional[FlowPreprocessor] = None) -> torch.Tensor:
    """Model-ready features for the given flow indices, in the given order."""
    X = feature_matrix(prep or load_preprocessor())
    return torch.from_numpy(np.asarray(X[np.asarray(flow_idx)], dtype=np.float32))
