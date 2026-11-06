"""
aura/data_loader.py — Compatibility layer over aura/dataset.py
===============================================================

Keeps the public names the rest of the codebase imports (CICIDSDataLoader,
load_client_partition, …) while routing everything through the canonical
data layer in aura/dataset.py:

  * one time-sorted flow table (the CSV is parsed once and cached),
  * one canonical, de-duplicated test / calibration / root / train split,
  * one FlowPreprocessor — the one train.py fitted and saved; it is LOADED
    here, never re-fitted, so every script sees the model's feature space.

New code should import aura.dataset directly.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from aura import dataset as ds

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATASET_PATH = ds.DATASET_PATH
CSV_FILES: List[str] = [str(DATASET_PATH)]
_ORG_NAMES = ds.ORG_NAMES


def _build_node_features(edge_features: np.ndarray, src_nodes: np.ndarray,
                         dst_nodes: np.ndarray, num_nodes: int, feature_dim: int) -> np.ndarray:
    return ds.build_node_features(np.asarray(edge_features, np.float32),
                                  np.asarray(src_nodes), np.asarray(dst_nodes), num_nodes)


class CICIDSDataLoader:
    """
    NF-UNSW-NB15-v3 loader (class name kept for backward compatibility).

    load_fraction is accepted for old call sites but no longer selects rows:
    the whole dataset is always used, and only cfg.TRAIN_DATA_FRACTION
    subsamples the training pool (see aura/dataset.py).
    """

    def __init__(self, csv_dir: Path = cfg.CSV_DIR, load_fraction: Optional[float] = None,
                 window_size: int = cfg.WINDOW_SIZE):
        if window_size != cfg.WINDOW_SIZE:
            raise ValueError("window_size is fixed by cfg.WINDOW_SIZE in the canonical data layer.")
        self.csv_dir = csv_dir
        self.load_fraction = load_fraction
        self.window_size = window_size
        self._feature_cols: List[str] = list(cfg.RAW_FEATURE_COLUMNS)
        self.num_nodes = ds.load_flow_table().n_hosts

    def fit_scaler(self, train_indices=None) -> ds.FlowPreprocessor:
        """Return the saved preprocessor (fitted by train.py on training benign flows)."""
        if train_indices is not None:
            raise ValueError("train_indices is no longer supported; the canonical split defines training data.")
        return ds.load_preprocessor()

    def stream_graphs(self, scaler: Optional[ds.FlowPreprocessor] = None,
                      csv_files: Optional[List[str]] = None) -> ds.WindowStore:
        """All graph windows, in time order, built lazily on access."""
        prep = scaler or ds.load_preprocessor()
        if not isinstance(prep, ds.FlowPreprocessor):
            raise TypeError("stream_graphs() needs the FlowPreprocessor from fit_scaler(); "
                            "a raw MinMaxScaler from an older run is not compatible.")
        return ds.WindowStore(prep)

    def _load_csv(self, path_str: str = str(DATASET_PATH)) -> pd.DataFrame:
        """Legacy accessor: the time-sorted flow table as a DataFrame of raw columns."""
        table = ds.load_flow_table()
        df = pd.DataFrame(table.columns(self._feature_cols), columns=self._feature_cols)
        df["Label"] = np.asarray(table.labels)
        df["Attack"] = np.asarray(table.class_names)[np.asarray(table.attack_class)]
        return df


# ─────────────────────────────────────────────────────────────────────────────
# FL client partitions
# ─────────────────────────────────────────────────────────────────────────────

def load_client_partition(
    client_id: str,
    scaler: Optional[ds.FlowPreprocessor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One organisation's benign training flows for local AE training.

    Organisations are assigned by DESTINATION IP (the traffic their sensors
    see), and only flows from the de-duplicated TRAINING set are used — never
    test, calibration or server-root flows. Returns (train, val) at 80/20.
    """
    parts = client_id.split("_")
    if len(parts) < 2 or parts[1].lower() not in _ORG_NAMES:
        raise ValueError(f"Cannot parse org from client_id '{client_id}'. "
                         f"Expected 'org_<name>_<num>' with name in {_ORG_NAMES}.")
    org_idx = _ORG_NAMES.index(parts[1].lower())

    table = ds.load_flow_table()
    split = ds.compute_split()
    flows = split.org_flows("train", org_idx, table)
    flows = flows[np.asarray(table.labels[flows]) == 0]
    if len(flows) < 2:
        raise RuntimeError(f"[{client_id}] has {len(flows)} benign training flows; "
                           "increase cfg.TRAIN_DATA_FRACTION.")

    rng = np.random.default_rng(cfg.SEED + org_idx)
    flows = rng.permutation(flows)
    X = ds.flows_tensor(flows, scaler or ds.load_preprocessor())
    cut = min(max(1, int(len(X) * 0.8)), len(X) - 1)

    logger.info("[%s] org=%s (dst-IP) train=%d val=%d features=%d",
                client_id, _ORG_NAMES[org_idx], cut, len(X) - cut, X.shape[1])
    return X[:cut], X[cut:]


def load_client_subpartition(
    client_idx: int,
    scaler: Optional[ds.FlowPreprocessor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One of 10 benchmark clients: clients 0-4 get the first half of org 0-4's
    training partition, clients 5-9 the second half. Validation is shared
    between the two halves of an org (it is disjoint from both).
    """
    org_key = _ORG_NAMES[client_idx % len(_ORG_NAMES)]
    X_train, X_val = load_client_partition(f"org_{org_key}_1", scaler=scaler)
    half = len(X_train) // 2
    if half < 1:
        raise RuntimeError(f"Partition for {org_key} too small to sub-partition.")
    X_train = X_train[half:] if client_idx >= len(_ORG_NAMES) else X_train[:half]
    return X_train, X_val


if __name__ == "__main__":
    import json
    table = ds.load_flow_table()
    split = ds.compute_split()
    print(json.dumps(ds.split_summary(split, table), indent=2))
    store = CICIDSDataLoader().stream_graphs()
    graph, labels = store[0]
    print(f"window 0: x={tuple(graph['x'].shape)} edges={graph['edge_index'].shape[1]} "
          f"attack_ratio={labels.float().mean():.3f}")
