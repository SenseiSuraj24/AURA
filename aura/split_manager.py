"""
aura/split_manager.py — Canonical split accessor (backed by aura/dataset.py)
=============================================================================

All scripts that train or evaluate models MUST obtain their windows here.
The split itself lives in aura/dataset.compute_split():

  * whole time-sorted dataset, cut into contiguous blocks of
    cfg.SPLIT_BLOCK_WINDOWS windows,
  * blocks assigned (seeded, stratified by attack era) to four disjoint sets:
    test (20%), calibration (10%), server root (5%), training pool (65%),
  * cfg.TRAIN_DATA_FRACTION subsamples the training pool only,
  * flow-level sets are de-duplicated with priority test > calib > root > train.

Usage
-----
    from aura.split_manager import get_canonical_split
    calibration, train, test, server_attack = get_canonical_split(all_windows)
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import numpy as np

import config as cfg
from aura import dataset as ds

logger = logging.getLogger(__name__)


def _select(all_windows: Sequence, window_indices: np.ndarray):
    if isinstance(all_windows, ds.WindowStore):
        return all_windows.subset(window_indices)
    by_index = {ds.window_index_of(item): item for item in all_windows}
    missing = [int(w) for w in window_indices if int(w) not in by_index]
    if missing:
        raise RuntimeError(f"[split_manager] {len(missing)} canonical windows missing from "
                           "all_windows — pass the full stream from stream_graphs().")
    return [by_index[int(w)] for w in window_indices]


def get_canonical_split(
    all_windows: Sequence,
    test_fraction: float = cfg.TEST_SPLIT_FRACTION,
    calib_fraction: float = cfg.CALIB_SPLIT_FRACTION,
    server_attack_fraction: float = None,
    force_recompute: bool = False,
) -> Tuple[List, List, List, List]:
    """
    Return (calibration_windows, train_windows, test_windows, server_attack_windows).

    The four sets are pairwise disjoint. calibration is NOT a subset of train.
    server_attack_windows are the attack-containing windows of the server-root
    blocks (the FLTrust server's labelled attack reserve).
    """
    if (abs(test_fraction - cfg.TEST_SPLIT_FRACTION) > 1e-9
            or abs(calib_fraction - cfg.CALIB_SPLIT_FRACTION) > 1e-9):
        raise ValueError("Split fractions are fixed in config.py "
                         f"(test={cfg.TEST_SPLIT_FRACTION}, calib={cfg.CALIB_SPLIT_FRACTION}); "
                         f"got test={test_fraction}, calib={calib_fraction}.")
    table = ds.load_flow_table()
    if len(all_windows) != table.n_windows:
        raise RuntimeError(f"[split_manager] expected all {table.n_windows} windows, "
                           f"got {len(all_windows)}.")
    split = ds.compute_split(table=table)

    root = split.windows["root"]
    starts = root * cfg.WINDOW_SIZE
    labels = np.asarray(table.labels)
    root_has_attack = np.array([labels[s:s + cfg.WINDOW_SIZE].any() for s in starts])

    out = (
        _select(all_windows, split.windows["calib"]),
        _select(all_windows, split.windows["train"]),
        _select(all_windows, split.windows["test"]),
        _select(all_windows, root[root_has_attack]),
    )
    logger.info(f"[split_manager] calib={len(out[0])} train={len(out[1])} "
                f"test={len(out[2])} server_attack={len(out[3])} "
                f"(train fraction {split.meta['train_fraction']})")
    return out
