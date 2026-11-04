import sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
"""
calibrate_thresholds.py — AURA Threshold Calibration & Feature Layout Audit
=============================================================================

Run after `python train.py`. Derives the AE alert thresholds from the
reconstruction-error distribution of BENIGN CALIBRATION flows — a set that is
disjoint from (and de-duplicated against) the AE's training flows, so the
percentiles are out-of-sample.

Writes logs/calibration_results.json, which config.py reads at import time:
  recommended_MSE_THRESHOLD_MEDIUM  → P90 of benign calibration MSE
  recommended_MSE_THRESHOLD_HIGH    → P99
  CH2_MSE_SPLIT_THRESHOLD           → P99 (DC-FLTrust two-pass routing;
                                      scripts/ae_routing_cut.py compares cuts)

Also audits that the preprocessor's output columns match config.FEATURE_NAMES
and the manifest written by train.py.

Usage
-----
    python calibrate_thresholds.py              # audit + calibration
    python calibrate_thresholds.py --audit-only # feature layout audit only
"""

import argparse
import json
import logging

import numpy as np
import torch

import config as cfg
from aura import dataset as ds
from aura.models import FlowAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("calibrate")

AE_CHECKPOINT = cfg.MODELS_DIR / "autoencoder_best.pth"


def load_ae() -> FlowAutoencoder:
    """Load the trained autoencoder; exit if train.py has not been run."""
    if not AE_CHECKPOINT.exists():
        log.critical(f"FATAL: {AE_CHECKPOINT} not found. Run `python train.py` first — "
                     "calibrating an untrained model produces meaningless thresholds.")
        sys.exit(1)
    ae = FlowAutoencoder()
    ae.load_state_dict(torch.load(AE_CHECKPOINT, map_location="cpu", weights_only=True))
    return ae.eval()


def collect_calibration_mse(ae: FlowAutoencoder) -> np.ndarray:
    """Per-flow reconstruction MSE over de-duplicated benign calibration flows."""
    table = ds.load_flow_table()
    split = ds.compute_split(json.loads(ds.MANIFEST_FILE.read_text())["train_data_fraction"], table)
    flows = split.benign_flows("calib", table)
    log.info(f"Calibration set: {len(flows):,} benign flows (disjoint from AE training).")
    mse = []
    for s in range(0, len(flows), 50_000):
        mse.append(ae.anomaly_score(ds.flows_tensor(flows[s:s + 50_000])).numpy())
    return np.concatenate(mse).astype(np.float32)


def print_mse_report(mse: np.ndarray, pct: dict) -> None:
    print("\n" + "=" * 70)
    print("  AURA — AUTOENCODER MSE CALIBRATION REPORT (benign calibration flows)")
    print("=" * 70)
    print(f"\n  Samples : {len(mse):,}")
    print(f"  Min / mean / max MSE : {mse.min():.6f} / {mse.mean():.6f} / {mse.max():.6f}")
    print(f"  Std MSE              : {mse.std():.6f}\n  Percentiles:")
    for p, v in pct.items():
        print(f"    P{str(p).ljust(5)} : {v:.6f}")
    print("\n  MEDIUM = P90, HIGH = P99, HIGH_CONSERVATIVE = P99.5, CH2 split = P99")
    print("=" * 70)


def audit_feature_layout() -> bool:
    """Preprocessor output columns must equal config.FEATURE_NAMES and the manifest."""
    print("\n" + "=" * 70)
    print("  AURA — FEATURE LAYOUT AUDIT")
    print("=" * 70)
    prep = ds.load_preprocessor()
    manifest = json.loads(ds.MANIFEST_FILE.read_text())
    checks = {
        "preprocessor == config.FEATURE_NAMES": prep.feature_names == list(cfg.FEATURE_NAMES),
        "manifest     == config.FEATURE_NAMES": manifest["feature_names"] == list(cfg.FEATURE_NAMES),
        f"FEATURE_DIM == {cfg.FEATURE_DIM}":     len(prep.feature_names) == cfg.FEATURE_DIM,
        "FEATURE_INDEX_MAP covers every column": all(
            cfg.FEATURE_INDEX_MAP[n] == i for i, n in enumerate(cfg.FEATURE_NAMES)),
        f"USE_TTL_FEATURES == {cfg.USE_TTL_FEATURES} (manifest)":
            manifest["use_ttl_features"] == cfg.USE_TTL_FEATURES,
    }
    for name, ok in checks.items():
        print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name}")
    print("=" * 70)
    return all(checks.values())


def main():
    parser = argparse.ArgumentParser(description="AURA threshold calibration + feature audit")
    parser.add_argument("--audit-only", action="store_true",
                        help="Only run the feature layout audit.")
    args = parser.parse_args()

    ds.check_manifest()
    if not audit_feature_layout():
        sys.exit("Feature layout audit failed — re-run train.py.")
    if args.audit_only:
        return

    mse = collect_calibration_mse(load_ae())
    pct = {p: float(np.percentile(mse, p)) for p in (50, 75, 90, 95, 99, 99.5, 99.9)}
    print_mse_report(mse, pct)

    results = {
        "n_samples": int(len(mse)),
        "source": "benign calibration flows (canonical split, de-duplicated)",
        "manifest_split_hash": json.loads(ds.MANIFEST_FILE.read_text())["split_hash"],
        "mse_min": float(mse.min()), "mse_mean": float(mse.mean()),
        "mse_max": float(mse.max()), "mse_std": float(mse.std()),
        "p50": pct[50], "p75": pct[75], "p90": pct[90], "p95": pct[95],
        "p99": pct[99], "p99_5": pct[99.5], "p99_9": pct[99.9],
        "recommended_MSE_THRESHOLD_MEDIUM": pct[90],
        "recommended_MSE_THRESHOLD_HIGH":   pct[99],
        "MSE_THRESHOLD_MEDIUM": pct[90],
        "MSE_THRESHOLD_HIGH":   pct[99],
        "MSE_THRESHOLD_HIGH_CONSERVATIVE": pct[99.5],
        "CH2_MSE_SPLIT_THRESHOLD": pct[99],
    }
    out_path = cfg.LOGS_DIR / "calibration_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
