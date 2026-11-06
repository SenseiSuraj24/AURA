# AURA: Adaptive Unified Resilient Architecture

AURA is a research testbed for federated network intrusion detection: several organisations train a shared flow anomaly detector without sharing their traffic, while some members may be malicious.

The repository contains the implementation, the attack simulations and the evaluation pipeline. It is under active re-evaluation (see [Results](#results-and-features)): no performance, robustness or privacy claim is made here until the corrected pipeline has been re-run.

## Components

*   **Flow autoencoder** (76 NetFlow features), trained centrally as a starting model and then federated across five organisations partitioned by destination IP.
*   **Aggregation**: FLTrust-style trust scoring against a server-held root dataset, including a dual-channel variant (DC-FLTrust) that scores the autoencoder and an attack head separately, plus FedAvg, coordinate-wise median and Krum baselines.
*   **Organisation-local graph model** (GraphSAGE), kept and trained locally, never federated.
*   **Attack harness**: sign-flip, scaled, data-poisoning and adaptive members.
*   **Privacy tooling**: Opacus DP-SGD integration and gradient-inversion (DLG) and membership-inference attacks. Privacy accounting is being corrected, so no privacy bound is claimed.
*   **Audit log** of per-round aggregation decisions (Merkle chain).

## Overview

Federated learning lets organisations collaborate on detection without exposing raw traffic, but it exposes the shared model to poisoning by malicious members and the members to leakage through their updates. AURA is built to measure both under controlled conditions on NF-UNSW-NB15-v3 (NF-ToN-IoT-v3 support is experimental).

---

## Repository Structure

The repository is modularized for research reproducibility:

*   **`Reproducibility.md`**: Strict chronological instructions for reproducing all paper artifacts.
*   **`aura/`**: Core system implementation. Contains the `models.py` (Flow Autoencoder, STGNN), federated learning logic (`fl_server.py`, `fl_client.py`), and the DC-FLTrust aggregation strategy (`dc_fltrust_aggregate()` in `fl_server.py`).
*   **`aura_attacks/`**: Forensic privacy evaluation modules. Contains `gradient_inversion_attack.py` (DLG) and `mia_attack.py` (Shadow-model based MIA).
*   **`scripts/`**: Orchestration and benchmarking scripts (`benchmark_byzantine.py`, `dp_epsilon_sweep.py`, `generate_publication_artifacts.py`, `train_explainer.py`).
*   **`dataset/`**: Expected directory for the target evaluation dataset (`NF-UNSW-NB15-v3.csv`).
*   **`saved_models/`**: Canonical pre-trained models (`aura_bundle.pth`) and dataset-derived heuristics (`attack_class_stats.json`).
*   **`logs/`**: Dynamic configuration thresholds (`calibration_results.json`) and runtime telemetry.
*   **`splits/`**: Deterministic dataset partitions (`canonical_split.npz`) ensuring strict isolation of train/test metrics.
*   **`results/`**: Output directory for generated CSVs, JSON reports, markdown tables, and `figures/`.

---

## Installation

Tested with Python 3.12
Compatible with Python 3.10+

```bash
# Clone the repository
git clone https://github.com/naren-kanchi/AURA.git
cd AURA

# Create and activate virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Set execution environment (Windows example)
$env:PYTHONPATH="."
$env:PYTHONIOENCODING="utf-8"
```

*Note: Ensure the dataset is downloaded and placed exactly at `dataset/NF-UNSW-NB15-v3.csv` before running.*

---

## Experimental Pipeline

The repository provides a strict, automated execution pipeline that mirrors the paper's methodology.

### 1. Training (State Preparation)
Establish the canonical pre-trained initialization ($G_0$) and calculate dataset-derived thresholds.
```bash
python train.py                    # fits the feature preprocessor, trains AE + STGNN, writes saved_models/manifest.json
python calibrate_thresholds.py     # thresholds from the held-out calibration split
python scripts/train_explainer.py  # explainer + attack_class_stats.json from training flows
python scripts/smoke_test.py       # wiring, leakage and manifest checks
```
The first run parses the CSV once into `cache/` (time-sorted flow table and feature matrices, ~2 GB). All scripts share one canonical, de-duplicated split (`aura/dataset.py`): test 20%, calibration 10%, server root 5%, training pool 65%. `TRAIN_DATA_FRACTION` in `config.py` subsamples the training pool only.

### 1b. Baselines (go/no-go)
```bash
python scripts/baselines.py        # → results/baselines.json / .md
```

### 2. Federated Benchmark (Robustness)
Evaluate DC-FLTrust utility against Byzantine poisoning.
```bash
python scripts/benchmark_byzantine.py
```

### 3. Privacy Evaluation (DP Sweep)
Execute the Differential Privacy $\sigma$ sweep, running DLG and MIA at each noise tier.
```bash
python scripts/dp_epsilon_sweep.py
```

### 4. Publication Artifacts
Regenerate all publication-ready plots and markdown tables from the JSON telemetry.
```bash
python scripts/generate_publication_artifacts.py
```

---

## Results and Features

> **Status: all results below are pending re-run.** An architecture audit (October 2026) found
> evaluation defects that invalidate the artifacts currently in `results/`. The previous
> artifacts and model checkpoints are preserved locally in `saved_models/pre_fix_backup/`.
> Numbers will be restored here only after the corrected pipeline has been re-run.

Executing the pipeline populates the `results/` folder with JSON reports, CSV/markdown tables and figures. Claims under re-evaluation:

*   **Byzantine Robustness** *(pending)*: DC-FLTrust's ability to separate Byzantine from honest clients. The previous benchmark ran a single configuration (5 clients, 1 Byzantine, 1 seed) without FedAvg/Krum/FLTrust baselines and did not persist its metrics; channel-2 classifier training and the deployed federation path also required fixes.
*   **Privacy-Utility Tradeoff** *(pending)*: DP-SGD accounting is being corrected (sampling, composition across rounds, coverage of the AttackHead update). Previously reported ε values are not valid privacy bounds.
*   **DLG Resistance** *(pending)*: requires a passing positive control (successful reconstruction without DP) before resistance can be claimed.
*   **MIA Resistance** *(pending)*: the previous sweep attacked one fixed checkpoint for every σ, with member/non-member sets that did not match the victim's training data.
*   **MITM (Man-in-the-Middle)** *(pending)*: client-side hash verification is being reworked so that rejection is driven by the hash comparison itself.

---

## Limitations and Future Work

*   **Datasets**: Currently evaluated exclusively on `NF-UNSW-NB15-v3`. Adapting to fundamentally different topographies (e.g., CIC-IDS-2017) requires updating the `config.py` statistical feature map.
*   **Hardware**: Due to Pandas-based GraphSAGE window streaming, multi-core CPU utilization is heavy. The total benchmark pipeline requires ~1.5 hours to execute end-to-end on consumer hardware.
*   **Federation Constraints**: Simulates synchronous cross-silo FL. Future extensions may integrate asynchronous updating policies for straggler resilience.

---

## Citation

If you build upon this work, please cite the repository:

```bibtex
@misc{AURA2026,
  title={AURA: Adaptive Unified Resilient Architecture},
  author={...},
  year={2026},
  note={GitHub repository}
}
```
