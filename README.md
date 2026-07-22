# AURA: Autonomous Unsupervised Response Architecture

AURA is a federated, privacy-preserving, and Byzantine-robust AI system for critical infrastructure anomaly detection. 

This repository contains the complete implementation, benchmarks, and attack simulations for the AURA research publication. It provides a dual-layer detection pipeline using a Flow Autoencoder (Statistical Tripwire) and a Spatio-Temporal Graph Neural Network (Contextual Validator), secured by a novel Dual-Channel FLTrust (DC-FLTrust) aggregation mechanism and Differential Privacy (DP-SGD).

## Overview

Modern critical infrastructures require collaborative threat intelligence without exposing sensitive network traffic. Federated Learning (FL) enables this, but exposes the global model to Byzantine data poisoning and gradient leakage attacks. 

AURA addresses this through three core contributions:
1. **Dual-Channel FLTrust (DC-FLTrust)**: Upgrades the traditional FLTrust protocol by utilizing the Autoencoder's unoptimized `H=32` trajectory bottleneck to reliably segregate honest-but-anomalous nodes (under attack) from actively malicious Byzantine nodes attempting to poison the model.
2. **Federated STGNNs**: Replaces traditional deep learning detection with GraphSAGE-inductive graph networks to capture the spatio-temporal dynamics of lateral movement.
3. **Provable Privacy Defenses**: Defends against Deep Leakage from Gradients (DLG) and Membership Inference Attacks (MIA) natively through architectural bottlenecks, reinforced by Opacus DP-SGD ($\epsilon \le 1.0$).

---

## Repository Structure

The repository is modularized for research reproducibility:

*   **`aura/`**: Core system implementation. Contains the `models.py` (Flow Autoencoder, STGNN), federated learning logic (`fl_server.py`, `fl_client.py`), and the DC-FLTrust aggregation strategy (`dc_fltrust_aggregate.py`).
*   **`aura_attacks/`**: Forensic privacy evaluation modules. Contains `gradient_inversion_attack.py` (DLG) and `mia_attack.py` (Shadow-model based MIA).
*   **`scripts/`**: Orchestration and benchmarking scripts (`benchmark_byzantine.py`, `dp_epsilon_sweep.py`, `generate_publication_artifacts.py`, `train_explainer.py`).
*   **`dataset/`**: Expected directory for the target evaluation dataset (`NF-UNSW-NB15-v3.csv`).
*   **`saved_models/`**: Canonical pre-trained models (`aura_bundle.pth`) and dataset-derived heuristics (`attack_class_stats.json`).
*   **`logs/`**: Dynamic configuration thresholds (`calibration_results.json`) and runtime telemetry.
*   **`splits/`**: Deterministic dataset partitions (`canonical_split.npz`) ensuring strict isolation of train/test metrics.
*   **`results/`**: Output directory for generated CSVs, JSON reports, markdown tables, and `figures/`.

---

## Installation

Ensure you are using Python $\ge$ 3.10 (tested on 3.12). 

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
python train.py
python calibrate_thresholds.py --train-quick
python scripts/train_explainer.py
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

Executing the pipeline populates the `results/` folder with complete forensic artifacts:
*   **Byzantine Robustness**: DC-FLTrust successfully detects and isolates $100\%$ of malicious label-flipping clients while preserving the gradients of honest nodes under high-volume DDoS/Exploit attacks.
*   **Privacy-Utility Tradeoff**: `dp_evaluation_table.md` summarizes the Opacus $\epsilon$ expenditure vs $F1$ utility across the sweep. Utility drops insignificantly ($F1 \approx 0.48$) while achieving mathematically rigorous privacy bounds ($\epsilon < 1.0$ at $\sigma=1.0$).
*   **DLG Resistance**: `results/figures/sigma_vs_dlg_mse.png` visually verifies that gradient inversion completely fails to reconstruct client flow vectors, yielding random noise cosine similarities.
*   **MIA Resistance**: Threshold and Shadow model attacks fail to cross the 0.55 AUROC threshold even in the $\sigma=0.0$ baseline, proving the `H=32` manifold bottleneck provides inherent privacy.
*   **MITM (Man-in-the-Middle)**: Simulated interception correctly triggers SHA-256 Merkle root validation failures during global broadcasting.

---

## Limitations and Future Work

*   **Datasets**: Currently evaluated exclusively on `NF-UNSW-NB15-v3`. Adapting to fundamentally different topographies (e.g., CIC-IDS-2017) requires updating the `config.py` statistical feature map.
*   **Hardware**: Due to Pandas-based GraphSAGE window streaming, multi-core CPU utilization is heavy. The total benchmark pipeline requires ~1.5 hours to execute end-to-end on consumer hardware.
*   **Federation Constraints**: Simulates synchronous cross-silo FL. Future extensions may integrate asynchronous updating policies for straggler resilience.

---

## Citation

If you build upon this work, please cite the repository:

```bibtex
@inproceedings{AURA2026,
  title={AURA: Autonomous Unsupervised Response Architecture for Federated Critical Infrastructure},
  author={[Author Placeholder]},
  booktitle={[Conference/Journal Placeholder]},
  year={2026}
}
```
