# Reproducibility Guide for AURA-1

This document provides exact, step-by-step instructions for reproducing all experiments, benchmarks, and figures published in the AURA-1 paper. The codebase is designed for deterministic execution and automated artifact generation.

---

## 1. Repository Audit

### Datasets
*   **Target Dataset**: `NF-UNSW-NB15-v3.csv` (NetFlow statistical traffic data).
*   **Expected Location**: `dataset/NF-UNSW-NB15-v3.csv`

### Pretrained Checkpoints & State Files
*   **Global Model Prior**: `saved_models/aura_bundle.pth` (Canonical starting state for all experiments).
*   **Attack Profiles**: `saved_models/attack_class_stats.json` (Dataset-derived corruption heuristics).
*   **Threshold Calibration**: `logs/calibration_results.json` (Dynamic benign-traffic thresholds).
*   **Canonical Data Split**: `splits/canonical_split.npz` (Strictly isolates train/test sets across all sweeps).

### Environment Requirements
*   **Python Version**: `>= 3.10` (Tested on `3.12.x`)
*   **Dependencies**: Defined in `requirements.txt` (Opacus, Flower, PyTorch, scikit-learn, etc.)
*   **CUDA / GPU**: Optional but recommended. The codebase automatically maps to `cpu` if `cuda` is unavailable, but GPU dramatically reduces Autoencoder pre-training time.

---

## 2. Environment Setup

Run the following commands exactly as written to instantiate the research environment:

```bash
# 1. Clone the repository
git clone https://github.com/naren-kanchi/AURA.git
cd AURA

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Set environment paths
# On Windows:
$env:PYTHONPATH="."
$env:PYTHONIOENCODING="utf-8"
# On Linux/macOS:
export PYTHONPATH="."
```

### File Placement
Ensure your dataset is placed at `dataset/NF-UNSW-NB15-v3.csv`. The `saved_models/`, `logs/`, and `splits/` directories will be automatically generated and populated by the execution pipeline.

---

## 3. The Canonical Execution Order

To reproduce the paper from scratch, execute the pipeline in this strict chronological order:

1.  **Baseline Training**
    ```bash
    python train.py
    ```
    *Builds the global model prior (`aura_bundle.pth`) and establishes the canonical graph topology.*

2.  **Threshold Calibration**
    ```bash
    python calibrate_thresholds.py --train-quick
    ```
    *Analyzes the benign traffic manifold and saves dynamic UCL thresholds to `logs/calibration_results.json`.*

3.  **Data-Driven Attack Profiles**
    ```bash
    python scripts/train_explainer.py
    ```
    *Calculates p05/p95 feature permutations and outputs `saved_models/attack_class_stats.json` for realistic benchmarking.*

4.  **Utility & Robustness Evaluation (DC-FLTrust)**
    ```bash
    python scripts/benchmark_byzantine.py
    ```
    *Executes the federated Byzantine robustness benchmarks.*

5.  **Privacy-Utility Evaluation & Attacks (DP, DLG, MIA)**
    ```bash
    python scripts/dp_epsilon_sweep.py
    ```
    *Sweeps DP noise multipliers ($\sigma$), tracks privacy budgets ($\epsilon$), and triggers Membership Inference and Gradient Inversion attacks.*

6.  **Artifact Generation**
    ```bash
    python scripts/generate_publication_artifacts.py
    ```
    *Parses all sweep logs to generate CSV tables and matplotlib figures.*

---

## 4. Expected Outputs

All experiment scripts are designed to cleanly export artifacts to designated directories without manual intervention:

*   **`results/`**: Contains raw JSON logs and aggregated metrics.
    *   `dp_epsilon_sweep.json`: Raw hierarchical output of the DP sweep.
    *   `dp_epsilon_sweep.csv`: Flat tabular data summarizing metrics.
    *   `dp_evaluation_table.md`: The publication Markdown table for immediate inclusion in the paper.
*   **`results/figures/`**: Contains all generated `matplotlib` publication plots.
    *   `sigma_vs_f1.png`
    *   `sigma_vs_epsilon.png`
    *   `sigma_vs_roc.png`
*   **`saved_models/`**: Houses all deterministic state files (`aura_bundle.pth`, `attack_class_stats.json`).
*   **`logs/`**: Holds internal run telemetry and calibration constraints (`calibration_results.json`).

---

## 5. Reproducing the Paper

To reproduce the exact figures and tables shown in the publication, execute the final stages of the pipeline:

### Byzantine Benchmark (DC-FLTrust)
```bash
python scripts/benchmark_byzantine.py
```
This isolates the model against simulated label-flipping and noisy clients, testing the Channel 1 / Channel 2 dual-pass detection architecture.

### DP Sweep & Privacy Attacks (DLG / MIA / MITM)
```bash
python scripts/dp_epsilon_sweep.py
```
This single script triggers:
1.  **Federated DP-SGD**: Training with varying `DP_NOISE_MULTIPLIER` bounds (from $\sigma=0.0$ to $\sigma=2.0$).
2.  **DLG Attack**: Reconstructs gradients via `gradient_inversion_attack.py` and reports Cosine Similarity / MSE.
3.  **MIA Attack**: Trains shadow models via `mia_attack.py` and reports ROC-AUC privacy leakage.
4.  **MITM Validation**: Simulates Man-in-the-Middle weight tampering to test hash validation.

### Final Table Generation
```bash
python scripts/generate_publication_artifacts.py
```
The output file `results/dp_evaluation_table.md` will perfectly mirror the DP Privacy-Utility trade-off table presented in the manuscript.

---

## 6. Hardware Estimates

Based on the default configuration parameters (`H=32`, `DP_NOISE_SWEEP=[0.0, 0.5, 1.0, 1.5, 2.0]`):

*   **CPU**: Multi-core processor (e.g., Intel i7 / Ryzen 7). The complete pipeline is heavily CPU-bound during Pandas graph-window streaming and Scikit-learn shadow training.
*   **RAM**: $\ge 16$ GB required (due to holding large graph snapshots in memory).
*   **GPU**: Optional. If running `train.py` from scratch, a CUDA-compatible GPU (e.g., RTX 3060+) reduces Autoencoder training time.
*   **Runtime**:
    *   `train.py`: ~10–15 mins.
    *   `benchmark_byzantine.py`: ~15 mins.
    *   `dp_epsilon_sweep.py`: ~40 mins (approx. 8 minutes per $\sigma$).
    *   **Total End-to-End**: ~1.5 hours.

---

## 7. Determinism

To ensure scientific validity, AURA-1 implements multiple layers of state consistency:

*   **Canonical Splits**: `splits/canonical_split.npz` guarantees that the test-set attack ratio and window distribution remain absolutely identical across every run and baseline comparison.
*   **Checkpoints**: The sweep strictly loads `aura_bundle.pth` before each $\sigma$ evaluation to ensure $G_0$ initialization is identical.
*   **Threshold Calibration**: `config.py` enforces data-driven thresholds dynamically, ensuring no "magic numbers" alter detection rates.
*   **DP Noise**: Opacus DP-SGD leverages pseudo-random generators; minor variance ($\pm 0.02$ F1) may occur across hardware architectures due to floating-point differences, but the decay trajectory is strictly deterministic.

---

## 8. Troubleshooting

| Error | Root Cause | Solution |
| :--- | :--- | :--- |
| **`FileNotFoundError: NF-UNSW-NB15-v3.csv`** | Dataset missing from project root. | Place the CSV inside the `dataset/` directory. |
| **`AttributeError: 'tuple' object has no attribute 'size'`** | Interface drift in Autoencoder MIA script. | Pull the latest commit; `mia_attack.py` was repaired to unpack `(x_hat, z)` correctly. |
| **`Sentinal Threshold WARNING`** | `calibrate_thresholds.py` was not run. | Run `python calibrate_thresholds.py --train-quick` before benchmarking. |
| **`KeyError: 'ch2_mse'`** | Using `benchmark_byzantine.py` before `train_explainer.py`. | Run `python scripts/train_explainer.py` to generate `attack_class_stats.json`. |
| **`Opacus: ModuleNotFoundError`** | Privacy engine not installed. | Run `pip install opacus`. Ensure you are activating the correct `.venv`. |
| **`RuntimeError: DataLoader worker ...`** | Windows multiprocess locking. | Ensure `$env:PYTHONIOENCODING="utf-8"` and `$env:PYTHONPATH="."` are set in PowerShell. |
