"""
config.py — AURA Global Configuration
======================================
Single source of truth for ALL hyperparameters, paths, and system constants.
Centralising config prevents magic numbers from scattering across modules.

Research-grade design goals
---------------------------
  * Zero hardcoded values in any other module — every tunable lives here.
  * AE thresholds are resolved dynamically from calibration_results.json
    (written by calibrate_thresholds.py) so they always reflect the actual
    benign MSE distribution — NOT a magic number.
  * Attack corruption profiles are resolved from attack_class_stats.json
    (written by scripts/train_explainer.py) so they reflect real dataset
    percentiles — NOT hand-typed ranges.
  * All paths constructed relative to BASE_DIR so the project is portable.
"""

import json
import logging
import os
from pathlib import Path

_cfg_log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.resolve()
CSV_DIR    = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "saved_models"
LOGS_DIR   = BASE_DIR / "logs"
CONTRACTS_DIR = BASE_DIR / "contracts"

# Ensure output dirs exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

SPLITS_DIR = BASE_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET  (NF-UNSW-NB15-v3)
# ─────────────────────────────────────────────────────────────────────────────

# NF-UNSW-NB15-v3 uses real source/destination IPs for genuine topology.
NUM_SYNTHETIC_NODES = 20

# Column name for the target label
LABEL_COL = "Label"

# The label value that represents benign (normal) traffic in NF-UNSW-NB15-v3
# Label column is binary: 0 = Benign, 1 = Attack
BENIGN_LABEL = 0

SEED = 42   # Single seed for split, subsampling, IsolationForest and model init

# ── Canonical split ──────────────────────────────────────────────────────────
# The WHOLE dataset is always loaded and time-sorted. The timeline is cut into
# contiguous blocks of SPLIT_BLOCK_WINDOWS windows, and whole blocks are assigned
# (seeded, stratified by attack era) to four pairwise-disjoint sets. The test,
# calibration and server-root sets never change with TRAIN_DATA_FRACTION.
TEST_SPLIT_FRACTION   = 0.20   # held-out evaluation
CALIB_SPLIT_FRACTION  = 0.10   # threshold calibration (never used for training)
ROOT_SPLIT_FRACTION   = 0.05   # FLTrust server root: benign root data + attack reserve
SPLIT_BLOCK_WINDOWS   = 200    # ~12k flows per block; keeps EMA-relevant runs contiguous

# Portion of the TRAINING POOL used for training (seeded, stratified window
# sample). 1.0 = the full pool. Test/calibration/root are unaffected.
TRAIN_DATA_FRACTION = 1.0      # 0.3 for development, 1.0 for final paper runs
DATA_LOAD_FRACTION  = TRAIN_DATA_FRACTION   # deprecated alias for older scripts

# Parsed, time-sorted flow table and transformed feature matrices (memory-mapped)
CACHE_DIR = BASE_DIR / "cache"

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SPECIFICATION (NF-UNSW-NB15-v3)
# ─────────────────────────────────────────────────────────────────────────────
# Measured on the full dataset (2026-10 audit):
#   * MIN_TTL / MAX_TTL alone separate attacks with AUROC 0.998 (benign TTL=31,
#     attacks 254/62) — a testbed artefact, so they are OFF for primary results.
#   * DNS_QUERY_ID is a random transaction id (64k values) — pure noise.
#   * ICMP_TYPE / ICMP_IPV4_TYPE are non-zero on 74% of non-ICMP flows (ICMP is
#     0.017% of traffic) — an exporter artefact; PROTOCOL already marks ICMP.
#   * 29/47 features scale below 0.01 under plain MinMax (heavy tails) → log1p.
USE_TTL_FEATURES = False

TTL_FEATURES      = ["MIN_TTL", "MAX_TTL"]
DROPPED_FEATURES  = ["DNS_QUERY_ID", "ICMP_TYPE", "ICMP_IPV4_TYPE"]

# TCP flag bitmasks (max value 31 in this dataset) → one binary feature per bit
TCP_FLAG_FEATURES = ["TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"]
TCP_FLAG_BITS     = {"fin": 1, "syn": 2, "rst": 4, "psh": 8, "ack": 16}

# Categorical codes → one-hot over the listed values + an "other" column.
# Vocabularies are the most frequent values in the dataset (coverage noted).
CATEGORICAL_FEATURES = {
    "PROTOCOL":             [6, 17, 89, 1],                         # 99.9%
    "L7_PROTO":             [37.0, 36.0, 5.0, 11.0, 7.0, 0.0, 1.0, 92.0],  # 94.5%
    "DNS_QUERY_TYPE":       [0, 1, 255, 28, 16],                    # 99.99%
    "FTP_COMMAND_RET_CODE": [0, 229, 125, 230, 331],                # 98.0%
}

# Every other statistical column is continuous: log1p, then MinMax fitted on
# training-pool benign flows. Identifiers and labels are never features.
NON_FEATURE_COLUMNS = [
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
    "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "Label", "Attack",
]
CONTINUOUS_FEATURES = [
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "DNS_TTL_ANSWER",
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
] + (TTL_FEATURES if USE_TTL_FEATURES else [])

# Raw CSV columns the preprocessor reads (everything except identifiers/labels)
RAW_FEATURE_COLUMNS = (
    CONTINUOUS_FEATURES + TCP_FLAG_FEATURES + list(CATEGORICAL_FEATURES)
)


def _build_feature_names() -> list:
    names = [c.lower() for c in CONTINUOUS_FEATURES]
    for col in TCP_FLAG_FEATURES:
        names += [f"{col.lower()}_{bit}" for bit in TCP_FLAG_BITS]
    for col, vocab in CATEGORICAL_FEATURES.items():
        names += [f"{col.lower()}_{v:g}" for v in vocab] + [f"{col.lower()}_other"]
    return names


# Model input columns, in order. FEATURE_DIM is derived — never hand-typed.
FEATURE_NAMES = _build_feature_names()

# Bump when the preprocessing changes so stale caches / manifests are rejected.
PREPROCESSING_VERSION = 2

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH / TTL EDGE DECAY
# ─────────────────────────────────────────────────────────────────────────────

# Rolling time-window size in simulated "ticks" (1 tick ≈ 1 second of NetFlow)
WINDOW_SIZE = 60          # number of flow rows per graph snapshot

# Time-To-Live: an edge is pruned after this many windows without traffic
EDGE_TTL_WINDOWS = 3

# ─────────────────────────────────────────────────────────────────────────────
# AUTOENCODER (Layer 1 — Statistical Tripwire)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_DIM     = len(FEATURE_NAMES)   # 76 without TTL, 78 with (see FEATURE SPECIFICATION)
ENCODER_DIMS    = [32, 24]   # Progressive compression (smaller network for 47 features)
LATENT_DIM      = 16    # Bottleneck: the latent fingerprint space
DECODER_DIMS    = [24, 32]   # Mirror of encoder (symmetric reconstruction)

AE_LEARNING_RATE = 1e-3
AE_EPOCHS        = 50        # Full training run on NF-UNSW-NB15-v3 subset
AE_BATCH_SIZE    = 256

# Contrastive negative-sampling margin (pushes attack embeddings away from
# the normal manifold during simulated baseline hardening)
CONTRASTIVE_MARGIN = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# STGNN (Layer 2 — Contextual Validator)
# ─────────────────────────────────────────────────────────────────────────────

# Node feature dimensionality fed to the GNN
# Each node's feature vector = mean of its incident edge (flow) features
GNN_INPUT_DIM  = FEATURE_DIM
GNN_HIDDEN_DIM = 64
GNN_OUTPUT_DIM = 32          # Latent node embedding dimension
GNN_LEARNING_RATE = 5e-4
GNN_EPOCHS        = 50

# Maximum graph windows collected for STGNN training per-phase.
# Phase 2 seeds the GNN with this many windows from train_windows before
# Phase 4 streams additional attack CSVs. Both phases cap at this value so
# changing once here propagates everywhere. 100 is the research-grade default
# that balances training speed vs. GNN coverage.
GNN_ATTACK_GRAPH_CAP = 100

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC THRESHOLDING (Exponential Moving Average over batch MSE)
# ─────────────────────────────────────────────────────────────────────────────

# EMA smoothing factor (α). Higher = reacts faster but is noisier.
# Lower = more stable but slower to adapt.
EMA_ALPHA = 0.05

# An alert is raised when:  loss > EMA_mean + (EMA_SIGMA_MULTIPLIER × EMA_std)
EMA_SIGMA_MULTIPLIER = 0.5

# Warm-up batches before thresholds are active (avoids cold-start false alarms)
EMA_WARMUP_BATCHES = 3

# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY ENGINE — Temporal Accumulator + EMA Trajectory
# ─────────────────────────────────────────────────────────────────────────────

# Rolling window for per-node event accumulation (seconds).
# Events older than this are purged before escalation rules are evaluated.
TEMPORAL_WINDOW_SECONDS = 300   # 5 minutes — configurable

# EMA trajectory persistence threshold.
# K consecutive readings above 1.0σ → triggers GNN check.
# Set to 1 so every window that clears the sigma band is evaluated (no streak required).
K_CONSECUTIVE_READINGS  = 1     # configurable


# ─────────────────────────────────────────────────────────────────────────────
# FEDERATED LEARNING (Flower + Krum Aggregation)
# ─────────────────────────────────────────────────────────────────────────────

FL_SERVER_ADDRESS   = "localhost:8080"
FL_NUM_ROUNDS       = 3          # Federation rounds; final round hash is minted
FL_MIN_CLIENTS      = 5          # All 5 org clients contribute each round
FL_MIN_AVAILABLE    = 5          # All 5 orgs must be present before round 1
FL_EXPORT_HORIZON   = 32         # H: the AE update is exported at this optimizer step
FL_SIM_CLIENT_FLOWS = 12000      # Flows per simulated client: enough benign flows to reach H
FL_HEAD_EPOCHS      = 3          # Full-batch AttackHead steps per round (client and server root)
FL_LOCAL_EPOCHS     = 3          # Local AE training epochs per client per FL round (reduced for fast simulation)

# Krum: number of clients to select per round (must be ≤ total clients - 2)
# Krum drops the m clients whose weight updates are most distant from the median.
KRUM_NUM_TO_SELECT  = 2          # Select 2 from 3 mock clients (drops 1 straggler)

# Straggler policy: if a client doesn't respond within this many seconds, drop it
FL_ROUND_TIMEOUT_SEC = 30

# ─────────────────────────────────────────────────────────────────────────────
# FLTRUST ROOT DATASET SOURCE
# ─────────────────────────────────────────────────────────────────────────────

# "real"      → load a benign partition from NF-UNSW-NB15-v3 as the server's
#               trusted root dataset (best accuracy — recommended for paper).
# "synthetic" → fall back to Gaussian samples when the dataset is unavailable.
# The server always falls back to synthetic if the real load fails, so this is
# a soft preference, not a hard requirement.
FL_ROOT_DATA_SOURCE = "real"

# ─────────────────────────────────────────────────────────────────────────────
# FLTRUST AGGREGATION (replaces Krum — Upgrade 6)
# ─────────────────────────────────────────────────────────────────────────────

# Number of synthetic benign samples the server holds as its trusted root dataset.
# These are used to train the server model by one step each round so it computes
# a reference gradient direction for cosine trust scoring of client updates.
# Range: 100–500 recommended; lower = faster, higher = more robust server gradient.
FLTRUST_ROOT_SAMPLES   = 2000

# Learning rate used for the server's single-step root-dataset gradient update.
# Kept separate from AE_LEARNING_RATE so it can be tuned independently.
FLTRUST_SERVER_LR      = 1e-3

# DC-FLTrust reference buffer configuration
CH2_WARMUP_ROUNDS = 10          # rounds before AttackHead participates in federation
CH2_REF_BUFFER_MAX = 5000       # maximum z vectors in dynamic reference buffer
CH2_REF_BUFFER_MIN = 20         # minimum z vectors before dynamic reference activates

# Trust score at or below this value causes the client to be flagged as Byzantine
# in the detection log (fed into Upgrade 3).  ReLU already zeroes negatives;
# this threshold lets you also zero out near-zero trust scores from noisy clients.
FLTRUST_MIN_TRUST_SCORE = 0.0   # 0.0 = ReLU only (strict); raise to e.g. 0.05 to be stricter

# ch1 disambiguation threshold for dc_fltrust_aggregate.
# Clients with ch1 above this are treated as honest (HEALTHY or UNDER_ATTACK);
# clients below it are treated as Byzantine.
# The first two are the 5th percentile of honest client scores over 20 rounds at
# 100% training data and the P99 routing cut, channel 2 counted after warmup
# (scripts/final_federation.py). Recalibrate if G0, the routing cut, the round
# recipe or the client data change.
FLTRUST_CH1_THRESHOLD = 0.0173
CH2_ANTI_ALIGN_THRESHOLD = 0.492  # ch2 cosine below this: head update is adversarial
# Not calibrated, and nothing acts on the label it sets.
CH2_UNDER_ATTACK_THRESHOLD = 0.5  # ch2 cosine above this: client is under real attack

# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE ENGINE — Critical Infrastructure Allowlist
# ─────────────────────────────────────────────────────────────────────────────

# Tier-1 "never hard-isolate" nodes (simulated by synthetic node IDs)
# In production these would be real IP CIDRs or hostnames.
CRITICAL_ALLOWLIST = {
    "node_0":  "Domain Controller (AD)",
    "node_1":  "Core HR Database",
    "node_2":  "Payment Gateway",
    "node_3":  "SCADA / ICS Controller",
}

# ─────────────────────────────────────────────────────────────────────────────
# ORGANISATION NETWORK MAP (canonical — referenced by fl_server, client_state)
# ─────────────────────────────────────────────────────────────────────────────

# Maps org key → simulated LAN CIDR for all 5 federation clients.
# Used by the FL simulation console and client-state display.
# Must stay consistent with client_state.ALL_CLIENTS.
ORG_NETWORK_MAP: dict = {
    "hospital":   "192.168.1.0/24",
    "bank":       "10.0.1.0/24",
    "university": "172.16.1.0/24",
    "isp":        "10.10.0.0/24",
    "retail":     "172.31.0.0/24",
}


# ─────────────────────────────────────────────────────────────────────────────
# DIFFERENTIAL PRIVACY (Opacus DP-SGD)  — Tier 2.1
# ─────────────────────────────────────────────────────────────────────────────
#
# PrivacyEngine wraps the AE optimizer ONLY.  The AttackHead optimizer (when
# present in the DC-FLTrust two-pass architecture) is intentionally excluded
# because it trains on AE latent representations, not on raw private client
# data — wrapping it would be architecturally incorrect.
#
# Privacy budget semantics:
#   ε (epsilon)  : privacy loss — lower = stronger privacy, lower utility
#   δ (delta)    : probability of privacy failure — typically 1/dataset_size
#   σ (noise_multiplier) : controls ε; higher σ = smaller ε = stronger privacy
#
# DP_TARGET_EPSILON is a soft reporting target, NOT an enforced bound.
# The actual ε is computed by Opacus after training and logged per round.
# ─────────────────────────────────────────────────────────────────────────────

DP_ENABLED          = True
DP_MAX_GRAD_NORM    = 1.0    # Per-sample gradient clipping bound (L2 norm clip)
DP_NOISE_MULTIPLIER = 1.1    # σ — Gaussian noise scale; tune for target ε budget
DP_DELTA            = 1e-5   # δ — typically set to 1 / training_dataset_size
DP_TARGET_EPSILON   = 10.0   # Soft reporting target ε; actual ε is computed by Opacus

# Noise multiplier sweep values for Table 3 in the paper.
# σ=0.0 means no DP noise (baseline); higher σ = more privacy, lower utility.
# Sweep order: no-DP baseline → weak → moderate → strong → very strong.
DP_NOISE_SWEEP = [0.0, 0.5, 1.0, 1.5, 2.0]

# Confidence thresholds for the 3-tier response policy
CONFIDENCE_LOW_THRESHOLD  = 0.40   # Below this: log only
CONFIDENCE_MED_THRESHOLD  = 0.70   # Below this: throttle + HITL
# Above MED_THRESHOLD → full isolation for non-critical nodes

# ─────────────────────────────────────────────────────────────────────────────
# HITL THREE-TIER ESCALATION THRESHOLDS (Section 3.5)
# Every number from the Section 3.5 table is a named constant here.
# Zero magic numbers in any benchmark or engine script.
# ─────────────────────────────────────────────────────────────────────────────

# "3 LOWs within 5-min window" → MEDIUM (Section 3.5 trigger, row 2)
HITL_LOW_TO_MEDIUM_THRESHOLD  = 3

# "3 MEDIUMs within window" → HIGH (Section 3.5 trigger, row 3, second condition)
HITL_MEDIUM_TO_HIGH_THRESHOLD = 3

# "5 LOWs within window" → HIGH (Section 3.5 trigger, row 3, first condition)
HITL_LOW_TO_HIGH_THRESHOLD    = 5

# Seconds to wait for operator approval before degrading to auto-throttle (DEGRADED tier)
HITL_TIMEOUT_SEC              = 30

# Simulated operator approval probability for benchmark_hitl_response.py only [0.0, 1.0].
# 0.85 means the simulated operator approves 85% of HIGH-tier isolation requests.
# DEGRADED-tier rate in the benchmark = 1.0 - HITL_APPROVAL_RATE.
# Never used by the live response engine — only for reproducible offline evaluation.
HITL_APPROVAL_RATE            = 0.85

# Deduplication window for the response engine (seconds).
# Duplicate actions on the same node within this window are suppressed.
# Centralised here so benchmark and live engine use the same constant.
RESPONSE_DEDUP_WINDOW_SEC     = 30

# ─────────────────────────────────────────────────────────────────────────────
# BLOCKCHAIN / GANACHE (Immutable Audit Log)
# ─────────────────────────────────────────────────────────────────────────────

GANACHE_URL              = "http://127.0.0.1:7545"
CONTRACT_ADDRESS_FILE    = str(MODELS_DIR / "contract_address.txt")
CONTRACT_ABI_FILE        = str(CONTRACTS_DIR / "ModelRegistry.abi")

# If Ganache is not running, AURA falls back to local SHA-256 file logging
BLOCKCHAIN_FALLBACK_LOG  = str(LOGS_DIR / "blockchain_fallback.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# ISOLATION FOREST (Baseline Sanitisation)
# ─────────────────────────────────────────────────────────────────────────────

# Contamination: expected fraction of mislabelled / poisoned rows in the
# "normal" training split.  NF-UNSW-NB15-v3 is ~94.6% benign but we
# apply a small contamination rate defensively.
IF_CONTAMINATION = 0.02   # 2 % — removes extreme statistical outliers

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

DASHBOARD_REFRESH_INTERVAL_MS = 1500   # Streamlit auto-refresh period
MAX_ALERTS_CACHE              = 30     # Alerts kept in the dashboard session
MAX_INCIDENTS_CACHE           = 20     # Response incidents kept in the dashboard session
PENDING_INJECT_FRESHNESS_SEC  = 30     # pending_inject.json older than this is ignored
LAST_EXPLANATION_FRESHNESS_SEC = 30    # last_explanation.json older than this is ignored
CUSTOM_INJECT_AE_THRESHOLD    = 0.3    # ae_threshold recorded on CUSTOM_INJECT events
NODE_STATE_EVALUATING_SIMPLE  = "Evaluating…"
NODE_STATE_EVALUATING_ICON    = "⚡ Evaluating…"
ALERT_LOG_FILE = str(LOGS_DIR / "aura_alerts.jsonl")
EVENT_LOG_FILE = str(LOGS_DIR / "aura_events.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
# API SECURITY BLOCKLIST (custom injection endpoint)
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that, if found in a user-submitted injection script, cause the
# request to be rejected with HTTP 400.  Extend this list for production.
# This centralises the filter so it is easily auditable and not buried in
# api_server.py business logic.
SECURITY_BLOCKLIST: list = [
    "os.system",
    "os.popen",
    "subprocess",
    "import os",
    "import sys",
    "import subprocess",
    "__import__",
    "exec(",
    "eval(",
    "open(",
    "socket",
    "shutil",
    "pathlib",
    "requests",
    "urllib",
]

# ─────────────────────────────────────────────────────────────────────────────
# MITM SIMULATION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Standard deviation of the Gaussian noise injected into model weights during
# a simulated Man-in-the-Middle attack.  A small value is sufficient to change
# the SHA-256 hash (even 1-bit flip is detected) while remaining numerically
# similar enough to be a plausible interception scenario.
# Range: 0.001–0.05  |  Lower = subtler tampering, harder to detect by eye.
MITM_NOISE_STD: float = 0.01

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM INJECTION CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Feature name → model-input column index, generated from FEATURE_NAMES.
# Used by api_server._run_inject_inference() and the attack-corruption profiles
# to address features by name. Generated so it can never drift from the
# preprocessor's actual column order.
FEATURE_INDEX_MAP: dict = {name: i for i, name in enumerate(FEATURE_NAMES)}
# Short alias used by the injection profiles
FEATURE_INDEX_MAP["flow_duration"] = FEATURE_INDEX_MAP["flow_duration_milliseconds"]

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC AE THRESHOLD LOADER
# ─────────────────────────────────────────────────────────────────────────────
#
# The AE threshold MUST be derived from the benign traffic distribution, NOT a
# magic number. Specifically it is the EMA-adjusted UCL (μ + 3σ upper control
# limit) over benign reconstruction errors — see Section 3.1 in the paper.
#
# Resolution order:
#   1. logs/calibration_results.json  (written by calibrate_thresholds.py)
#   2. Loud WARNING + magic-number sentinel fallback
#
# Run `python calibrate_thresholds.py` once after training to produce the JSON.
# ─────────────────────────────────────────────────────────────────────────────

_CALIB_JSON_PATH = LOGS_DIR / "calibration_results.json"


# Sentinel fallback used ONLY before calibration has ever been run (e.g. during
# `calibrate_thresholds.py`'s own bootstrap, which needs to import config before
# it can produce the JSON this function reads). These are NOT research-valid
# thresholds — AE_THRESHOLDS_CALIBRATED tells the rest of the codebase whether
# the values in use are real or sentinel.
_SENTINEL_THRESHOLD_HIGH   = 0.05
_SENTINEL_THRESHOLD_MEDIUM = 0.02


def load_ae_thresholds() -> tuple[float, float, bool]:
    """
    Load MSE_THRESHOLD_HIGH and MSE_THRESHOLD_MEDIUM from calibration_results.json.

    Returns
    -------
    (threshold_high, threshold_medium, calibrated) where `calibrated` is False
    only when sentinel bootstrap values are returned because the calibration
    file doesn't exist yet. Any code path that needs research-valid thresholds
    (training, response engine, dashboards) should check this flag and refuse
    to proceed on sentinel values rather than silently using them.
    """
    if not _CALIB_JSON_PATH.exists():
        msg = (
            "[CONFIG] ⚠️  logs/calibration_results.json NOT FOUND. "
            "Falling back to SENTINEL AE thresholds so config.py can still be "
            "imported (this is required for calibrate_thresholds.py to bootstrap "
            "itself). These sentinel values are NOT valid for training, detection, "
            "or the paper. Run: `python calibrate_thresholds.py --train-quick` "
            "to generate real calibrated thresholds, then re-run your command."
        )
        _cfg_log.warning(msg)
        return _SENTINEL_THRESHOLD_HIGH, _SENTINEL_THRESHOLD_MEDIUM, False

    try:
        data = json.loads(_CALIB_JSON_PATH.read_text())
        high   = float(data["recommended_MSE_THRESHOLD_HIGH"])
        medium = float(data["recommended_MSE_THRESHOLD_MEDIUM"])

        p90 = data.get("p90", medium)
        p99 = data.get("p99", high)
        if abs(p99 - p90) < 0.002:
            _cfg_log.warning(
                "[CONFIG] ⚠️  AE threshold collapse detected: "
                f"P90={p90:.6f}  P99={p99:.6f}  gap={abs(p99-p90):.6f} < 0.002. "
                "Three-tier severity system is functionally degraded. "
                "Consider the parallel fusion architecture (AE + GNN scores combined) "
                "as documented in the Tier 2.5 experiment."
            )

        _cfg_log.info(
            f"[CONFIG] AE thresholds loaded from calibration JSON — "
            f"HIGH={high:.6f}  MEDIUM={medium:.6f}"
        )
        return high, medium, True

    except Exception as exc:
        msg = (
            f"[CONFIG] ❌ Failed to parse calibration_results.json: {exc}. "
            "Run: `python calibrate_thresholds.py` to regenerate it."
        )
        _cfg_log.error(msg)
        raise ValueError(msg) from exc


# Resolved at import time — every consumer (api_server, dashboard, dashboard_service)
# picks up the calibrated value automatically once calibration_results.json exists.
_ae_thresh_high, _ae_thresh_medium, AE_THRESHOLDS_CALIBRATED = load_ae_thresholds()

MSE_THRESHOLD_HIGH   = _ae_thresh_high    # EMA-UCL P99 of benign MSE distribution
MSE_THRESHOLD_MEDIUM = _ae_thresh_medium  # EMA-UCL P90 of benign MSE distribution

# Sentinel CH2 split used ONLY before calibration has ever been run, for the
# same bootstrap reason as _SENTINEL_THRESHOLD_* above (train.py and
# calibrate_thresholds.py must be able to import config on a fresh clone).
_SENTINEL_CH2_SPLIT_THRESHOLD = 0.01


def load_ch2_split_threshold() -> tuple[float, bool]:
    """
    Load CH2_MSE_SPLIT_THRESHOLD from calibration_results.json.

    Returns (threshold, calibrated). `calibrated` is False only when the
    calibration file does not exist yet and the sentinel is returned.
    DC-FLTrust entry points enforce calibration via preflight_dc_fltrust_check().
    """
    if not _CALIB_JSON_PATH.exists():
        _cfg_log.warning(
            "[CONFIG] ⚠️  logs/calibration_results.json NOT FOUND. Using SENTINEL "
            "CH2_MSE_SPLIT_THRESHOLD so config.py can be imported. Not valid for "
            "federation or the paper. Run: `python calibrate_thresholds.py`."
        )
        return _SENTINEL_CH2_SPLIT_THRESHOLD, False
    try:
        data = json.loads(_CALIB_JSON_PATH.read_text())
        if "CH2_MSE_SPLIT_THRESHOLD" not in data:
            raise KeyError("CH2_MSE_SPLIT_THRESHOLD missing from calibration file.")
        return float(data["CH2_MSE_SPLIT_THRESHOLD"]), True
    except Exception as e:
        raise ValueError(f"Failed to read CH2_MSE_SPLIT_THRESHOLD from calibration: {e}")

# DC-FLTrust two-pass split threshold
# Flows above this threshold are considered potentially anomalous 
# and routed to AttackHead for channel 2 analysis.
# Set to P99 of benign MSE, the same point as MSE_THRESHOLD_HIGH. A lower cut
# withholds hard benign flows from the AE every round, and its error on them grows
# until attacks fall below the alarm threshold (scripts/ae_routing_cut.py).
# Updated automatically by calibrate_thresholds.py alongside other thresholds.
CH2_MSE_SPLIT_THRESHOLD, CH2_SPLIT_CALIBRATED = load_ch2_split_threshold()

# ─────────────────────────────────────────────────────────────────────────────
# DATA-DRIVEN ATTACK CORRUPTION PROFILES
# ─────────────────────────────────────────────────────────────────────────────
#
# Each profile maps feature keys → (lo, hi) normalised perturbation ranges.
# These ranges MUST come from the actual NF-UNSW-NB15-v3 dataset, not from
# manually crafted guesses.  The canonical source is:
#   saved_models/attack_class_stats.json
# which is produced by scripts/train_explainer.py and stores per-class
# p05 / p95 percentiles for all 47 features.
#
# Resolution order:
#   1. saved_models/attack_class_stats.json  — real dataset percentiles
#   2. Hardcoded sentinel profiles           — fallback with loud WARNING
# ─────────────────────────────────────────────────────────────────────────────

# Map: config profile key → Attack column value in NF-UNSW-NB15-v3.csv
# Dataset Attack column uses: 'DoS', 'Exploits', 'Fuzzers', 'Reconnaissance',
# 'Backdoor', 'Generic', 'Shellcode', 'Analysis', 'Worms', 'Benign'
_ATTACK_STATS_KEY_MAP: dict = {
    "ddos":           "ddos",
    "lateral":        "lateral",
    "exfil":          "exfil",
    "port_scan":      "portscan",
    "web":            "web",
    "exploits":       "exploits",
    "fuzzers":        "fuzzers",
    "reconnaissance": "portscan",   # mapped to same class in stats JSON
    "backdoor":       "backdoor",
    "custom":         None,          # no real-data class; stays as sentinel
}

# Sentinel profiles — used ONLY when attack_class_stats.json is absent.
# These are hand-approximated and MUST NOT be cited in the paper as ground truth.
_SENTINEL_ATTACK_PROFILES: dict = {
    "ddos": {
        "in_pkts":                  (0.90, 0.99),
        "out_pkts":                 (0.85, 0.99),
        "src_to_dst_second_bytes":  (0.88, 0.99),
        "src_to_dst_iat_avg":       (0.00, 0.03),
        "src_to_dst_iat_stddev":    (0.00, 0.02),
        "tcp_flags":                (0.80, 0.99),
        "flow_duration":            (0.00, 0.05),
    },
    "lateral": {
        "flow_duration":            (0.50, 0.75),
        "in_pkts":                  (0.50, 0.65),
        "src_to_dst_iat_stddev":    (0.75, 0.95),
        "dst_to_src_iat_stddev":    (0.75, 0.95),
        "duration_in":              (0.80, 0.98),
        "duration_out":             (0.03, 0.10),
        "client_tcp_flags":         (0.60, 0.80),
    },
    "exfil": {
        "in_bytes":                 (0.88, 0.99),
        "out_bytes":                (0.00, 0.06),
        "src_to_dst_second_bytes":  (0.85, 0.99),
        "dst_to_src_second_bytes":  (0.00, 0.04),
        "src_to_dst_iat_avg":       (0.40, 0.55),
        "src_to_dst_iat_stddev":    (0.00, 0.03),
        "flow_duration":            (0.80, 0.98),
    },
    "port_scan": {
        "flow_duration":            (0.00, 0.03),
        "in_bytes":                 (0.00, 0.04),
        "out_bytes":                (0.00, 0.03),
        "tcp_flags":                (0.80, 0.99),
        "client_tcp_flags":         (0.70, 0.90),
        "src_to_dst_second_bytes":  (0.05, 0.15),
    },
    "web": {
        "in_bytes":                 (0.80, 0.95),
        "out_bytes":                (0.20, 0.30),
        "client_tcp_flags":         (0.85, 0.98),
        "server_tcp_flags":         (0.85, 0.98),
        "flow_duration":            (0.05, 0.12),
        "src_to_dst_iat_avg":       (0.02, 0.08),
    },
    "exploits": {
        "in_bytes":                 (0.70, 0.95),
        "longest_flow_pkt":         (0.85, 0.99),
        "tcp_flags":                (0.60, 0.85),
        "flow_duration":            (0.10, 0.30),
        "retransmitted_in_bytes":   (0.40, 0.70),
        "src_to_dst_iat_stddev":    (0.50, 0.80),
    },
    "fuzzers": {
        "in_bytes":                 (0.60, 0.90),
        "in_pkts":                  (0.70, 0.95),
        "longest_flow_pkt":         (0.70, 0.99),
        "shortest_flow_pkt":        (0.00, 0.05),
        "src_to_dst_iat_stddev":    (0.80, 0.99),
        "flow_duration":            (0.10, 0.40),
        "protocol":                 (0.80, 0.99),
    },
    "reconnaissance": {
        "flow_duration":            (0.00, 0.05),
        "in_bytes":                 (0.00, 0.08),
        "out_bytes":                (0.00, 0.06),
        "tcp_flags":                (0.70, 0.95),
        "num_pkts_up_to_128_bytes": (0.80, 0.99),
        "src_to_dst_second_bytes":  (0.05, 0.15),
    },
    "backdoor": {
        "flow_duration":            (0.60, 0.90),
        "in_bytes":                 (0.40, 0.65),
        "out_bytes":                (0.40, 0.65),
        "src_to_dst_iat_avg":       (0.50, 0.70),
        "src_to_dst_iat_stddev":    (0.00, 0.05),
        "dst_to_src_iat_avg":       (0.50, 0.70),
        "dst_to_src_iat_stddev":    (0.00, 0.05),
    },
    "custom": {
        # No real-data class for 'custom' — sentinel only.
        "longest_flow_pkt":         (0.85, 0.99),
        "shortest_flow_pkt":        (0.85, 0.99),
        "max_ip_pkt_len":           (0.85, 0.99),
        "min_ip_pkt_len":           (0.85, 0.99),
        "src_to_dst_iat_avg":       (0.02, 0.05),
        "src_to_dst_iat_stddev":    (0.92, 0.99),
        "protocol":                 (0.90, 0.99),
    },
}


def load_attack_corruption_profiles() -> dict:
    """
    Load attack corruption profiles derived from the NF-UNSW-NB15-v3 dataset.

    Reads saved_models/attack_class_stats.json (produced by
    scripts/train_explainer.py) and converts per-class (p05, p95) percentiles
    into the {feature_key: (lo, hi)} format expected by _run_inject_inference().

    The inverse-feature map (dataset column index → config key) is constructed
    from FEATURE_INDEX_MAP so there is a single source of truth for column
    ordering.

    Returns the sentinel profiles (with WARNING) if the JSON is absent.
    """
    stats_path = MODELS_DIR / "attack_class_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            "[CONFIG] attack_class_stats.json NOT FOUND. "
            "Channel 2 federation requires real data-derived profiles. "
            "Run: python scripts/train_explainer.py before benchmarking. "
            f"Expected path: {stats_path.resolve()}"
        )

    try:
        stats_data = json.loads(stats_path.read_text())
    except Exception as exc:
        raise ValueError(
            f"[CONFIG] Failed to parse attack_class_stats.json: {exc}. "
            "Channel 2 federation requires a valid data-derived profile JSON."
        )

    # Build index → feature-key reverse map from FEATURE_INDEX_MAP
    idx_to_key: dict[int, str] = {v: k for k, v in FEATURE_INDEX_MAP.items()}

    profiles: dict = {}
    for profile_key, stats_key in _ATTACK_STATS_KEY_MAP.items():
        if stats_key is None or stats_key not in stats_data:
            # Keep the sentinel for profiles without a real-data class
            profiles[profile_key] = _SENTINEL_ATTACK_PROFILES.get(profile_key, {})
            continue

        cls_stats = stats_data[stats_key]
        p05 = cls_stats.get("p05", [])
        p95 = cls_stats.get("p95", [])

        if not p05 or not p95:
            _cfg_log.warning(
                f"[CONFIG] attack_class_stats.json missing p05/p95 for class '{stats_key}'. "
                f"Using sentinel profile for '{profile_key}'."
            )
            profiles[profile_key] = _SENTINEL_ATTACK_PROFILES.get(profile_key, {})
            continue

        if len(p05) != FEATURE_DIM:
            # Stats were written for a different feature layout; positional
            # percentiles would be paired with the wrong feature names.
            _cfg_log.warning(
                f"[CONFIG] attack_class_stats.json has {len(p05)} features, model has "
                f"{FEATURE_DIM}. Re-run scripts/train_explainer.py. Using sentinel "
                f"profile for '{profile_key}'."
            )
            profiles[profile_key] = _SENTINEL_ATTACK_PROFILES.get(profile_key, {})
            continue

        # Convert: only include features present in FEATURE_INDEX_MAP
        feature_profile: dict = {}
        for feat_idx, feat_key in idx_to_key.items():
            if feat_idx < len(p05) and feat_idx < len(p95):
                lo = round(float(p05[feat_idx]), 6)
                hi = round(float(p95[feat_idx]), 6)
                # Only include features where the attack class actually differs
                # from the trivial [0,1] range (skip uninformative features)
                if hi > lo:
                    feature_profile[feat_key] = (lo, hi)

        profiles[profile_key] = feature_profile
        _cfg_log.info(
            f"[CONFIG] Loaded dataset-derived profile for '{profile_key}' "
            f"({stats_key}): {len(feature_profile)} features from NF-UNSW-NB15-v3."
        )

    # Always include the sentinel 'custom' profile unchanged
    if "custom" not in profiles:
        profiles["custom"] = _SENTINEL_ATTACK_PROFILES["custom"]

    return profiles


# Resolved at import time — derived from NF-UNSW-NB15-v3 via attack_class_stats.json.
# Falls back to sentinel profiles with WARNING if the JSON is absent.
ATTACK_CORRUPTION_PROFILES: dict = load_attack_corruption_profiles()


def preflight_dc_fltrust_check():
    """Call at the top of any script using DC-FLTrust before round 1."""
    import os, json
    from datetime import datetime
    
    stats_path = MODELS_DIR / "attack_class_stats.json"

    if not stats_path.exists():
        raise RuntimeError(
            f"[PREFLIGHT FAIL] attack_class_stats.json missing at {stats_path}. "
            "Run train_explainer.py first."
        )

    if not (AE_THRESHOLDS_CALIBRATED and CH2_SPLIT_CALIBRATED):
        raise RuntimeError(
            f"[PREFLIGHT FAIL] {_CALIB_JSON_PATH} missing — AE / CH2 thresholds are "
            "sentinel values. Run train.py then calibrate_thresholds.py first."
        )
    
    mtime = os.path.getmtime(stats_path)
    mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    
    with open(stats_path) as f:
        profiles = json.load(f)
    
    print(f"[PREFLIGHT PASS] attack_class_stats.json found.")
    print(f"  Modified: {mtime_str}")
    print(f"  Attack classes present: {list(profiles.keys())}")
    print(f"  This timestamp should match your most recent train_explainer.py run.")
    return True
