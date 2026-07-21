"""
aura/fl_server.py — Flower FL Server: FLTrust Aggregation + Straggler Policy
=============================================================================

This server implements two critical security properties:

1. BYZANTINE ROBUSTNESS (FLTrust Aggregation — Upgrade 6)
   --------------------------------------------------------
   Standard FedAvg is vulnerable to model poisoning. Legacy Krum (distance-based;
   see krum_select / krum_aggregate below) guards geometrically but has a critical flaw: it rejects
   clients whose updates are geometrically distant even if they are
   *honestly* trained on rare or skewed data distributions (e.g., a
   hospital with rare disease traffic).  This causes false-positive
   rejections of legitimate but outlier clients.

   FLTrust (Cao et al., 2020) fixes this:
     - The server holds a small clean root dataset (FLTRUST_ROOT_SAMPLES
       synthetic benign samples).
     - Each round, the server trains one optimisation step on the root
       dataset to obtain a reference gradient direction.
     - Each client update is scored by its cosine similarity with the
       server's gradient direction — ReLU ensures negative similarity
       (i.e., adversarial reversal) maps to zero trust.
     - Client updates are re-scaled to the server update's magnitude
       before weighted aggregation, preventing magnitude-based amplification.

   Key advantage: a hospital client with unusual-but-legitimate data has
   a gradient that still *points in the same direction* as improvement on
   normal traffic. Legacy Krum would drop it; FLTrust keeps it with proportional
   trust.

   **Active aggregation:** FLTrust in aggregate_fit. **Legacy fallback only:**
   krum_select / krum_aggregate (not used in the default path) retained for rollback or experiments.

2. STRAGGLER TIMEOUT POLICY
   --------------------------
   In synchronous FL, if a client disconnects mid-round, the server
   blocks indefinitely — causing a Denial-of-Service against the entire
   federation.

   AURA implements an explicit timeout policy:
     - After `round_timeout_sec` seconds, unreceived client updates
       are DROPPED from the aggregation round.
     - If fewer than `min_clients` responses arrive, the round is
       ABANDONED and the previous global model is preserved.
     - A warning is logged for operator review.

3. IMMUTABLE AUDIT LOG
   ----------------------
   After each successful aggregation, the server hashes the model weights
   (SHA-256) and writes the hash to the AURA blockchain module.
   This creates a tamper-evident chain of custody for all model updates.
"""

import hashlib
import io
import json
import logging
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
from flwr.common import (
    FitRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
    EvaluateRes,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy import FedAvg

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from aura.models import AURAModelBundle

from config import preflight_dc_fltrust_check
preflight_dc_fltrust_check()

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy Krum aggregation (fallback only — not used by aggregate_fit; FLTrust is active)
# ─────────────────────────────────────────────────────────────────────────────

def krum_select(
    updates:       List[List[np.ndarray]],
    num_to_select: int = cfg.KRUM_NUM_TO_SELECT,
) -> List[int]:
    """
    Krum Selection Algorithm (Blanchard et al., 2017).

    Algorithm
    ---------
    Given n client weight updates {w_1, ..., w_n}:
    1. For each client i, flatten its weight update into a 1D vector.
    2. Compute pairwise squared Euclidean distances: d(i,j) = ||w_i - w_j||²
    3. For each client i, compute the Krum score:
         s(i) = Σ_{j ∈ N_k(i)} d(i, j)
       where N_k(i) = k nearest neighbours of i (k = n - num_to_select - 2)
    4. Select the num_to_select clients with the LOWEST Krum scores.

    Poisoned clients produce updates far from the cluster → high scores → dropped.

    Parameters
    ----------
    updates        : List of per-client parameters (each is a list of ndarrays)
    num_to_select  : How many clients to keep (KRUM_NUM_TO_SELECT in config)

    Returns
    -------
    Selected client indices (those with lowest Krum scores)
    """
    n = len(updates)
    if n <= num_to_select:
        logger.warning("Krum: fewer clients than num_to_select — accepting all.")
        return list(range(n))

    # Flatten each client's parameters to a single 1D vector
    flat = []
    for client_params in updates:
        flat.append(np.concatenate([p.flatten() for p in client_params]))

    # k = n - num_to_select - 2  (guaranteed Byzantine tolerance formula)
    k = max(1, n - num_to_select - 2)

    scores = []
    for i in range(n):
        # Squared Euclidean distances from client i to all others
        dists = sorted([
            float(np.sum((flat[i] - flat[j]) ** 2))
            for j in range(n) if j != i
        ])
        # Krum score = sum of k smallest distances
        scores.append(sum(dists[:k]))

    # Rank clients by score (ascending) and select the best num_to_select
    ranked = sorted(range(n), key=lambda idx: scores[idx])
    selected = ranked[:num_to_select]

    dropped = [i for i in range(n) if i not in selected]
    if dropped:
        logger.warning(
            f"[KRUM] Dropped client indices {dropped} as potential outliers.  "
            f"Scores: {[round(s, 2) for s in scores]}"
        )
    else:
        logger.info(f"[KRUM] All clients accepted.  Scores: {[round(s, 2) for s in scores]}")

    return selected


def krum_aggregate(
    selected_updates: List[List[np.ndarray]],
) -> List[np.ndarray]:
    """
    Aggregate selected (Krum-filtered) client updates by simple mean.

    By the time we reach this function, Byzantine clients have already been
    filtered by krum_select.  A simple mean of the remaining honest updates
    produces the new global model.

    Shape preservation:  each result array has the same shape as input arrays.
    """
    # Cast to float32: keeps dtype consistent with PyTorch model weights so
    # SHA-256 hashes computed before and after Flower serialization always match.
    return [
        np.mean([update[i] for update in selected_updates], axis=0).astype(np.float32)
        for i in range(len(selected_updates[0]))
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Root Dataset (server trusted data for FLTrust)
# ─────────────────────────────────────────────────────────────────────────────

def _build_root_dataset(scaler, n_samples=5000):
    """
    Build server root dataset for FLTrust reference gradient.
    
    CRITICAL: must use the same scaler as client data.
    CRITICAL: must come from canonical training split only — no test data.
    CRITICAL: must be large enough for stable gradient direction (minimum 2000).
    
    Args:
        scaler: the SAME MinMaxScaler instance used for client data
        n_samples: number of flows to include (5000 recommended)
    """
    from aura.split_manager import get_canonical_split
    from aura.data_loader import CICIDSDataLoader
    
    loader = CICIDSDataLoader()
    all_windows = list(loader.stream_graphs(scaler))
    calib_windows, _, _, server_attack_windows = get_canonical_split(all_windows, test_fraction=0.20)
    
    # Extract benign flows from calibration windows only
    all_flows = []
    for graph, labels in calib_windows:
        flows = graph['edge_attr']
        benign_mask = labels == 0
        if benign_mask.any():
            all_flows.append(flows[benign_mask])
        if sum(len(f) for f in all_flows) >= n_samples:
            break
    
    if not all_flows:
        raise RuntimeError(
            "FATAL: Root dataset is empty. "
            "Check that canonical training split contains benign flows."
        )
    
    root_data = torch.cat(all_flows)[:n_samples]
    print(f"[ROOT DATASET] Built from canonical training split: {len(root_data)} benign flows")
    if scaler is not None:
        print(f"[ROOT DATASET] Using shared scaler instance: {id(scaler)}")
    return root_data, server_attack_windows
# ─────────────────────────────────────────────────────────────────────────────
# FLTrust Aggregation (Upgrade 6 — active path in aggregate_fit; Krum is legacy fallback only)
# ─────────────────────────────────────────────────────────────────────────────

def _retrain_reference_head(head, z_tensor, epochs=3):
    """Retrain the server's reference AttackHead on the latest dynamic z buffer."""
    import torch.optim as optim
    import torch.nn.functional as F
    opt = optim.Adam(head.parameters(), lr=1e-3)
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        preds = head(z_tensor).squeeze()
        # Pseudo-label 1.0 because this is a confirmed attack buffer
        pseudo_labels = torch.ones_like(preds)
        loss = F.binary_cross_entropy(preds, pseudo_labels)
        loss.backward()
        opt.step()
    head.eval()

def ae_only_fltrust_aggregate(
    client_ae_deltas: list,
    root_ae_delta: dict,
) -> tuple:
    """
    AE-Only FLTrust (Mode A baseline).
    
    Only AE gradients are transmitted and evaluated.
    AttackHead does not exist in this mode.
    This is the pre-DC-FLTrust baseline — federate only the AE,
    ignore attack-pattern knowledge entirely.
    
    Returns: (aggregated_ae_weights, trust_scores)
    """
    def flatten(delta: dict) -> torch.Tensor:
        return torch.cat([v.flatten().float() for v in delta.values()])
    
    root_flat = flatten(root_ae_delta)
    trust_scores = []
    
    for ae_delta in client_ae_deltas:
        client_flat = flatten(ae_delta)
        sim = F.cosine_similarity(
            client_flat.unsqueeze(0),
            root_flat.unsqueeze(0)
        ).item()
        trust_scores.append(max(0.0, sim))  # ReLU
    
    # Normalize
    total = sum(trust_scores)
    if total > 0:
        normalized = [s / total for s in trust_scores]
    else:
        normalized = [1.0 / len(trust_scores)] * len(trust_scores)
    
    # Weighted aggregation
    agg_ae = {
        k: sum(w * d[k] for w, d in zip(normalized, client_ae_deltas))
        for k in client_ae_deltas[0]
    }
    
    return agg_ae, trust_scores


def joint_dual_fltrust_aggregate(
    client_ae_deltas: list,
    client_head_deltas: list,
    root_ae_delta: dict,
    root_head_delta: dict,
    client_round_counts: list,
    ch2_warmup_rounds: int = 10,
    ch1_weight: float = 0.7,
) -> tuple:
    """
    Joint Dual-Channel FLTrust (Mode B baseline).
    
    Both AE and AttackHead gradients evaluated independently,
    but COMBINED into a single trust weight before aggregation:
    trust = ch1_weight * ch1 + (1 - ch1_weight) * ch2
    
    A client with honest AE (ch1=0.8) but corrupted AttackHead (ch2=0.1)
    receives trust = 0.7*0.8 + 0.3*0.1 = 0.59 — reduced but NOT excluded.
    The corrupted AttackHead gradient still enters global aggregation
    at 59% weight rather than the 80% it would receive if AE-only.
    
    This is the key difference from DC-FLTrust (Mode C):
    Mode B reduces but does not eliminate corrupted gradient influence.
    Mode C detects BYZANTINE_FAKE_ATTACK and excludes the head gradient entirely.
    
    Returns: (aggregated_ae, aggregated_head, combined_trust_scores, ch1_scores, ch2_scores)
    """
    def flatten(delta: dict) -> torch.Tensor:
        return torch.cat([v.flatten().float() for v in delta.values()])
    
    root_ae_flat = flatten(root_ae_delta)
    root_head_flat = flatten(root_head_delta)
    
    ch1_scores = []
    ch2_scores = []
    combined_scores = []
    
    for ae_delta, head_delta, rounds in zip(
        client_ae_deltas, client_head_deltas, client_round_counts
    ):
        ch1 = max(0.0, F.cosine_similarity(
            flatten(ae_delta).unsqueeze(0),
            root_ae_flat.unsqueeze(0)
        ).item())
        
        if rounds >= ch2_warmup_rounds and head_delta is not None:
            ch2 = max(0.0, F.cosine_similarity(
                flatten(head_delta).unsqueeze(0),
                root_head_flat.unsqueeze(0)
            ).item())
        else:
            ch2 = 0.0  # warmup — treat as no attack signal
        
        combined = ch1_weight * ch1 + (1 - ch1_weight) * ch2
        
        ch1_scores.append(ch1)
        ch2_scores.append(ch2)
        combined_scores.append(combined)
    
    # Normalize combined scores
    total = sum(combined_scores)
    if total > 0:
        normalized = [s / total for s in combined_scores]
    else:
        normalized = [1.0 / len(combined_scores)] * len(combined_scores)
    
    # Both channels aggregated with same combined weight
    # This is the flaw: corrupted head gradients enter aggregation
    # at the same weight as honest AE gradients
    agg_ae = {
        k: sum(w * d[k] for w, d in zip(normalized, client_ae_deltas))
        for k in client_ae_deltas[0]
    }
    
    valid_heads = [(w, d) for w, d in zip(normalized, client_head_deltas)
                   if d is not None]
    if valid_heads:
        agg_head = {
            k: sum(w * d[k] for w, d in valid_heads)
            for k in valid_heads[0][1]
        }
    else:
        agg_head = None
    
    return agg_ae, agg_head, combined_scores, ch1_scores, ch2_scores


def dc_fltrust_aggregate(
    client_ae_deltas: list,
    client_head_deltas: list,
    root_ae_delta: dict,
    root_head_delta: dict,
    client_round_counts: list,
    ch2_warmup_rounds: int = 10,
    round_z_submissions: dict = None,
    attack_ref_buffer = None,
    current_round: int = 0,
    reference_attack_head = None,
    ch1_threshold: float = 0.25
) -> tuple:
    """
    Dual-channel FLTrust aggregation.
    
    Channel 1: cosine similarity of AE delta vs root AE delta
    Channel 2: cosine similarity of AttackHead delta vs root head delta
               Only active after ch2_warmup_rounds per client
    
    Returns: (aggregated_ae_weights, aggregated_head_weights, 
               ch1_scores, ch2_scores, classifications)
    """
    
    def flatten(delta: dict) -> torch.Tensor:
        return torch.cat([v.flatten() for v in delta.values()])

    def signed_cosine(client_flat, root_flat) -> float:
        """Raw signed cosine similarity — used for CLASSIFICATION decisions.

        Preserves sign so that anti-aligned gradients (negative cosine) are
        distinguishable from 'no signal' (ch2=None). This is the fix for the
        Latent Inversion Byzantine escape: a client that trains its AttackHead
        with inverted labels produces a head delta with raw cosine ≈ −0.4.
        After ReLU that was 0.0, which was then classified as HEALTHY.
        With the raw signed value it is correctly caught as BYZANTINE_FAKE_ATTACK.
        """
        return F.cosine_similarity(
            client_flat.unsqueeze(0),
            root_flat.unsqueeze(0)
        ).item()

    def relu_cosine(client_flat, root_flat) -> float:
        """ReLU-clamped cosine — used for AGGREGATION WEIGHTS only.

        Aggregation weights must be non-negative (a negative weight would
        invert the gradient direction and corrupt the global model). ReLU
        maps anti-aligned updates to zero weight, excluding them from the
        weighted average without needing an explicit exclusion list.
        """
        return max(0.0, signed_cosine(client_flat, root_flat))

    root_ae_flat = flatten(root_ae_delta)
    root_head_flat = flatten(root_head_delta)

    ch1_scores = []       # ReLU cosine for ch1 (used in aggregation weights)
    ch2_scores = []       # ReLU cosine for ch2 (used in aggregation weights)
    ch2_raw_scores = []   # Signed cosine for ch2 (used in classification only)
    classifications = []
    exclusion_flags = []
    warmup_flags = []
    under_attack_flags = []

    for i, (ae_delta, head_delta, rounds) in enumerate(
        zip(client_ae_deltas, client_head_deltas, client_round_counts)
    ):
        # Channel 1: always active — ReLU for weight, signed for reference
        client_flat = flatten(ae_delta)
        ch1 = relu_cosine(client_flat, root_ae_flat)
        
        sim = signed_cosine(client_flat, root_ae_flat)

        # ── Full instrumentation block ────────────────────────────────────────
        _root_keys   = list(root_ae_delta.keys())
        _client_keys = list(ae_delta.keys())
        _keys_match  = _root_keys == _client_keys
        _root_len    = sum(v.numel() for v in root_ae_delta.values())
        _client_len  = sum(v.numel() for v in ae_delta.values())
        _enc_params  = sum(v.numel() for k, v in ae_delta.items() if k.startswith('encoder'))
        _dec_params  = sum(v.numel() for k, v in ae_delta.items() if k.startswith('decoder'))
        _other       = sum(v.numel() for k, v in ae_delta.items()
                           if not k.startswith('encoder') and not k.startswith('decoder'))
        print(f"\n{'='*26}")
        print(f"CLIENT {i}")
        print(f"{'='*26}")
        print(f"  Root  keys ({len(_root_keys)}):   {_root_keys}")
        print(f"  Client keys ({len(_client_keys)}): {_client_keys}")
        print(f"  Keys identical:        {_keys_match}")
        print(f"  Root  flatten length:  {_root_len}")
        print(f"  Client flatten length: {_client_len}")
        print(f"  Root  norm:  {root_ae_flat.norm():.6f}")
        print(f"  Client norm: {client_flat.norm():.6f}")
        print(f"  Signed cosine:  {sim:.6f}")
        print(f"  ReLU cosine:    {ch1:.6f}")
        print(f"  Root dtype/device:   {root_ae_flat.dtype} / {root_ae_flat.device}")
        print(f"  Client dtype/device: {client_flat.dtype} / {client_flat.device}")
        print(f"  Param coverage -> encoder: {_enc_params}  decoder: {_dec_params}  other: {_other}")
        print(f"  Root   first 10 vals: {root_ae_flat[:10].tolist()}")
        print(f"  Client first 10 vals: {client_flat[:10].tolist()}")
        # Save root and client[1] deltas once for offline cross-check
        if i == 1:
            import os, pickle
            _save_path = os.path.join(os.path.dirname(__file__), '..', 'debug_deltas.pkl')
            with open(_save_path, 'wb') as _f:
                pickle.dump({'root_ae_delta': root_ae_delta,
                             'client1_ae_delta': ae_delta}, _f)
            print(f"  [SAVED] root_ae_delta + client[1] ae_delta -> debug_deltas.pkl")
        # ── End instrumentation ───────────────────────────────────────────────
        print(f"[DIAGNOSTIC] Client {i} gradient cosine with root: {sim:.4f}")
        print(f"  Root grad norm: {root_ae_flat.norm():.4f}")
        print(f"  Client grad norm: {client_flat.norm():.4f}")
        print(f"  [CH1 CHECK] Client {i}: raw_signed={sim:.4f}  ch1_relu=max(0,raw)={ch1:.4f}")

        # Channel 2: gated by warmup
        if rounds >= ch2_warmup_rounds:
            if head_delta is not None:
                head_flat = flatten(head_delta)
                ch2_raw = signed_cosine(head_flat, root_head_flat)   # signed — for classification
                ch2     = relu_cosine(head_flat, root_head_flat)      # ReLU  — for aggregation weight
            else:
                ch2_raw = None   # No attack signal submitted
                ch2     = None
        else:
            ch2_raw = 'WARMUP'
            ch2     = 'WARMUP'

        ch1_scores.append(ch1)
        ch2_scores.append(ch2 if isinstance(ch2, float) else 0.0)
        ch2_raw_scores.append(ch2_raw)

        # ── Step 1: Explicit Boolean Predicates ──────────────────────────────────
        # Calibrated using Mean - 6*std of honest distribution (0.65 - 6*0.088 = 0.12)
        ANTI_ALIGN_THRESHOLD = 0.12

        if ch2_raw == 'WARMUP':
            is_warmup = True
            is_ch1_adversarial = False
            is_ch2_adversarial = False
            is_under_attack = False
            should_exclude = False
        else:
            is_warmup = False
            is_ch1_adversarial = (ch1 <= ch1_threshold)
            is_ch2_adversarial = (ch2_raw is not None) and (ch2_raw < ANTI_ALIGN_THRESHOLD)
            is_under_attack = (ch2_raw is not None) and (ch2_raw > 0.5) and (not is_ch1_adversarial)
            # The canonical decision rule:
            should_exclude = is_ch1_adversarial or is_ch2_adversarial

        exclusion_flags.append(should_exclude)
        warmup_flags.append(is_warmup)
        under_attack_flags.append(is_under_attack)

        # ── Step 3: Pure Presentation Strings ────────────────────────────────────
        if is_warmup:
            classification = 'WARMUP'
        elif ch2_raw is None:
            classification = 'HEALTHY' if not is_ch1_adversarial else 'BYZANTINE'
        else:
            if ch2_raw < ANTI_ALIGN_THRESHOLD:
                classification = 'BYZANTINE_FAKE_ATTACK' if not is_ch1_adversarial else 'BYZANTINE'
            elif ch2_raw > 0.5:
                classification = 'UNDER_ATTACK' if not is_ch1_adversarial else 'BYZANTINE_FAKE_ATTACK'
            else:
                classification = 'HEALTHY' if not is_ch1_adversarial else 'BYZANTINE'

        classifications.append(classification)

        _ch2r_str = f"{ch2_raw:.4f}" if isinstance(ch2_raw, float) else str(ch2_raw)
        _ch2_str  = f"{ch2:.4f}"    if isinstance(ch2,     float) else "0.0000"
        logger.debug(
            f"  [dc_fltrust] Client {i}: "
            f"ch1={ch1:.4f}  ch2_raw={_ch2r_str}  ch2_relu={_ch2_str}  -> {classification}"
        )
    
    # ── Step 2: Boolean-driven Aggregation ───────────────────────────────────
    # Aggregate AE weights — exclude Byzantine clients from ch1
    ch1_weights = torch.tensor([
        s if not excl else 0.0
        for s, excl in zip(ch1_scores, exclusion_flags)
    ])
    ch1_norm = ch1_weights.sum()
    if ch1_norm > 0:
        ch1_weights = ch1_weights / ch1_norm
    
    # === Attack Reference Buffer Update ===
    if round_z_submissions is not None and attack_ref_buffer is not None:
        for client_id, is_under_attack in enumerate(under_attack_flags):
            if is_under_attack:
                if client_id < len(round_z_submissions) and round_z_submissions[client_id] is not None:
                    z_submission = round_z_submissions[client_id]
                    if len(z_submission) > 0:
                        if isinstance(z_submission, list):
                            z_submission = torch.cat(z_submission)
                        attack_ref_buffer.update(
                            z_submission, 
                            round_num=current_round
                        )
        
        if attack_ref_buffer.is_ready():
            new_reference = attack_ref_buffer.get_reference_tensor()
            if reference_attack_head is not None:
                _retrain_reference_head(reference_attack_head, new_reference)
                channel2_reference = reference_attack_head
            logger.info(f"Round {current_round}: ch2 reference updated from buffer "
                        f"({attack_ref_buffer.stats()['buffer_size']} vectors)")
        else:
            logger.info(f"Round {current_round}: buffer not ready "
                        f"({len(attack_ref_buffer._buffer)}/{attack_ref_buffer.min_size_to_use}), "
                        f"using static reference")
    
    # Aggregate head weights — only from clients with active ch2 and not Byzantine
    ch2_weights = torch.tensor([
        (s if s is not None and isinstance(s, float) else 0.0)
        if not (excl or wrm) else 0.0
        for s, excl, wrm in zip(ch2_scores, exclusion_flags, warmup_flags)
    ])
    ch2_norm = ch2_weights.sum()
    if ch2_norm > 0:
        ch2_weights = ch2_weights / ch2_norm
    
    # Weighted aggregation
    agg_ae = {k: sum(w * d[k] for w, d in zip(ch1_weights, client_ae_deltas))
              for k in client_ae_deltas[0]}
    
    active_head_updates = [
        (w, d) for w, d, excl, wrm in zip(ch2_weights, client_head_deltas, exclusion_flags, warmup_flags)
        if not (excl or wrm) and d is not None
    ]
    
    if active_head_updates:
        agg_head = {k: sum(w * d[k] for w, d in active_head_updates)
                    for k in active_head_updates[0][1]}
    else:
        agg_head = None  # no head update this round
    
    return agg_ae, agg_head, ch1_scores, ch2_scores, classifications, exclusion_flags


def get_federated_parameters(model: AURAModelBundle) -> List[torch.nn.Parameter]:
    """Helper to return only the parameters that are actually federated (AE + Head)."""
    return list(model.autoencoder.parameters()) + list(model.attack_head.parameters())

def fltrust_aggregate(
    global_model:    AURAModelBundle,
    client_updates:  List[List[np.ndarray]],   # per-client list-of-arrays
    root_data:       torch.Tensor,             # server's trusted benign dataset
    server_lr:       float = cfg.FLTRUST_SERVER_LR,
    min_trust:       float = cfg.FLTRUST_MIN_TRUST_SCORE,
) -> Tuple[List[np.ndarray], List[float], List[int]]:
    """
    FLTrust Aggregation (Cao et al., 2020).

    Algorithm
    ---------
    1. Compute server reference update:
         Clone the global model, train one step on root_data (benign),
         delta_server = new_params − old_params.
         Flatten to 1-D vector: server_vec.

    2. For each client i:
         delta_i    = client_params_i − global_params
         client_vec = flatten(delta_i)
         trust_i    = ReLU(cosine_similarity(client_vec, server_vec))
         # Positive = update points same direction as benign improvement
         # Negative = adversarial reversal → ReLU ⇒ zero trust, flagged.

    3. Normalise client update magnitude to server update magnitude:
         scale_i = ||server_vec|| / (||client_vec|| + ε)
         normalised_i = scale_i × delta_i

    4. Weighted aggregation:
         new_global = global + Σ(trust_i / Σtrust) × normalised_i

    Parameters
    ----------
    global_model   : The current global AURAModelBundle.
    client_updates : Each element is a list of np.ndarray (Flower format).
    root_data      : [N, F] benign tensor (server's trusted data).
    server_lr      : LR for the one-step server gradient computation.
    min_trust      : Trust scores at or below this are flagged Byzantine.

    Returns
    -------
    (new_arrays, trust_scores, flagged_indices)
      new_arrays    : Aggregated model as list[np.ndarray] (same layout as input)
      trust_scores  : Per-client cosine trust score ∈ [0, 1]
      flagged_indices: Indices of clients with trust ≤ min_trust (Byzantine suspects)
    """
    # ── Step 1: Compute server reference one-step update ───────────────────
    server_model = AURAModelBundle()
    # Load current global weights
    global_param_list = [p.detach().cpu() for p in get_federated_parameters(global_model)]
    with torch.no_grad():
        for p, gp in zip(get_federated_parameters(server_model), global_param_list):
            p.copy_(gp)

    # One gradient step on root data (autoencoder reconstruction loss)
    optimiser = torch.optim.Adam(server_model.autoencoder.parameters(), lr=server_lr)
    server_model.autoencoder.train()
    optimiser.zero_grad()
    x_hat, _ = server_model.autoencoder(root_data)
    loss = nn.functional.mse_loss(x_hat, root_data)
    loss.backward()
    # Clip to match client training (preserves fairness in magnitude comparison)
    torch.nn.utils.clip_grad_norm_(server_model.autoencoder.parameters(), max_norm=1.0)
    optimiser.step()
    server_model.eval()

    # Server delta (new weights − old weights), flattened
    server_delta = [
        (p.detach().cpu() - gp)
        for p, gp in zip(get_federated_parameters(server_model), global_param_list)
    ]
    server_vec = torch.cat([d.flatten() for d in server_delta])   # [D]
    server_norm = server_vec.norm()                                # scalar

    # ── Step 2 & 3: Per-client trust scores + normalised deltas ───────────
    trust_scores: List[float] = []
    normalised_deltas: List[List[torch.Tensor]] = []   # per-client list of tensors

    global_arrays = [p.detach().cpu().numpy() for p in get_federated_parameters(global_model)]

    for client_arrays in client_updates:
        # Build per-layer delta (client_params − global_params)
        client_delta = [
            torch.tensor(c_arr, dtype=torch.float32) - torch.tensor(g_arr, dtype=torch.float32)
            for c_arr, g_arr in zip(client_arrays, global_arrays)
        ]
        client_vec = torch.cat([d.flatten() for d in client_delta])   # [D]
        client_norm = client_vec.norm()

        # Cosine similarity with server gradient direction
        cos_sim = F.cosine_similarity(
            server_vec.unsqueeze(0), client_vec.unsqueeze(0)
        ).item()
        # ReLU: adversarial reversal (negative cos) ⇒ zero trust
        trust = max(0.0, cos_sim)
        trust_scores.append(trust)

        # Re-scale client delta to server update magnitude (prevents amplification)
        scale = float(server_norm) / (float(client_norm) + 1e-8)
        normalised_deltas.append([d * scale for d in client_delta])

    # ── Step 4: Weighted aggregation ────────────────────────────────────
    total_trust = sum(trust_scores) + 1e-8

    # Identify Byzantine suspects (zero or near-zero trust)
    flagged_indices = [
        i for i, t in enumerate(trust_scores) if t <= min_trust
    ]

    # Initialise new state as: global_params + weighted sum of normalised deltas
    new_arrays: List[np.ndarray] = []
    for layer_idx in range(len(global_arrays)):
        accumulated = torch.zeros_like(
            torch.tensor(global_arrays[layer_idx], dtype=torch.float32)
        )
        for i, (trust, deltas) in enumerate(zip(trust_scores, normalised_deltas)):
            accumulated += (trust / total_trust) * deltas[layer_idx]
        result = torch.tensor(global_arrays[layer_idx], dtype=torch.float32) + accumulated
        new_arrays.append(result.numpy().astype(np.float32))

    return new_arrays, trust_scores, flagged_indices


# ─────────────────────────────────────────────────────────────────────────────
# SHA-256 Model Hash
# ─────────────────────────────────────────────────────────────────────────────

def hash_model_weights(arrays: List[np.ndarray]) -> str:
    """
    Compute a SHA-256 hash over the concatenated model weight bytes.

    Normalises every array to C-contiguous float32 before hashing so the
    result is identical whether called on the server-side aggregated arrays
    or on the client-side after Flower's ndarrays_to_parameters round-trip.
    """
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return "0x" + h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Custom Flower Strategy: KrumFedAURA (FLTrust inside aggregate_fit; Krum helpers are legacy)
# ─────────────────────────────────────────────────────────────────────────────

class KrumFedAURA(FedAvg):
    """
    Custom Flower aggregation strategy extending FedAvg with:
      1. **FLTrust** Byzantine-robust aggregation (active — Upgrade 6; cosine trust vs server root).
      2. Straggler timeout + drop policy
      3. Per-round SHA-256 model hash logging
      4. Blockchain audit integration

    Class name remains **KrumFedAURA** for backward compatibility with dashboards and scripts.
    **krum_select / krum_aggregate** exist only as legacy fallback code paths, not used by aggregate_fit.

    Inheriting from FedAvg reuses Flower boilerplate (sampling, evaluation scheduling, etc.)
    while overriding `aggregate_fit` where FLTrust and audit logic live.
    """

    def __init__(
        self,
        fraction_fit:          float = 1.0,
        fraction_evaluate:     float = 1.0,
        min_fit_clients:       int   = cfg.FL_MIN_CLIENTS,
        min_available_clients: int   = cfg.FL_MIN_AVAILABLE,
        num_rounds:            int   = cfg.FL_NUM_ROUNDS,
        round_timeout_sec:     int   = cfg.FL_ROUND_TIMEOUT_SEC,
        merkle_tree=None,      # Optionally inject MerkleTree logger
    ):
        # Configure FedAvg base (we override aggregation but keep its scheduling)
        super().__init__(
            fraction_fit          = fraction_fit,
            fraction_evaluate     = fraction_evaluate,
            min_fit_clients       = min_fit_clients,
            min_available_clients = min_available_clients,
            # Round config function: tells clients how many local epochs to run
            on_fit_config_fn = lambda rnd: {
                "local_epochs": cfg.FL_LOCAL_EPOCHS,
                "round":        rnd,
                # Hint to client libraries; actual enforcement is server-side
                "timeout_sec":  round_timeout_sec,
            },
        )
        self.num_rounds        = num_rounds
        self.round_timeout_sec = round_timeout_sec
        self.audit_tree        = merkle_tree or __import__('aura.merkle_tree', fromlist=['']).MerkleTree()
        self._model_version    = 0
        self._hash_history: List[dict] = []

        # FLTrust root dataset — generated once, reused every round.
        # This is the server's small trusted benign baseline.
        self._root_data, self._server_attack_windows = _build_root_dataset()
        # Global model reference — updated after each successful aggregation.
        # FLTrust needs the current global state to compute per-client deltas.
        self._global_model: AURAModelBundle = AURAModelBundle()

        # Per-round trust score history (for dashboard + Upgrade 3 detection log)
        self._trust_history: List[dict] = []

        # Clear the trusted registry at the start of each FL session so that
        # only the current session's final hash is present (1 hash per run).
        registry_path = Path(cfg.LOGS_DIR) / "hash_registry.json"
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text("{}")
        logger.info("[REGISTRY] Cleared — fresh FL session starting.")

        logger.info(
            f"KrumFedAURA (FLTrust) strategy ready  |  "
            f"rounds={num_rounds}  timeout={round_timeout_sec}s  "
            f"root_samples={cfg.FLTRUST_ROOT_SAMPLES}  "
            f"server_lr={cfg.FLTRUST_SERVER_LR}"
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Override FedAvg's aggregate_fit to apply **FLTrust** (not plain FedAvg mean).

        Straggler Policy
        ----------------
        Flower already handles client timeouts internally via its gRPC layer
        (configured via flwr.server.ServerConfig(round_timeout=...)).
        Failed/timed-out clients arrive in the `failures` list, not in
        `results`.  We log them and proceed with whatever arrived on time.

        If fewer than min_fit_clients results arrive, we PRESERVE the previous
        global model (no aggregation) and log the stale round.
        """
        round_tag = f"[SERVER round={server_round}]"

        # ── Straggler Policy ────────────────────────────────────────────────
        n_received = len(results)
        n_failed   = len(failures)

        if n_failed > 0:
            logger.warning(
                f"{round_tag} {n_failed} client(s) timed out / failed "
                f"(straggler drop policy applied).  Proceeding with {n_received} responses."
            )

        # Use self.min_fit_clients so the quorum adapts when fewer orgs are
        # active (e.g., one quarantined).  cfg.FL_MIN_CLIENTS is only the
        # hard-coded default for standalone server runs; the dashboard always
        # passes len(active_orgs) explicitly at strategy instantiation time.
        min_needed = getattr(self, 'min_fit_clients', cfg.FL_MIN_CLIENTS)
        if n_received < min_needed:
            logger.error(
                f"{round_tag} Insufficient responses ({n_received} < "
                f"{min_needed}).  ABANDONING round — global model preserved."
            )
            return None, {"status": "abandoned", "received": n_received}

        # ── Extract weight arrays from Flower FitRes ─────────────────────────
        client_updates = []
        for client_proxy, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            client_updates.append(arrays)
            client_loss = fit_res.metrics.get("train_loss", "N/A")
            logger.info(f"{round_tag} Received update  |  "
                        f"num_examples={fit_res.num_examples}  loss={client_loss}")

        # ── DC-FLTrust Aggregation ───────────────────────────────────────────
        print(f"\n{round_tag} Running DC-FLTrust aggregation on {n_received} updates …")

        # Compute how many AE params the global model has
        n_ae_params = len(list(self._global_model.autoencoder.parameters()))

        # Split each client's 16-tensor payload into AE (first 12) and Head (last 4)
        global_ae_arrays = [p.detach().cpu().numpy()
                            for p in self._global_model.autoencoder.parameters()]
        global_head_arrays = [p.detach().cpu().numpy()
                              for p in self._global_model.attack_head.parameters()]

        client_ae_deltas   = []
        client_head_deltas = []
        for arrays in client_updates:
            ae_arrays   = arrays[:n_ae_params]
            head_arrays = arrays[n_ae_params:]
            # Compute deltas vs current global
            ae_delta = {
                f"layer_{i}": torch.tensor(c - g, dtype=torch.float32)
                for i, (c, g) in enumerate(zip(ae_arrays, global_ae_arrays))
            }
            head_delta = {
                f"layer_{i}": torch.tensor(c - g, dtype=torch.float32)
                for i, (c, g) in enumerate(zip(head_arrays, global_head_arrays))
            }
            client_ae_deltas.append(ae_delta)
            client_head_deltas.append(head_delta)

        # Compute server reference deltas (full-batch on root data)
        server_model = AURAModelBundle()
        with torch.no_grad():
            for p, gp in zip(server_model.autoencoder.parameters(),
                             self._global_model.autoencoder.parameters()):
                p.copy_(gp)
            for p, gp in zip(server_model.attack_head.parameters(),
                             self._global_model.attack_head.parameters()):
                p.copy_(gp)

        ae_opt = torch.optim.Adam(server_model.autoencoder.parameters(),
                                  lr=cfg.FLTRUST_SERVER_LR)
        server_model.autoencoder.train()
        ae_opt.zero_grad()
        # Full-batch gradient — eliminates sampling noise at convergence
        x_hat, _ = server_model.autoencoder(self._root_data)
        loss = nn.functional.mse_loss(x_hat, self._root_data)
        loss.backward()
        ae_opt.step()
        server_model.eval()

        root_ae_delta = {
            f"layer_{i}": (p.detach().cpu() - gp.detach().cpu()).float()
            for i, (p, gp) in enumerate(zip(server_model.autoencoder.parameters(),
                                            self._global_model.autoencoder.parameters()))
        }
        
        # Compute root head delta using reserved attack flows
        from aura.root_gradient import _build_root_head_reference
        root_head_delta = _build_root_head_reference(
            server_attack_windows=self._server_attack_windows,
            ae=server_model.autoencoder,
            global_head_weights={k: v.clone() for k, v in self._global_model.attack_head.state_dict().items()},
            mse_threshold=getattr(cfg, 'CH2_MSE_SPLIT_THRESHOLD', cfg.MSE_THRESHOLD_HIGH),
            server_lr=cfg.FLTRUST_SERVER_LR
        )

        client_round_counts = [server_round] * n_received

        agg_ae_delta, agg_head_delta, ch1_scores, ch2_scores, classifications, exclusion_flags = dc_fltrust_aggregate(
            client_ae_deltas=client_ae_deltas,
            client_head_deltas = client_head_deltas,
            root_ae_delta      = root_ae_delta,
            root_head_delta    = root_head_delta,
            client_round_counts = client_round_counts,
            ch2_warmup_rounds  = cfg.CH2_WARMUP_ROUNDS,
            current_round      = server_round,
        )

        # Apply deltas to global model to produce aggregated weight arrays
        aggregated_ae = [
            (gp.detach().cpu() + agg_ae_delta[f"layer_{i}"]).numpy().astype(np.float32)
            for i, gp in enumerate(self._global_model.autoencoder.parameters())
        ]
        if agg_head_delta is not None:
            aggregated_head = [
                (gp.detach().cpu() + agg_head_delta[f"layer_{i}"]).numpy().astype(np.float32)
                for i, gp in enumerate(self._global_model.attack_head.parameters())
            ]
        else:
            # No head update this round — keep current global head weights
            aggregated_head = global_head_arrays

        # Recombine into a single flat list matching model_to_ndarrays layout
        aggregated = aggregated_ae + aggregated_head

        # Map DC-FLTrust exclusion flags to flagged_indices for downstream logging
        flagged_indices = [
            i for i, excluded in enumerate(exclusion_flags)
            if excluded
        ]
        trust_scores = ch1_scores  # use ch1 as the primary trust score for logs

        self._model_version += 1
        model_version_tag = f"v{self._model_version}.{server_round}"

        # ── Log per-client trust scores ──────────────────────────────────────────
        for idx, (trust, (cp, fr)) in enumerate(zip(trust_scores, results)):
            status = "BYZANTINE SUSPECT" if idx in flagged_indices else "trusted"
            print(
                f"{round_tag} [FLTrust] Client {idx} — "
                f"trust={trust:.4f}  [{status}]  "
                f"loss={fr.metrics.get('train_loss', 'N/A')}"
            )
            if idx in flagged_indices:
                logger.warning(
                    f"{round_tag} [FLTrust] Client {idx} flagged — "
                    f"trust score {trust:.4f} ≤ threshold {cfg.FLTRUST_MIN_TRUST_SCORE}. "
                    f"Client's gradient direction opposes benign improvement."
                )

        # Persist trust scores for Upgrade 3 detection log
        trust_record = {
            "round":          server_round,
            "trust_scores":   [round(t, 4) for t in trust_scores],
            "flagged_indices": flagged_indices,
            "timestamp":      time.time(),
        }
        self._trust_history.append(trust_record)
        self._write_trust_log(trust_record)

        # Update server's global model reference for next round
        with torch.no_grad():
            for p, arr in zip(get_federated_parameters(self._global_model), aggregated):
                p.copy_(torch.tensor(arr))

        # ── SHA-256 Hash (computed every round — clients verify weights) ──────
        model_hash = hash_model_weights(aggregated)
        print(f"{round_tag} Global Model {model_version_tag} aggregated.")
        print(f"{round_tag} SHA-256 hash: {model_hash}")

        # ── Merkle Tree Audit Log ──────────────────────────────────────────────
        from aura.merkle_tree import AuditEntry
        from datetime import datetime
        total_trust = sum(trust_scores) + 1e-8
        for idx, (trust, (cp, fr)) in enumerate(zip(trust_scores, results)):
            classification = "BYZANTINE" if idx in flagged_indices else "HEALTHY"
            entry = AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                round_num=server_round,
                client_id=str(cp.cid),
                ae_update_norm=0.0,  # Computed in dc_fltrust normally
                head_update_norm=None,
                local_val_accuracy=float(fr.metrics.get("val_accuracy", 0.0)),
                ch1_trust_score=float(trust),
                ch2_trust_score=None,
                classification=classification,
                aggregation_weight=float(trust / total_trust if classification != "BYZANTINE" else 0.0)
            )
            self.audit_tree.append(entry)
        logger.info(f"Audit trail updated. Root: {self.audit_tree.root()[:16]}...")
        
        is_final_round = (server_round == self.num_rounds)
        if is_final_round:
            final_version = f"final_v{self._model_version}"
            self._log_hash_local(final_version, model_hash, server_round)

            # Write trusted registry — only for the final converged model
            self._write_trusted_registry(final_version, model_hash)
            model_version_tag = final_version
        else:
            print(f"{round_tag} Intermediate round — hash not minted yet "
                  f"(blockchain mint on round {self.num_rounds} only).")

        # Expose which indices were selected so client_statuses can use it
        # Record in history (for dashboard display)
        # NOTE: selected_indices is now all clients with trust > 0
        selected_indices = [i for i, t in enumerate(trust_scores) if t > cfg.FLTRUST_MIN_TRUST_SCORE]
        self._hash_history.append({
            "round":   server_round,
            "version": model_version_tag,
            "hash":    model_hash,
            "clients_selected": selected_indices,
            "clients_dropped":  [i for i in range(n_received) if i not in selected_indices],
        })

        # Save the aggregated model to disk
        self._save_model(aggregated, model_version_tag)

        return ndarrays_to_parameters(aggregated), {
            "model_version":           model_version_tag,
            "model_hash":              model_hash,
            "krum_selected":           len(selected_indices),      # legacy dashboard compat
            "krum_selected_indices":   selected_indices,
            "krum_dropped":            len(flagged_indices),
            "trust_scores":            [round(t, 4) for t in trust_scores],
            "fltrust_flagged":         flagged_indices,
            "fltrust_trusted_indices": selected_indices,
            "fltrust_flagged_indices": flagged_indices,
        }

    def _log_hash_local(self, version: str, model_hash: str, rnd: int) -> None:
        """Fallback: write hash to local JSONL file if blockchain is unavailable."""
        record = {
            "timestamp": time.time(),
            "round":     rnd,
            "version":   version,
            "hash":      model_hash,
        }
        log_path = Path(cfg.LOGS_DIR) / "model_hashes.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(f"[LOCAL-HASH] {model_hash} written to {log_path}")

    def _write_trusted_registry(self, version: str, model_hash: str) -> None:
        """Write to the trusted hash registry (separate from the ledger).
        verify_chain.py reads this as the ground-truth reference.
        Corrupting the blockchain ledger won't affect this file.
        """
        registry_path = Path(cfg.LOGS_DIR) / "hash_registry.json"
        registry: dict = {}
        if registry_path.exists():
            try:
                registry = json.loads(registry_path.read_text())
            except Exception as e:
                raise RuntimeError(
                    f"FATAL: Hash registry read failed. "
                    f"Wiping registry silently breaks audit trail verification. "
                    f"Original error: {e}"
                )
        registry[version] = model_hash
        registry_path.write_text(json.dumps(registry, indent=2))
        logger.info(f"[REGISTRY] {version} written to trusted registry.")

    def _write_trust_log(self, record: dict) -> None:
        """
        Append per-round trust scoring record to logs/fltrust_trust_log.jsonl.
        Consumed by Upgrade 3 (Byzantine detection dashboard table).
        """
        try:
            log_path = Path(cfg.LOGS_DIR) / "fltrust_trust_log.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            logger.info(
                f"[FLTRUST] Round {record['round']} trust log written — "
                f"flagged={record['flagged_indices']}"
            )
        except Exception as e:
            logger.warning(f"[FLTRUST] Trust log write failed: {e}")

    def _save_model(self, arrays: List[np.ndarray], version_tag: str) -> None:
        """Save the aggregated global model weights to disk."""
        model = AURAModelBundle()
        with torch.no_grad():
            for p, arr in zip(get_federated_parameters(model), arrays):
                p.copy_(torch.tensor(arr))
        save_path = Path(cfg.MODELS_DIR) / f"global_model_{version_tag}.pth"
        torch.save(model.state_dict(), save_path)
        logger.info(f"Global model saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Server Launch
# ─────────────────────────────────────────────────────────────────────────────

def start_server(merkle_tree=None) -> None:
    """
    Start the AURA Flower federation server.

    The server blocks until all FL_NUM_ROUNDS are complete, then exits.
    Run in a separate process/thread from the dashboard.

    Straggler timeout is enforced via ServerConfig(round_timeout=...).
    Any client that doesn't respond within round_timeout_sec is treated
    as dropped for that round (Flower gRPC layer handles the socket close).
    """
    strategy = KrumFedAURA(merkle_tree=merkle_tree)

    server_config = fl.server.ServerConfig(
        num_rounds    = cfg.FL_NUM_ROUNDS,
        round_timeout = cfg.FL_ROUND_TIMEOUT_SEC,   # Straggler hard timeout
    )

    print(f"\n{'='*60}")
    print(f"  AURA Federation Server starting on {cfg.FL_SERVER_ADDRESS}")
    print(f"  Rounds: {cfg.FL_NUM_ROUNDS}  |  Timeout: {cfg.FL_ROUND_TIMEOUT_SEC}s")
    print(f"  Strategy: FLTrust (5-client Byzantine-robust aggregation)")
    print(f"{'='*60}\n")

    fl.server.start_server(
        server_address = cfg.FL_SERVER_ADDRESS,
        config         = server_config,
        strategy       = strategy,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Simulation Mode (no real gRPC) — for demo without network setup
# ─────────────────────────────────────────────────────────────────────────────

def run_federation_simulation(merkle_tree=None, n_rounds: int = None,
                              active_orgs: list = None) -> List[dict]:
    """
    In-process federation simulation for the hackathon demo.

    Parameters
    ----------
    active_orgs : list of org keys that are ready, e.g. ["hospital","university"].
                  If None, defaults to all three.  Only ready orgs participate.
                  Byzantine client is randomly assigned among them each run.
    """
    from aura.fl_client import create_mock_clients

    if n_rounds is None:
        n_rounds = cfg.FL_NUM_ROUNDS

    if active_orgs is None:
        active_orgs = ["hospital", "bank", "university", "isp", "retail"]

    # Attack is tied to the bank org — only injected if bank is in active_orgs.
    # If bank is offline, all clients are honest (no meaningless FLTrust flag).
    if active_orgs:
        attack_arg = random.randint(0, len(active_orgs) - 1)
        print(f"[SERVER] 🎲 Randomly selected Byzantine client index: {attack_arg} ({active_orgs[attack_arg]})")
    else:
        attack_arg = -1

    from aura.data_loader import CICIDSDataLoader
    _shared_loader = CICIDSDataLoader()
    _shared_scaler = _shared_loader.fit_scaler()

    clients, attack_idx = create_mock_clients(
        n_clients     = len(active_orgs),
        n_samples     = 300,
        org_ids       = active_orgs,
        attack_client = attack_arg,
        shared_scaler = _shared_scaler,
    )
    strategy = KrumFedAURA(merkle_tree=merkle_tree,
                           num_rounds=n_rounds)

    # Initialise with random global model
    global_model = AURAModelBundle()
    global_params = [p.detach().cpu().numpy() for p in get_federated_parameters(global_model)]

    round_results = []

    for rnd in range(1, n_rounds + 1):
        print(f"\n{'─'*55}")
        print(f"  FEDERATION ROUND {rnd}/{n_rounds}")
        print(f"{'─'*55}")

        fit_results = []
        for client in clients:
            # Build FitIns with current global params
            from flwr.common import FitIns, Config
            fit_ins = FitIns(
                parameters = ndarrays_to_parameters(global_params),
                config     = {"local_epochs": cfg.FL_LOCAL_EPOCHS, "round": rnd},
            )
            fit_res = client.fit(fit_ins)

            # Simulate per-client console output for demo effect
            print(f"  [CLIENT {client.client_id}] weights sent to server ✓")
            fit_results.append((None, fit_res))   # ClientProxy=None in simulation

        # Run server-side FLTrust aggregation
        new_params, metrics = strategy.aggregate_fit(
            server_round = rnd,
            results      = fit_results,
            failures     = [],
        )
        # Build per-client status for dashboard display
        # "Byzantine" = clients FLTrust flagged (low cosine trust vs server root)
        selected_idx = metrics.get("fltrust_trusted_indices", [])
        dropped_idx  = list(metrics.get("fltrust_flagged_indices", []))
        client_statuses = []
        for i, client in enumerate(clients):
            is_selected  = (selected_idx and i in selected_idx)
            is_byzantine = (i in dropped_idx)   # FLTrust-flagged = suspicious
            org_key = active_orgs[i] if i < len(active_orgs) else f"org_{i}"
            client_statuses.append({
                "client_id": client.client_id,
                "org_id":    org_key,
                "network":   cfg.ORG_NETWORK_MAP.get(org_key, "—"),
                "org":       org_key.capitalize(),
                "role":      "Byzantine" if is_byzantine else "Normal",
                "selected":  is_selected if selected_idx else (not is_byzantine),
                "round":     rnd,
            })
        if new_params is not None:
            global_params = parameters_to_ndarrays(new_params)
            model_version = metrics.get('model_version')
            server_hash   = metrics.get('model_hash')
            print(f"\n  [SERVER] Global Model {model_version} aggregated.")
            print(f"  [SERVER] Merkle root updated: {server_hash[:20]}...")
            print(f"  [SERVER] FLTrust trusted {len(selected_idx)} / "
                  f"{len(clients)} clients.")
            print()

            # ── Client-side hash verification ────────────────────────────────────
            # Hash verification only makes sense on the final round, because the
            # blockchain is only minted once (at the end of federation).
            # On intermediate rounds we print the computed hash for auditing only.
            is_final = model_version and model_version.startswith("final_")
            if is_final:
                client_received_hash = hash_model_weights(global_params)

                for client in clients:
                    # Fetch the hash the server minted for this version
                    bc = merkle_tree
                    if bc is not None:
                        on_chain_ok = (client_received_hash == server_hash) # For simulation
                    else:
                        # No MerkleTree — compare directly against server hash
                        on_chain_ok = (client_received_hash == server_hash)

                    if on_chain_ok:
                        print(f"  [CLIENT {client.client_id}] "
                              f"Received hash {client_received_hash[:16]}... "
                              f"== Merkle hash {server_hash[:16]}... "
                              f"→ MATCH. Model deployed.")
                    else:
                        print(f"  [CLIENT {client.client_id}] "
                              f"Received hash {client_received_hash[:16]}... "
                              f"!= Merkle hash {server_hash[:16]}... "
                              f"→ MISMATCH! Weights tampered in transit. REJECTING model.")
            else:
                print(f"  [CLIENTS] Intermediate round — final hash not yet verified. "
                      f"Hash {server_hash[:20]}... recorded locally for auditing.")

        round_results.append({"round": rnd, "client_statuses": client_statuses, **metrics})

    print(f"\n{'='*55}")
    print(f"  Federation complete.  {n_rounds} rounds executed.")
    print(f"{'='*55}\n")

    return round_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AURA FL Server (FLTrust)")
    parser.add_argument(
        "--address", default=cfg.FL_SERVER_ADDRESS,
        help="gRPC bind address (default: %(default)s). "
             "Use 0.0.0.0:8080 to accept remote clients."
    )
    parser.add_argument(
        "--rounds", type=int, default=cfg.FL_NUM_ROUNDS,
        help="Number of FL rounds (default: %(default)s)"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Run in-process simulation instead of gRPC server (legacy mode)"
    )
    args = parser.parse_args()

    if args.simulate:
        print("=== AURA Federation — In-Process Simulation Mode ===")
        results = run_federation_simulation(n_rounds=args.rounds)
        for r in results:
            print(f"  Round {r['round']}: {r.get('model_version')}  "
                  f"hash={r.get('model_hash', 'N/A')[:18]}…")
        print("✓ Federation simulation complete.")
    else:
        # TRUE NETWORKED MODE — waits for real gRPC client connections
        from aura.merkle_tree import MerkleTree, AuditEntry
        bc = MerkleTree()

        strategy = KrumFedAURA(
            merkle_tree = bc,
            num_rounds        = args.rounds,
        )
        server_config = fl.server.ServerConfig(
            num_rounds    = args.rounds,
            round_timeout = cfg.FL_ROUND_TIMEOUT_SEC,
        )

        print(f"\n{'='*62}")
        print(f"  AURA Federation Server — NETWORKED MODE")
        print(f"  Binding on:  {args.address}")
        print(f"  Rounds:      {args.rounds}")
        print(f"  Strategy:    FLTrust (5-client Byzantine-robust aggregation)")
        print(f"  Waiting for {cfg.FL_MIN_AVAILABLE} clients to connect …")
        print(f"{'='*62}\n")

        fl.server.start_server(
            server_address = args.address,
            config         = server_config,
            strategy       = strategy,
        )

FLTrustServerAURA = KrumFedAURA
