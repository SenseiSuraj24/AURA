#!/usr/bin/env python3
"""
scripts/benchmark_byzantine.py
==============================
Implements the 5.1.5 FLTrust Byzantine Benchmark (Hypothesis mapping: H2).

Runs a comparative evaluation of:
  - FedAvg (no defense)
  - Krum (distance-based exclusion)
  - FLTrust (cosine similarity against server root dataset)

Under various Byzantine attack ratios (10%, 20%, 30%, 40%).
Also includes the rare-client contribution preservation experiment.

NOTE: Runs in-process (no Ray/Flower simulation daemon required).

NOTE: Runs in-process (no Ray/Flower simulation daemon required).
"""

import sys
import logging
import hashlib
import hashlib
from pathlib import Path
from typing import List, Tuple
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
import torch
import torch.nn as nn
import numpy as np
import io

from aura.fl_server import fltrust_aggregate, krum_select, krum_aggregate, _build_root_dataset, get_federated_parameters
from aura import fl_round
from aura.root_gradient import attack_flows_from_windows
from aura.fl_client import AURAFlowerClient
from aura.data_loader import CICIDSDataLoader, load_client_partition
from aura.attack_injector import _benign_profile
from aura.models import AURAModelBundle, FlowAutoencoder, AttackHead
import torch.nn.functional as F

from config import preflight_dc_fltrust_check
preflight_dc_fltrust_check()  # Hard stops if profiles are missing or stale

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("byz_bench")

# Force stdout to UTF-8 so Unicode characters render correctly on Windows
if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# -----------------------------------------------------------------------------
# Initialize global scaler once for the entire benchmark run
# -----------------------------------------------------------------------------
_loader = None
_shared_scaler = None
server_attack_windows = None
_all_train_flows = None
_client_pool = None
_root_size = cfg.FLTRUST_ROOT_SAMPLES

def initialize_benchmark_data():
    global _loader, _shared_scaler, server_attack_windows, _all_train_flows, _client_pool
    _loader = CICIDSDataLoader()
    try:
        from aura import dataset as ds
        from aura.split_manager import get_canonical_split

        ds.check_manifest()
        _shared_scaler = ds.load_preprocessor()

        # Server attack reserve (root blocks); the server's benign root data is
        # built separately by _build_root_dataset() from the same root blocks.
        _, _, _, server_attack_windows = get_canonical_split(_loader.stream_graphs(_shared_scaler))

        # Client data: de-duplicated TRAINING flows (natural benign/attack mix),
        # disjoint from test, calibration and server-root flows by construction.
        split = ds.compute_split()
        _all_train_flows = ds.flows_tensor(split.flows["train"], _shared_scaler)
        _all_train_flows = _all_train_flows[torch.randperm(len(_all_train_flows))]

        # --- Privacy-preserving partition boundary ---
        # Root dataset takes the FIRST cfg.FLTRUST_ROOT_SAMPLES rows.
        # Each client gets a non-overlapping slice from the remainder,
        # so no client flow ever appears in the server's root dataset.
        _client_pool = _all_train_flows[_root_size:]  # everything after root slice

    except Exception as e:
        logger.error(f"FATAL: Could not fit or load scaler on CSV dataset: {e}.")
        raise RuntimeError(f"Scaler initialization failed: {e}")

def _build_benchmark_root_dataset(n_samples=None):
    """Return the fixed root dataset (first _root_size flows)."""
    n = n_samples or _root_size
    return _all_train_flows[:n]

def _get_client_slice(client_idx: int, num_clients: int) -> torch.Tensor:
    """Return a non-overlapping slice of the client pool for client_idx."""
    per_client = len(_client_pool) // num_clients
    start = client_idx * per_client
    end   = start + per_client
    return _client_pool[start:end]



def generate_client_data(
    client_idx: int,
    is_byzantine: bool,
    is_rare: bool,
    num_clients: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate local training data for a client from the canonical train split.

    Each client receives a non-overlapping slice of _client_pool — the portion
    of the train split that comes after the root dataset reserved slice. This
    guarantees zero overlap between root dataset and client data.

    - Benign/rare clients: raw flows from their slice (~100k rows each)
    - Byzantine client:    same slice, but 80% of rows replaced with DDoS profile
    """
    feature_dim = cfg.FEATURE_DIM

    # Real data: non-overlapping slice from canonical train split
    train_data = _get_client_slice(client_idx, num_clients)
    val_size   = max(1, len(train_data) // 5)
    val_data   = train_data[:val_size]
    train_data = train_data[val_size:]  # train is the remaining 80%

    logger.info(
        f"[generate_client_data] Client {client_idx}: "
        f"train={len(train_data)} flows, val={len(val_data)} flows"
    )

    if is_rare:
        # Legitimate but distribution-shifted client (e.g. hospital with rare traffic).
        # +0.15 global shift simulates higher baseline volume — still benign direction.
        train_data = train_data + 0.15

    if is_byzantine:
        # Poison 80% of local batch using DDoS feature ranges from NF-UNSW-NB15-v3
        ddos_profile = cfg.ATTACK_CORRUPTION_PROFILES.get("ddos", {})
        feat_map     = cfg.FEATURE_INDEX_MAP
        n_attack     = int(len(train_data) * 0.8)
        attack_rows  = train_data[:n_attack].clone()
        for feat_name, (lo, hi) in ddos_profile.items():
            if feat_name in feat_map:
                col_idx = feat_map[feat_name]
                attack_rows[:, col_idx] = torch.rand(n_attack) * (hi - lo) + lo
        train_data[:n_attack] = attack_rows

    return train_data, val_data


def _run_local_training(
    client:        AURAFlowerClient,
    global_arrays: List[np.ndarray],
) -> Tuple[List[np.ndarray], float]:
    """
    One FL round on a single client (in-process, no gRPC needed):
      1. Load global weights into client's local model.
      2. Run FL_LOCAL_EPOCHS of unsupervised AE training.
      3. Return updated weight arrays + training loss.
    """
    with torch.no_grad():
        for p, arr in zip(get_federated_parameters(client.model), global_arrays):
            p.copy_(torch.tensor(arr, dtype=torch.float32))

    num_examples, train_loss = client._local_train()
    updated_arrays = [p.detach().cpu().numpy() for p in get_federated_parameters(client.model)]
    return updated_arrays, train_loss


def _run_local_training_dual(
    ae: FlowAutoencoder,
    attack_head: AttackHead,
    all_flows: torch.Tensor,
    ae_optimizer: torch.optim.Optimizer,
    head_optimizer: torch.optim.Optimizer,
    global_ae_weights: dict,
    global_head_weights: dict,
    mse_threshold_high: float,
    head_epochs: int = 3,
    batch_size: int = 256
) -> tuple:
    from aura.local_training import run_two_pass_local_training
    
    z_buffer, n_benign, n_high_mse, _, ch1_export_state = run_two_pass_local_training(
        ae, attack_head, all_flows,
        ae_optimizer, head_optimizer,
        mse_threshold=mse_threshold_high,
        head_epochs=head_epochs,
        batch_size=batch_size
    )
    
    assert n_benign > 0 or n_high_mse > 0, "FATAL: No flows processed in two-pass training"
    logger.info(f"Two-pass: benign={n_benign}, high_mse={n_high_mse}, z_buffer={sum(len(z) for z in z_buffer)}")
    
    # Export CH1 representation based on optimization horizon policy
    ae_delta = {k: ch1_export_state[k] - global_ae_weights[k]
                for k in ch1_export_state}
    
    if n_high_mse > 0:
        head_delta = {k: attack_head.state_dict()[k].clone() - global_head_weights[k]
                      for k in attack_head.state_dict()}
    else:
        head_delta = None
        
    return ae_delta, head_delta, z_buffer, n_benign, n_high_mse




def run_experiment(
    strategy_name:   str,
    num_clients:     int,
    byzantine_ratio: float,
    rare_client:     bool = False,
    mode:            str = "dc_fltrust",
    num_rounds:      int = 10,
    attack_mode:     str = "none",
    seed:            int = None,
    export_tensors:  bool = False
):
    """
    Run the AURA federation loop locally (no gRPC/Flower overhead).

    Parameters
    ----------
    strategy_name   : "FedAvg", "Krum", or "FLTrust"
    num_clients     : Total clients in the federation (10)
    byzantine_ratio : Fraction of clients that are adversarial (0.1 to 0.4)
    rare_client     : If True, last client gets shifted-but-benign distribution
    seed            : If set, fixes torch/numpy RNG so runs are comparable
                       across strategies/ratios (needed for divergence metric).

    Returns
    -------
    dict with keys: strategy, byzantine_ratio, num_byzantine, roles,
    flagged_indices (clients this strategy excluded/flagged, empty for
    FedAvg since it has no defense), tp/fp/fn/tn and balanced_accuracy for
    Byzantine-client detection, final_arrays (the resulting global model
    weights, for computing divergence against a clean baseline), model_hash.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    logger.info("\n" + "=" * 60)
    logger.info(
        f"Running {strategy_name} | Byzantine Ratio: {byzantine_ratio*100:.0f}% "
        f"| Rare Client: {rare_client}"
    )
    logger.info("=" * 60)

    num_byzantine = int(num_clients * byzantine_ratio)
    roles         = ["benign"] * num_clients
    for i in range(num_byzantine):
        roles[i] = "byzantine"
    if rare_client and "benign" in roles:
        roles[-1] = "rare"


    logger.info(f"Client Roles: {roles}")

    # Build clients — each gets a non-overlapping slice of the train split
    clients: List[AURAFlowerClient] = []
    for idx in range(num_clients):
        role = roles[idx]
        train_data, val_data = generate_client_data(
            idx,
            is_byzantine=(role == "byzantine"),
            is_rare=(role == "rare"),
            num_clients=num_clients,
        )
        clients.append(AURAFlowerClient(f"client_{idx}", train_data, val_data))

    # Shared global model (G0)
    global_model = fl_round.load_global_model()

    # The STGNN is organisation-local: only AE and AttackHead are federated.
    global_arrays = [p.detach().cpu().numpy() for p in get_federated_parameters(global_model)]

    # FLTrust server root dataset (benign reference -- built once per experiment)
    root_data, _ = _build_root_dataset(_shared_scaler, n_samples=cfg.FLTRUST_ROOT_SAMPLES)
    root_attack_flows = attack_flows_from_windows(server_attack_windows)

    # Federated rounds
    for rnd in range(1, num_rounds + 1):
        print("\n" + "-" * 60)
        print(f"  [{strategy_name} | {byzantine_ratio*100:.0f}% Byzantine] Round {rnd}/{num_rounds}")
        print("-" * 60)

        # FedAvg/Krum clients train here; the FLTrust family trains in its own branch.
        client_updates: List[List[np.ndarray]] = []
        if strategy_name in ("FedAvg", "Krum"):
            for idx, client in enumerate(clients):
                updated_arrays, train_loss = _run_local_training(client, global_arrays)
                print(f"  Client {idx:2d} [{roles[idx]:10s}]  train_loss={train_loss:.4f}")
                client_updates.append(updated_arrays)

        # Aggregation strategy
        if strategy_name == "FedAvg":
            # Plain arithmetic mean -- no defense
            new_arrays = [
                np.mean([upd[i] for upd in client_updates], axis=0).astype(np.float32)
                for i in range(len(client_updates[0]))
            ]
            print(f"\n  [FedAvg] Plain mean aggregation applied.")
            print(
                f"  [FedAvg] All {num_clients} clients contributed equally -- "
                f"including {num_byzantine} Byzantine (NO FILTER)."
            )
            flagged_indices = []  # FedAvg has no defense -- never flags anyone

        elif strategy_name == "Krum":
            # Distance-based Krum selection
            num_select       = max(1, num_clients - num_byzantine - 2)
            selected_indices = krum_select(client_updates, num_to_select=num_select)
            selected_updates = [client_updates[i] for i in selected_indices]
            new_arrays       = krum_aggregate(selected_updates)
            dropped          = [i for i in range(num_clients) if i not in selected_indices]
            flagged_indices  = dropped  # unify naming with FLTrust's flagged_indices
            print(f"\n  [Krum] Selected: {selected_indices} | Dropped: {dropped}")
            for i in dropped:
                print(f"  [Krum] WARNING: Client {i:2d} [{roles[i]}] DROPPED (high Euclidean distance from cluster)")
            if rare_client:
                rare_idx = num_clients - 1
                if rare_idx in dropped:
                    print(
                        f"\n  [Krum] *** FALSE POSITIVE: Rare client {rare_idx} DROPPED "
                        f"(legitimate but geometrically distant) ***"
                    )
                else:
                    print(f"\n  [Krum] OK: Rare client {rare_idx} correctly kept.")

        else:
            # FLTrust cosine-trust aggregation
            with torch.no_grad():
                for p, arr in zip(get_federated_parameters(global_model), global_arrays):
                    p.copy_(torch.tensor(arr, dtype=torch.float32))

            from aura.fl_server import ae_only_fltrust_aggregate, joint_dual_fltrust_aggregate

            g_ae_w = fl_round.snapshot(global_model.autoencoder)
            g_head_w = fl_round.snapshot(global_model.attack_head)

            r_ae_delta, r_head_delta = fl_round.root_reference(g_ae_w, g_head_w, root_data, root_attack_flows)

            c_ae_deltas = []
            c_head_deltas = []

            from scripts.experiments.byzantine_deception_experiment import _run_latent_inversion_byzantine, _run_true_labelflip_byzantine

            for idx, client in enumerate(clients):
                is_byzantine = (roles[idx] == "byzantine")

                if is_byzantine and attack_mode in ('latent_inversion', 'true_labelflip'):
                    run_attack = (_run_latent_inversion_byzantine if attack_mode == 'latent_inversion'
                                  else _run_true_labelflip_byzantine)
                    client.model.autoencoder.load_state_dict(g_ae_w)
                    client.model.attack_head.load_state_dict(g_head_w)
                    ae_opt = torch.optim.Adam(client.model.autoencoder.parameters(), lr=cfg.AE_LEARNING_RATE)
                    head_opt = torch.optim.Adam(client.model.attack_head.parameters(), lr=cfg.AE_LEARNING_RATE)
                    ae_delta, head_delta, _, n_benign, n_attack = run_attack(
                        client.model.autoencoder, client.model.attack_head, client.train_data, ae_opt, head_opt,
                        g_ae_w, g_head_w, mse_threshold_high=cfg.CH2_MSE_SPLIT_THRESHOLD, head_epochs=cfg.FL_HEAD_EPOCHS
                    )
                else:
                    u = fl_round.client_update(client.model.autoencoder, client.model.attack_head,
                                               g_ae_w, g_head_w, client.train_data)
                    ae_delta, head_delta, n_benign, n_attack = u.ae_delta, u.head_delta, u.n_benign, u.n_high_mse

                c_ae_deltas.append(ae_delta)
                c_head_deltas.append(head_delta)

                logger.info(
                    f"Client {idx} round {rnd}: "
                    f"benign_flows={n_benign}, high_mse_flows={n_attack}"
                )

            client_round_counts = [rnd] * num_clients

            import torch.nn.functional as F_func
            def _cos(d1, d2):
                t1 = torch.cat([v.flatten() for v in d1.values()])
                t2 = torch.cat([v.flatten() for v in d2.values()])
                if t1.norm() == 0 or t2.norm() == 0: return 0.0
                return F_func.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()

            benchmark_ch1 = []
            for idx, ae_d in enumerate(c_ae_deltas):
                raw_cos = _cos(r_ae_delta, ae_d)
                relu_cos = max(0.0, raw_cos)
                benchmark_ch1.append(relu_cos)

            if export_tensors:
                import pickle
                export_path = cfg.MODELS_DIR / f"exported_tensors_seed_{seed}_round_{rnd}.pkl"
                with open(export_path, 'wb') as f:
                    pickle.dump({
                        'root_ae_delta': r_ae_delta,
                        'root_head_delta': r_head_delta,
                        'client_ae_deltas': c_ae_deltas,
                        'client_head_deltas': c_head_deltas,
                        'global_ae_weights': g_ae_w,
                        'global_head_weights': g_head_w,
                        'roles': roles,
                        'metadata': {
                            'seed': seed,
                            'round': rnd,
                            'attack_mode': attack_mode,
                            'benchmark_ch1': benchmark_ch1
                        }
                    }, f)
                logger.info(f"[EXPORT] Saved tensors and benchmark metrics to {export_path}")

            
            if mode == 'ae_only':
                new_ae, trust_scores = ae_only_fltrust_aggregate(
                    c_ae_deltas, r_ae_delta
                )
                new_head = None
                ch1_scores = trust_scores
                ch2_scores = [0.0] * num_clients
                classifications = ['HEALTHY' if t > 0.0 else 'BYZANTINE' for t in trust_scores]
            elif mode == 'joint_dual':
                new_ae, new_head, combined_scores, ch1_scores, ch2_scores = joint_dual_fltrust_aggregate(
                    c_ae_deltas, c_head_deltas, r_ae_delta, r_head_delta, client_round_counts,
                    ch2_warmup_rounds=cfg.CH2_WARMUP_ROUNDS, ch1_weight=0.7
                )
                trust_scores = combined_scores
                classifications = ['HEALTHY' if t > 0.0 else 'BYZANTINE' for t in trust_scores]
            else:
                # mode == 'dc_fltrust'

                new_ae, new_head, ch1_scores, ch2_scores, classifications, exclusion_flags = fl_round.aggregate(
                    c_ae_deltas, c_head_deltas, r_ae_delta, r_head_delta, rnd
                )
                trust_scores = ch1_scores

            # Reconstruct model from aggregated deltas
            with torch.no_grad():
                if new_ae is not None:
                    for k, p in global_model.autoencoder.named_parameters():
                        p.copy_(g_ae_w[k] + new_ae[k])
                if new_head is not None:
                    for k, p in global_model.attack_head.named_parameters():
                        p.copy_(g_head_w[k] + new_head[k])
            
            new_arrays = [p.detach().cpu().numpy() for p in get_federated_parameters(global_model)]
            
            if mode == 'dc_fltrust':
                flagged_indices = [i for i, excl in enumerate(exclusion_flags) if excl]
            else:
                flagged_indices = [i for i, c in enumerate(classifications) if 'BYZANTINE' in c]
            
            for idx in range(num_clients):
                print(f"  [{mode}] Client {idx:2d} [{roles[idx]:10s}] ch1={ch1_scores[idx]:.4f} ch2={ch2_scores[idx] if ch2_scores[idx] is not None else 0.0:.4f} -> {classifications[idx]}")
                
                
            print(f"\n  [{mode}] Per-Client Trust Scores:")
            for idx, trust in enumerate(trust_scores):
                flag = "[BYZANTINE SUSPECT]" if idx in flagged_indices else "[trusted         ]"
                print(
                    f"  [{mode}] Client {idx:2d} [{roles[idx]:10s}]  "
                    f"trust={trust:.4f}  {flag}"
                )
            print(
                f"\n  [{mode}] Flagged Byzantine: {flagged_indices}  "
                f"(expected adversarial clients: {list(range(num_byzantine))})"
            )
            if rare_client:
                rare_idx = num_clients - 1
                if rare_idx in flagged_indices:
                    print(
                        f"  [FLTrust] *** FALSE POSITIVE: Rare client {rare_idx} incorrectly flagged ***"
                    )
                else:
                    print(
                        f"  [FLTrust] *** TRUE NEGATIVE:  Rare client {rare_idx} correctly PRESERVED "
                        f"(direction-aligned despite distribution shift) ***"
                    )

        # Update shared global model
        global_arrays = new_arrays
        with torch.no_grad():
            for p, arr in zip(get_federated_parameters(global_model), global_arrays):
                p.copy_(torch.tensor(arr, dtype=torch.float32))

    # Final SHA-256 hash (tamper-evident audit trail)
    h = hashlib.sha256()
    for arr in global_arrays:
        h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    model_hash = "0x" + h.hexdigest()
    print(f"\n  [{strategy_name} | {byzantine_ratio*100:.0f}% Byzantine] Final Model SHA-256: {model_hash}")
    
    # Save the post-FL model to disk so we can ablate it
    post_fl_path = cfg.MODELS_DIR / f"aura_bundle_post_fl_{mode}.pth"
    torch.save(global_model.state_dict(), post_fl_path)
    print(f"  [FLTrust] Saved post-FL model to {post_fl_path.name}")
    logger.info(f"Finished {strategy_name} | {byzantine_ratio*100:.0f}% Byzantine simulation.")


def main():
    import argparse
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        choices=['ae_only', 'joint_dual', 'dc_fltrust'],
        default='dc_fltrust',
        help=(
            'ae_only: AE-only FLTrust baseline (Mode A) — '
            'no AttackHead federation. '
            'joint_dual: combined ch1+ch2 trust score (Mode B) — '
            'both channels evaluated but not disambiguated. '
            'dc_fltrust: full DC-FLTrust with disambiguation (Mode C).'
        )
    )
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--attack-mode',
        choices=['none', 'latent_inversion', 'true_labelflip'],
        default='none'
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--export-tensors', action='store_true', help='Export deltas to pkl')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    initialize_benchmark_data()

    print("\n" + "=" * 70)
    print("  AURA Byzantine Benchmark  --  DC-FLTrust Deception Experiment")
    print("=" * 70)
    
    num_clients = 5
    ratio = 0.2  # 1 byzantine client
    run_experiment("FLTrust", num_clients, byzantine_ratio=ratio, mode=args.mode, num_rounds=args.rounds, attack_mode=args.attack_mode, seed=args.seed, export_tensors=args.export_tensors)

    print("\n" + "=" * 70)
    print("  Byzantine Benchmark Complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
