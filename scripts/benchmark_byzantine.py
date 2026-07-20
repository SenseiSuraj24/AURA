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

from aura.fl_server import fltrust_aggregate, krum_select, krum_aggregate, _build_root_dataset
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
        import joblib
        import os
        scaler_path = os.path.join(cfg.MODELS_DIR, "scaler.joblib")
        if os.path.exists(scaler_path):
            _shared_scaler = joblib.load(scaler_path)
            logger.info("Loaded saved global dataset scaler.")
        else:
            _shared_scaler = _loader.fit_scaler()
            logger.info("Global dataset scaler initialized successfully (fitted on fly).")
        
        from aura.split_manager import get_canonical_split

        # Load all windows
        all_windows = list(_loader.stream_graphs(_shared_scaler))

        # Get canonical train/test split
        calib_windows, train_windows, test_windows, server_attack_windows = get_canonical_split(
            all_windows, test_fraction=0.20
        )
        
        # CRITICAL: calibration_windows are a prefix of train_windows.
        # To guarantee zero data leakage between the server's root dataset
        # (built from calib_windows) and client training data, we must remove them.
        train_windows = train_windows[len(calib_windows):]

        # Extract ALL flows from train windows, randomised once for reproducibility
        _all_train_flows = torch.cat([
            graph['edge_attr'] for graph, labels in train_windows
        ])
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
    print(f"[CLIENT DATA] Client {client_idx} using shared scaler instance: {id(_shared_scaler)}")

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
        for p, arr in zip(client.model.parameters(), global_arrays):
            p.copy_(torch.tensor(arr, dtype=torch.float32))

    num_examples, train_loss = client._local_train()
    updated_arrays = [p.detach().cpu().numpy() for p in client.model.parameters()]
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
        for p, arr in zip(client.model.parameters(), global_arrays):
            p.copy_(torch.tensor(arr, dtype=torch.float32))

    num_examples, train_loss = client._local_train()
    updated_arrays = [p.detach().cpu().numpy() for p in client.model.parameters()]
    return updated_arrays, train_loss


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

    # Shared global model
    import os
    global_model  = AURAModelBundle()
    
    # Load full pretrained bundle (so STGNN is also preserved, not random!)
    bundle_path = cfg.MODELS_DIR / 'aura_bundle.pth'
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Pretrained bundle not found at {bundle_path}. "
            "Run train.py before benchmark_byzantine.py."
        )
    state = torch.load(bundle_path, map_location='cpu', weights_only=True)
    incompatible_keys = global_model.load_state_dict(state, strict=False)
    if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
        logger.warning(f"[INIT] Checkpoint mismatches found in {bundle_path}:")
        if incompatible_keys.missing_keys:
            logger.warning(f"       Missing keys: {incompatible_keys.missing_keys}")
        if incompatible_keys.unexpected_keys:
            logger.warning(f"       Unexpected keys: {incompatible_keys.unexpected_keys}")
        
        # Fail loudly if ANY autoencoder keys are missing, as that breaks Channel 1
        ae_missing = [k for k in incompatible_keys.missing_keys if k.startswith('autoencoder')]
        if ae_missing:
            raise RuntimeError(f"Architecture mismatch! Missing Autoencoder keys: {ae_missing}")
            
    global_model.autoencoder.eval()
    logger.info(f"[INIT] Loaded pretrained bundle from {bundle_path}")

    global_arrays = [p.detach().cpu().numpy() for p in global_model.parameters()]

    # FLTrust server root dataset (benign reference -- built once per experiment)
    root_data, _ = _build_root_dataset(_shared_scaler, n_samples=2000)

    # Federated rounds
    from aura.attack_reference import AttackReferenceBuffer
    if mode in ["joint_dual", "dc_fltrust"]:
        logger.debug(f"[{mode}] CH2 MSE split threshold: {cfg.CH2_MSE_SPLIT_THRESHOLD:.6f} "
              f"(P75 of benign distribution — flows above this route to AttackHead)")
              
        attack_ref_buffer = AttackReferenceBuffer(
            max_size=cfg.CH2_REF_BUFFER_MAX,
            min_size_to_use=cfg.CH2_REF_BUFFER_MIN
        )
        
        # Diagnostic
        ae = global_model.autoencoder
        ae.eval()
        with torch.no_grad():
            sample = clients[0].train_data[:100]
            recon, _ = ae(sample)
            mse = F.mse_loss(recon, sample, reduction='none').mean(dim=1)
            all_mse = mse.cpu().numpy()
        ae.train()

        above_threshold = (all_mse > cfg.CH2_MSE_SPLIT_THRESHOLD).sum()
        logger.debug(f"[INIT] Sanity check: {above_threshold}/100 flows above CH2 threshold "
              f"(expected ~25 for P75 threshold)")
        logger.debug(f"  MSE range: [{all_mse.min():.6f}, {all_mse.max():.6f}]")
        logger.debug(f"  MSE P75: {np.percentile(all_mse, 75):.6f}")
        logger.debug(f"  MSE P90: {np.percentile(all_mse, 90):.6f}")
        logger.debug(f"  MSE P99: {np.percentile(all_mse, 99):.6f}")
    else:
        attack_ref_buffer = None
        
    for rnd in range(1, num_rounds + 1):
        print("\n" + "-" * 60)
        print(f"  [{strategy_name} | {byzantine_ratio*100:.0f}% Byzantine] Round {rnd}/{num_rounds}")
        print("-" * 60)

        # Each client trains locally
        client_updates: List[List[np.ndarray]] = []
        c_ae_deltas = []
        c_head_deltas = []
        round_z_submissions = {}
        
        for idx, client in enumerate(clients):
            is_byzantine = (roles[idx] == "byzantine")
            if mode == "dual_channel":
                global_ae_weights = {k: global_model.autoencoder.state_dict()[k].clone() for k in global_model.autoencoder.state_dict()}
                global_head_weights = {k: global_model.attack_head.state_dict()[k].clone() for k in global_model.attack_head.state_dict()}
                
                ae_opt = torch.optim.Adam(client.model.autoencoder.parameters(), lr=1e-3)
                head_opt = torch.optim.Adam(client.model.attack_head.parameters(), lr=1e-3)
                
                with torch.no_grad():
                    for p, arr in zip(client.model.autoencoder.parameters(), global_model.autoencoder.parameters()):
                        p.copy_(arr)
                    for p, arr in zip(client.model.attack_head.parameters(), global_model.attack_head.parameters()):
                        p.copy_(arr)
                
                ae_delta, head_delta, z_buffer, n_benign, n_high_mse = _run_local_training_dual(
                    ae=client.model.autoencoder,
                    attack_head=client.model.attack_head,
                    all_flows=client.train_data,
                    ae_optimizer=ae_opt,
                    head_optimizer=head_opt,
                    global_ae_weights=global_ae_weights,
                    global_head_weights=global_head_weights,
                    mse_threshold_high=cfg.CH2_MSE_SPLIT_THRESHOLD,
                    head_epochs=3
                )
                c_ae_deltas.append(ae_delta)
                c_head_deltas.append(head_delta)
                round_z_submissions[idx] = z_buffer
                
                logger.info(
                    f"Client {idx} round {rnd}: "
                    f"benign_flows={n_benign}, high_mse_flows={n_high_mse}, "
                    f"z_buffer_size={sum(len(z) for z in z_buffer)}"
                )
                train_loss = 0.0
                updated_arrays = [p.detach().cpu().numpy() for p in client.model.parameters()]
            else:
                updated_arrays, train_loss = _run_local_training(client, global_arrays)
                
            role_tag = roles[idx]
            print(f"  Client {idx:2d} [{role_tag:10s}]  train_loss={train_loss:.4f}")
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
                for p, arr in zip(global_model.parameters(), global_arrays):
                    p.copy_(torch.tensor(arr, dtype=torch.float32))

            from aura.fl_server import ae_only_fltrust_aggregate, joint_dual_fltrust_aggregate, dc_fltrust_aggregate
            
            root_ae = FlowAutoencoder()
            root_head = AttackHead()
            import os
            assert os.path.exists('saved_models/autoencoder_best.pth'), \
                "Pretrained AE must exist before computing root gradient"
            
            # CRITICAL: Must use EXACTLY the same weights the clients start from this round.
            # global_model has the latest federated weights (which in round 1 is the pretrained bundle).
            root_ae.load_state_dict(global_model.autoencoder.state_dict())
            root_head.load_state_dict(global_model.attack_head.state_dict())
            
            print(f"[ROOT] Loaded pretrained AE for root gradient computation")
            root_ae_opt = torch.optim.Adam(root_ae.parameters(), lr=1e-3)
            root_head_opt = torch.optim.Adam(root_head.parameters(), lr=1e-3)
            
            g_ae_w = {k: v.clone() for k, v in global_model.autoencoder.state_dict().items()}
            g_head_w = {k: v.clone() for k, v in global_model.attack_head.state_dict().items()}
            
            # ── Strategy B: Symmetric Mini-batch root AE reference ────────────
            # The server's AE reference must execute the same optimization trajectory
            # as an honest client in one FL round. We wrap root_data in a DataLoader
            # exactly like run_two_pass_local_training.

            # Pass 0: classify flows without updating weights
            root_ae.eval()
            with torch.no_grad():
                _recon_pass0, _ = root_ae(root_data)
                mse_per_root_flow = F.mse_loss(_recon_pass0, root_data, reduction='none').mean(dim=1)
            
            root_benign_mask = mse_per_root_flow < cfg.CH2_MSE_SPLIT_THRESHOLD
            filtered_root_data = root_data[root_benign_mask]
            
            discarded_root_data = root_data[~root_benign_mask]
            kept_mse = mse_per_root_flow[root_benign_mask]
            discarded_mse = mse_per_root_flow[~root_benign_mask]
            
            print(f"[ROOT DIAGNOSTICS]")
            print(f"  Initial root samples:    {len(root_data)}")
            print(f"  Filtered root samples:   {len(filtered_root_data)}")
            print(f"  Discarded root samples:  {len(discarded_root_data)}")
            print(f"  Percentage discarded:    {100.0 * len(discarded_root_data) / len(root_data):.2f}%")
            if len(kept_mse) > 0: print(f"  Mean kept MSE:           {kept_mse.mean().item():.6f}")
            if len(discarded_mse) > 0: print(f"  Mean discarded MSE:      {discarded_mse.mean().item():.6f}")

            actual_bs = min(cfg.AE_BATCH_SIZE, len(filtered_root_data)) if cfg.AE_BATCH_SIZE > 0 else len(filtered_root_data)
            if len(filtered_root_data) > 0:
                root_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(filtered_root_data),
                    batch_size=actual_bs, shuffle=True
                )
            else:
                root_loader = []
                
            print(f"[ROOT] Strategy B: Symmetric Mini-batch AE steps "
                  f"(root={len(filtered_root_data)}, bs={actual_bs})")
            
            root_ae.train()
            ch1_export_state = None
            step_count = 0
            
            for (batch,) in root_loader:
                root_ae_opt.zero_grad()
                _recon, _ = root_ae(batch)
                _ae_loss = F.mse_loss(_recon, batch)
                _ae_loss.backward()
                root_ae_opt.step()
                
                step_count += 1
                if step_count == 32:
                    ch1_export_state = {k: v.clone() for k, v in root_ae.state_dict().items()}
                    
            if ch1_export_state is None:
                ch1_export_state = {k: v.clone() for k, v in root_ae.state_dict().items()}
            
            r_ae_delta = {k: ch1_export_state[k] - g_ae_w[k]
                          for k in g_ae_w}
            
            from aura.root_gradient import _build_root_head_reference
            r_head_delta = _build_root_head_reference(
                server_attack_windows=server_attack_windows,
                ae=global_model.autoencoder,
                global_head_weights=g_head_w,
                mse_threshold=cfg.CH2_MSE_SPLIT_THRESHOLD
            )
            
            c_ae_deltas = []
            c_head_deltas = []
            round_z_submissions = {}
            
            from scripts.experiments.byzantine_deception_experiment import _run_latent_inversion_byzantine, _run_true_labelflip_byzantine
            
            for idx, client in enumerate(clients):
                is_byzantine = (roles[idx] == "byzantine")
                
                ae_opt = torch.optim.Adam(client.model.autoencoder.parameters(), lr=1e-3)
                head_opt = torch.optim.Adam(client.model.attack_head.parameters(), lr=1e-3)
                
                with torch.no_grad():
                    for p, arr in zip(client.model.autoencoder.parameters(), global_model.autoencoder.parameters()):
                        p.copy_(arr)
                    for p, arr in zip(client.model.attack_head.parameters(), global_model.attack_head.parameters()):
                        p.copy_(arr)
                        
                if is_byzantine and attack_mode == 'latent_inversion':
                    ae_delta, head_delta, z_buffer, n_benign, n_attack = _run_latent_inversion_byzantine(
                        client.model.autoencoder, client.model.attack_head, client.train_data, ae_opt, head_opt,
                        g_ae_w, g_head_w, mse_threshold_high=cfg.CH2_MSE_SPLIT_THRESHOLD, head_epochs=3
                    )
                elif is_byzantine and attack_mode == 'true_labelflip':
                    ae_delta, head_delta, z_buffer, n_benign, n_attack = _run_true_labelflip_byzantine(
                        client.model.autoencoder, client.model.attack_head, client.train_data, ae_opt, head_opt,
                        g_ae_w, g_head_w, mse_threshold_high=cfg.CH2_MSE_SPLIT_THRESHOLD, head_epochs=3
                    )
                else:
                    ae_delta, head_delta, z_buffer, n_benign, n_attack = _run_local_training_dual(
                        client.model.autoencoder, client.model.attack_head, client.train_data, ae_opt, head_opt,
                        g_ae_w, g_head_w, mse_threshold_high=cfg.CH2_MSE_SPLIT_THRESHOLD, head_epochs=3
                    )
                    
                c_ae_deltas.append(ae_delta)
                c_head_deltas.append(head_delta)
                round_z_submissions[idx] = z_buffer
                
                logger.info(
                    f"Client {idx} round {rnd}: "
                    f"benign_flows={n_benign}, high_mse_flows={n_attack}, "
                    f"z_buffer_size={sum(len(z) for z in z_buffer)}"
                )
            
            client_round_counts = [rnd] * num_clients

            import torch.nn.functional as F_func
            def _flat_norm(d):
                return torch.cat([v.flatten() for v in d.values()]).norm()
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

            root_head_flat = torch.cat([v.flatten() for v in r_head_delta.values()])
            print(f"[DIAGNOSTIC] root_head_delta norm: {root_head_flat.norm():.6f}")
            print(f"[DIAGNOSTIC] root_ae_delta norm: {torch.cat([v.flatten() for v in r_ae_delta.values()]).norm():.6f}")
            
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
                print(f"Root AE delta norm: {_flat_norm(r_ae_delta):.6f}")
                print(f"Root AttackHead delta norm: {_flat_norm(r_head_delta):.6f}")
                print("--- BENCHMARK PRE-AGGREGATION DIAGNOSTICS ---")
                
                for idx, (ae_d, h_d, role) in enumerate(zip(c_ae_deltas, c_head_deltas, roles)):
                    raw_cos = _cos(r_ae_delta, ae_d)
                    relu_cos = benchmark_ch1[idx]
                    ch2_cos = _cos(r_head_delta, h_d) if h_d is not None else 0.0
                    print(f"Client {idx} | Role: {role} | Raw: {raw_cos:.6f} | ReLU: {relu_cos:.6f} | Ch2: {ch2_cos:.6f}")
                print("---------------------------------------------")

                new_ae, new_head, ch1_scores, ch2_scores, classifications, exclusion_flags = dc_fltrust_aggregate(
                    c_ae_deltas, c_head_deltas, r_ae_delta, r_head_delta, client_round_counts,
                    ch2_warmup_rounds=cfg.CH2_WARMUP_ROUNDS,
                    round_z_submissions=round_z_submissions,
                    attack_ref_buffer=attack_ref_buffer,
                    current_round=rnd,
                    reference_attack_head=global_model.attack_head,
                    ch1_threshold=cfg.FLTRUST_CH1_THRESHOLD
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
            
            new_arrays = [p.detach().cpu().numpy() for p in global_model.parameters()]
            
            if mode == 'dc_fltrust':
                flagged_indices = [i for i, excl in enumerate(exclusion_flags) if excl]
            else:
                flagged_indices = [i for i, c in enumerate(classifications) if 'BYZANTINE' in c]
            
            for idx in range(num_clients):
                print(f"  [{mode}] Client {idx:2d} [{roles[idx]:10s}] ch1={ch1_scores[idx]:.4f} ch2={ch2_scores[idx] if ch2_scores[idx] is not None else 0.0:.4f} -> {classifications[idx]}")
                
            if attack_ref_buffer is not None:
                print(f"  [DC-FLTrust] Buffer size: {len(attack_ref_buffer._buffer)}")
                
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
            for p, arr in zip(global_model.parameters(), global_arrays):
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
