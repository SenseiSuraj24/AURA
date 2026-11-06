"""
train.py — AURA Model Training Pipeline
=========================================
Trains the FlowAutoencoder on benign NF-UNSW-NB15-v3 training flows and the
STGNN on training graph windows, using the canonical data layer
(aura/dataset.py). Also fits and saves the feature preprocessor and writes
saved_models/manifest.json, which every downstream script checks.

Usage:
  python train.py                       # Train both models (cfg.TRAIN_DATA_FRACTION)
  python train.py --ae-only             # Autoencoder only
  python train.py --epochs 10           # Override epoch count
  python train.py --train-fraction 1.0  # Use the full training pool
  python train.py --quick               # 5% of the pool, 5 epochs (smoke run)
"""

import argparse
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config as cfg
from aura import dataset as ds
from aura.models import FlowAutoencoder, AuraSTGNN, AURAModelBundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def seed_everything(seed: int = cfg.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Autoencoder Training
# ─────────────────────────────────────────────────────────────────────────────

def train_autoencoder(
    ae:         FlowAutoencoder,
    train_data: torch.Tensor,
    val_data:   torch.Tensor,
    epochs:     int = cfg.AE_EPOCHS,
    device:     str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple:
    """
    Train the unsupervised autoencoder on benign flow features with pure MSE
    reconstruction loss. Keeps the best-validation checkpoint.

    Returns (autoencoder, best_val_loss).
    """
    ae = ae.to(device)
    optimizer  = torch.optim.Adam(ae.parameters(), lr=cfg.AE_LEARNING_RATE)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(
        TensorDataset(train_data.to(device)),
        batch_size=cfg.AE_BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(cfg.SEED),
    )
    val_loader = DataLoader(
        TensorDataset(val_data.to(device)),
        batch_size=cfg.AE_BATCH_SIZE
    )

    best_val_loss = float("inf")

    print(f"\n{'='*58}")
    print(f"  Training FlowAutoencoder: {epochs} epochs  device={device}")
    print(f"{'='*58}")

    for epoch in range(1, epochs + 1):
        ae.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            x_hat, z = ae(batch)

            loss = ae.reconstruction_loss(batch, x_hat, z)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Validation
        ae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (vbatch,) in val_loader:
                xh, _ = ae(vbatch)
                val_loss += nn.functional.mse_loss(xh, vbatch).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ae.state_dict(), cfg.MODELS_DIR / "autoencoder_best.pth")

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train={epoch_loss/len(train_loader):.5f}  "
                  f"val={val_loss:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

    # Load best model
    ae.load_state_dict(torch.load(cfg.MODELS_DIR / "autoencoder_best.pth", weights_only=True))
    print(f"\n[OK] Autoencoder training complete.  Best val loss: {best_val_loss:.5f}")
    return ae, best_val_loss


def train_stgnn(
    gnn:     AuraSTGNN,
    graphs:  list,   # List of (graph, edge_labels) windows
    epochs:  int = cfg.GNN_EPOCHS,
    device:  str = "cpu",
) -> AuraSTGNN:
    """
    Train the STGNN with Binary Cross-Entropy on per-node anomaly scores.

    Node labels are approximated from edge labels (any node with an incident
    attack edge is suspicious) — weak supervision. The loss covers only nodes
    active in the window: inactive hosts have all-zero features and carry no
    information.
    """
    gnn = gnn.to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=cfg.GNN_LEARNING_RATE)
    bce_loss  = nn.BCELoss()

    print(f"\n{'='*58}")
    print(f"  Training AuraSTGNN: {epochs} epochs  device={device}")
    print(f"  Graphs available: {len(graphs)}")
    print(f"{'='*58}")

    for epoch in range(1, epochs + 1):
        gnn.train()
        epoch_loss = 0.0

        for graph, edge_labels in graphs:
            x          = graph["x"].to(device)
            edge_index = graph["edge_index"].to(device)
            active     = graph["active_nodes"].to(device)

            # Approximate node labels: node is suspicious if any incident edge is attack
            N = x.shape[0]
            node_labels = torch.zeros(N, device=device)
            if edge_labels.sum() > 0:
                attack_edges = edge_labels.bool()
                src = edge_index[0][attack_edges]
                dst = edge_index[1][attack_edges]
                node_labels[src] = 1.0
                node_labels[dst] = 1.0

            optimizer.zero_grad()
            scores, _ = gnn(x, edge_index)
            loss = bce_loss(scores[active], node_labels[active])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={epoch_loss/max(len(graphs), 1):.5f}")

    torch.save(gnn.state_dict(), cfg.MODELS_DIR / "stgnn_trained.pth")
    print(f"[OK] STGNN training complete.")
    return gnn


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AURA Model Training")
    parser.add_argument("--ae-only",  action="store_true", help="Train autoencoder only")
    parser.add_argument("--epochs",   type=int, default=None, help="Override epoch count")
    parser.add_argument("--train-fraction", type=float, default=None,
                        help=f"Portion of the training pool to use (default {cfg.TRAIN_DATA_FRACTION})")
    parser.add_argument("--quick",    action="store_true", help="5%% of the pool, 5 epochs")
    args = parser.parse_args()

    seed_everything()
    epochs = 5 if args.quick else (args.epochs or cfg.AE_EPOCHS)
    train_fraction = 0.05 if args.quick else (args.train_fraction or cfg.TRAIN_DATA_FRACTION)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training device: {device}")

    # ── Step 1: Canonical split + preprocessor ───────────────────────────────
    print(f"\n[Phase 1] Canonical split (train fraction {train_fraction}) and preprocessor …")
    table = ds.load_flow_table()
    split = ds.compute_split(train_fraction, table)
    prep = ds.fit_preprocessor(split, table)

    # ── Step 2: Benign-only AE training flows ────────────────────────────────
    # De-duplicated benign TRAINING flows (label == 0 per flow, not per window),
    # minus IsolationForest outliers — the poisoned-baseline defence.
    benign = split.benign_flows("train", table)
    inlier = prep.is_inlier(table.raw[benign], table.raw_columns)
    ae_flows = benign[inlier]                     # time-ordered (flow indices are sorted)
    n_val   = int(len(ae_flows) * cfg.TEST_SPLIT_FRACTION)
    n_train = len(ae_flows) - n_val
    train_tensor = ds.flows_tensor(ae_flows[:n_train], prep)
    val_tensor   = ds.flows_tensor(ae_flows[n_train:], prep)
    logger.info(f"AE flows: {len(benign):,} benign train flows, {int((~inlier).sum()):,} "
                f"IsolationForest outliers removed → train {n_train:,} / val {n_val:,} "
                f"(chronological), {cfg.FEATURE_DIM} features")

    # ── Step 3: Train Autoencoder ────────────────────────────────────────────
    print("[Phase 3] Training FlowAutoencoder …")
    seed_everything()
    ae, best_val_loss = train_autoencoder(FlowAutoencoder(), train_tensor, val_tensor,
                                          epochs=epochs, device=device)

    # ── Step 4: Train STGNN (optional) ───────────────────────────────────────
    gnn = AuraSTGNN()
    n_gnn_graphs = 0
    if not args.ae_only:
        train_windows = ds.WindowStore(prep, split.windows["train"], table)
        # Seed with the first windows, then top up with attack-containing ones.
        graphs = [train_windows[i] for i in range(min(cfg.GNN_ATTACK_GRAPH_CAP, len(train_windows)))]
        for i in range(len(graphs), len(train_windows)):
            if len(graphs) >= cfg.GNN_ATTACK_GRAPH_CAP * 2:
                break
            item = train_windows[i]
            if item[1].sum() > 0:
                graphs.append(item)
        # Materialise once — the lazy store rebuilds a window on every access.
        graphs = [({k: g[k] for k in ("x", "edge_index", "active_nodes")}, lbl)
                  for g, lbl in graphs]
        n_gnn_graphs = len(graphs)
        print(f"[Phase 4] Training AuraSTGNN on {n_gnn_graphs} train windows …")
        seed_everything()
        gnn = train_stgnn(gnn, graphs, epochs=min(epochs, cfg.GNN_EPOCHS), device=device)
    else:
        logger.info("Skipping STGNN training (--ae-only).")

    # ── Step 5: Save full bundle (G0) ────────────────────────────────────────
    # The bundle includes a seeded, freshly initialised AttackHead so every
    # federation run starts from the same complete G0.
    seed_everything()
    bundle = AURAModelBundle()
    bundle.autoencoder.load_state_dict(ae.state_dict())
    bundle.stgnn.load_state_dict(gnn.state_dict())
    bundle_path = cfg.MODELS_DIR / "aura_bundle.pth"
    torch.save(bundle.state_dict(), bundle_path)
    print(f"\n[OK] Full AURA bundle saved: {bundle_path}")
    print(f"  Total parameters: {bundle.total_params():,}")

    # ── Step 6: Manifest ─────────────────────────────────────────────────────
    ds.write_manifest(prep, split, extra={
        "training": {
            "ae_epochs": epochs, "ae_best_val_mse": best_val_loss,
            "ae_train_flows": n_train, "ae_val_flows": n_val,
            "if_outliers_removed": int((~inlier).sum()),
            "gnn_trained": not args.ae_only, "gnn_graphs": n_gnn_graphs,
        },
    })
    print(f"[OK] Manifest written: {ds.MANIFEST_FILE}")

    print(f"\n{'='*58}")
    print("  AURA Training Complete — next: python calibrate_thresholds.py")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
