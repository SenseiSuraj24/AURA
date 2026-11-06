"""
aura/fl_round.py — one federation round, shared by the server and the benchmarks
================================================================================
FLTrust compares update directions, so the server's reference has to follow the
same recipe as an honest client. Everything that defines that recipe lives here;
the Flower strategy and the benchmarks call these functions rather than keeping
their own copies, so what gets evaluated is what gets deployed.

Only model updates cross the client boundary. Latent vectors stay on the client.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

import config as cfg
from aura.local_training import route_flows, run_two_pass_local_training, train_ae_pass
from aura.models import AttackHead, AURAModelBundle, FlowAutoencoder
from aura.root_gradient import root_head_delta

State = Dict[str, torch.Tensor]


@dataclass
class ClientUpdate:
    ae_delta: State
    head_delta: Optional[State]     # None when the client saw no high-MSE flows
    n_benign: int
    n_high_mse: int
    loss: float


def load_global_model() -> AURAModelBundle:
    """
    G0: the trained bundle every federation path starts from. A random start routes
    every flow away from the AE (all reconstruction errors are high), so clients
    would not train. Refuses stale or mismatched artefacts.
    """
    from aura import dataset as ds
    ds.check_manifest()
    path = cfg.MODELS_DIR / "aura_bundle.pth"
    model = AURAModelBundle()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def snapshot(module: torch.nn.Module) -> State:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def apply_delta(state: State, delta: Optional[State]) -> State:
    """Global weights + aggregated delta (unchanged when there is no delta)."""
    if delta is None:
        return {k: v.clone() for k, v in state.items()}
    return {k: state[k] + delta[k] for k in state}


def client_update(
    ae: FlowAutoencoder,
    head: AttackHead,
    global_ae: State,
    global_head: State,
    flows: torch.Tensor,
) -> ClientUpdate:
    """An honest client's update: load the global weights, train, return deltas."""
    ae.load_state_dict(global_ae)
    head.load_state_dict(global_head)
    ae_opt = torch.optim.Adam(ae.parameters(), lr=cfg.AE_LEARNING_RATE)
    head_opt = torch.optim.Adam(head.parameters(), lr=cfg.AE_LEARNING_RATE)
    _, n_benign, n_high_mse, loss, export = run_two_pass_local_training(
        ae, head, flows, ae_opt, head_opt,
        mse_threshold=cfg.CH2_MSE_SPLIT_THRESHOLD,
        head_epochs=cfg.FL_HEAD_EPOCHS,
        batch_size=cfg.AE_BATCH_SIZE,
        local_epochs=cfg.FL_LOCAL_EPOCHS,
    )
    ae_delta = {k: export[k] - global_ae[k] for k in global_ae}
    head_delta = None
    if n_high_mse > 0:
        head_delta = {k: v.detach() - global_head[k] for k, v in head.state_dict().items()}
    return ClientUpdate(ae_delta, head_delta, n_benign, n_high_mse, loss)


def root_reference(
    global_ae: State,
    global_head: State,
    root_benign: torch.Tensor,
    root_attack: torch.Tensor,
) -> Tuple[State, State]:
    """
    The server's reference update, following the honest-client recipe: same flow
    routing, optimizer, batch size and export horizon. The root set is small, so
    it is repeated until the horizon is reached.
    """
    ae = FlowAutoencoder()
    ae.load_state_dict(global_ae)
    benign, _, _ = route_flows(ae, root_benign, cfg.CH2_MSE_SPLIT_THRESHOLD)
    ae_opt = torch.optim.Adam(ae.parameters(), lr=cfg.FLTRUST_SERVER_LR)
    export, _, _ = train_ae_pass(ae, ae_opt, benign, cfg.AE_BATCH_SIZE, cycle_to_horizon=True)
    ae_delta = {k: export[k] - global_ae[k] for k in global_ae}
    head_delta = root_head_delta(
        root_attack, ae, global_head,
        server_lr=cfg.FLTRUST_SERVER_LR, steps=cfg.FL_HEAD_EPOCHS,
    )
    return ae_delta, head_delta


def aggregate(
    ae_deltas: List[State],
    head_deltas: List[Optional[State]],
    root_ae: State,
    root_head: State,
    current_round: int,
):
    """DC-FLTrust aggregation with the configured thresholds."""
    from aura.fl_server import dc_fltrust_aggregate
    return dc_fltrust_aggregate(
        ae_deltas, head_deltas, root_ae, root_head,
        client_round_counts=[current_round] * len(ae_deltas),
        ch2_warmup_rounds=cfg.CH2_WARMUP_ROUNDS,
        current_round=current_round,
        ch1_threshold=cfg.FLTRUST_CH1_THRESHOLD,
    )
