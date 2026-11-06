import torch
import torch.nn.functional as F
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


def attack_flows_from_windows(server_attack_windows: List[Tuple]) -> torch.Tensor:
    """Collect the known-attack flows from the reserved server attack windows."""
    flows = []
    for graph, labels in server_attack_windows:
        attack_mask = (labels == 1)
        if attack_mask.any():
            flows.append(graph['edge_attr'][attack_mask])
    return torch.cat(flows, dim=0) if flows else torch.empty(0)


def root_head_delta(
    attack_flows: torch.Tensor,
    ae: torch.nn.Module,
    global_head_weights: Dict[str, torch.Tensor],
    server_lr: float = 1e-3,
    steps: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Computes the server's root reference delta for the AttackHead (Channel 2).

    Passes known attack flows (reserved windows, never in client train/test splits)
    through the given Autoencoder and trains the AttackHead on the z-vectors with
    pseudo-labels of 1.0 for `steps` full-batch steps.
    """
    from aura.models import AttackHead

    if len(attack_flows) == 0:
        logger.warning("[ROOT] No attack flows available. Head delta will be zero.")
        return {k: torch.zeros_like(v) for k, v in global_head_weights.items()}

    ae.eval()
    with torch.no_grad():
        # Process in batches to avoid OOM if there are many flows
        all_z = torch.cat([ae(attack_flows[i:i + 1024])[1]
                           for i in range(0, len(attack_flows), 1024)], dim=0)

    root_head = AttackHead()
    root_head.load_state_dict(global_head_weights)
    root_head.train()
    head_opt = torch.optim.Adam(root_head.parameters(), lr=server_lr)

    for _ in range(steps):
        head_opt.zero_grad()
        preds = root_head(all_z).squeeze(-1)
        F.binary_cross_entropy(preds, torch.ones_like(preds)).backward()
        head_opt.step()

    return {k: root_head.state_dict()[k].clone() - global_head_weights[k]
            for k in global_head_weights}


def _build_root_head_reference(
    server_attack_windows: List[Tuple],
    ae: torch.nn.Module,
    global_head_weights: Dict[str, torch.Tensor],
    mse_threshold: float,
    server_lr: float = 1e-3
) -> Dict[str, torch.Tensor]:
    """Window-based form of root_head_delta(), kept for the verification scripts."""
    return root_head_delta(attack_flows_from_windows(server_attack_windows),
                           ae, global_head_weights, server_lr)
