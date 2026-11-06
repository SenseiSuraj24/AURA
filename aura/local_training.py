import torch
import torch.nn.functional as F

import config as cfg


def route_flows(ae, flows, mse_threshold):
    """Split flows by reconstruction error, without updating weights."""
    ae.eval()
    with torch.no_grad():
        recon, _ = ae(flows)
        mse_per_flow = F.mse_loss(recon, flows, reduction='none').mean(dim=1)
    ae.train()
    benign_mask = mse_per_flow < mse_threshold
    return flows[benign_mask], flows[~benign_mask], mse_per_flow[~benign_mask]


def train_ae_pass(ae, ae_optimizer, flows, batch_size=256, epochs=1, cycle_to_horizon=False):
    """
    Mini-batch AE training on benign flows with the optimization-horizon export policy.

    Target horizon: H = cfg.FL_EXPORT_HORIZON optimizer steps. If local optimization
    reaches H, the step-H snapshot is exported; otherwise the terminal local update
    is exported. cycle_to_horizon repeats the data until H is reached, so a small
    dataset (the server root) follows the same trajectory length as a client.

    Returns (export_state, last_loss, step_count).
    """
    last_loss = 0.0
    export_state = None
    step_count = 0
    if len(flows) > 0:
        actual_bs = min(batch_size, len(flows)) if batch_size > 0 else len(flows)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(flows),
            batch_size=actual_bs, shuffle=True
        )
        epoch = 0
        while True:
            for (batch,) in loader:
                ae_optimizer.zero_grad()
                recon, _ = ae(batch)
                loss = F.mse_loss(recon, batch)
                loss.backward()
                ae_optimizer.step()
                last_loss = loss.item()
                step_count += 1
                if step_count == cfg.FL_EXPORT_HORIZON:
                    export_state = {k: v.clone() for k, v in ae.state_dict().items()}
                    if cycle_to_horizon:
                        break
            epoch += 1
            if (export_state is not None) if cycle_to_horizon else (epoch >= epochs):
                break

    if export_state is None:
        export_state = {k: v.clone() for k, v in ae.state_dict().items()}
    return export_state, last_loss, step_count


def run_two_pass_local_training(ae, attack_head, all_flows,
                                 ae_optimizer, head_optimizer,
                                 mse_threshold, head_epochs=3,
                                 privacy_engine=None,
                                 batch_size=256,
                                 local_epochs=1):
    """
    Canonical two-pass local training used by both Flower client
    and benchmark scripts. Single implementation — no divergence.
    """
    # Pass 0: classify flows without updating weights
    benign_flows, high_mse_flows, high_mse_values = route_flows(ae, all_flows, mse_threshold)

    # Pass 1: AE trains on benign flows only
    ch1_export_state, ae_loss_val, _ = train_ae_pass(ae, ae_optimizer, benign_flows, batch_size, epochs=local_epochs)

    # Pass 2: inference-only z collection
    z_buffer = []
    if len(high_mse_flows) > 0:
        ae.eval()
        with torch.no_grad():
            for i in range(0, len(high_mse_flows), 256):
                batch = high_mse_flows[i:i+256]
                _, z = ae(batch)
                z_buffer.append(z.detach().cpu())
        ae.train()

    # AttackHead training with soft MSE weighting
    if z_buffer:
        z_tensor = torch.cat(z_buffer)
        mse_weights = high_mse_values.cpu()[:len(z_tensor)]
        mse_weights = (mse_weights - mse_weights.min()) / \
                      (mse_weights.max() - mse_weights.min() + 1e-8)
        for _ in range(head_epochs):
            head_optimizer.zero_grad()
            preds = attack_head(z_tensor).squeeze(-1)
            pseudo_labels = torch.ones(len(z_tensor))
            loss = F.binary_cross_entropy(preds, pseudo_labels, weight=mse_weights)
            loss.backward()
            head_optimizer.step()

    return z_buffer, len(benign_flows), len(high_mse_flows), ae_loss_val, ch1_export_state
