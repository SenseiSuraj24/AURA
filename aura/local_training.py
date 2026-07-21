import torch
import torch.nn.functional as F

def run_two_pass_local_training(ae, attack_head, all_flows,
                                 ae_optimizer, head_optimizer,
                                 mse_threshold, head_epochs=3,
                                 privacy_engine=None,
                                 batch_size=256):
    """
    Canonical two-pass local training used by both Flower client
    and benchmark scripts. Single implementation — no divergence.
    """
    # Pass 0: classify flows without updating weights
    ae.eval()
    with torch.no_grad():
        recon, _ = ae(all_flows)
        mse_per_flow = F.mse_loss(recon, all_flows, reduction='none').mean(dim=1)
    ae.train()

    benign_mask = mse_per_flow < mse_threshold
    benign_flows = all_flows[benign_mask]
    high_mse_flows = all_flows[~benign_mask]
    high_mse_values = mse_per_flow[~benign_mask]

    # Pass 1: AE trains on benign flows only
    ae_loss_val = 0.0
    
    # -----------------------------------------------------------
    # Optimization-Horizon Export Policy
    # Target optimization horizon: H = 32 optimizer steps.
    # If local optimization reaches H, the Step32 snapshot is exported.
    # Otherwise, the terminal local update is exported.
    # -----------------------------------------------------------
    ch1_export_state = None
    step_count = 0
    if len(benign_flows) > 0:
        actual_bs = min(batch_size, len(benign_flows)) if batch_size > 0 else len(benign_flows)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(benign_flows),
            batch_size=actual_bs, shuffle=True
        )
        for (batch,) in loader:
            ae_optimizer.zero_grad()
            recon, _ = ae(batch)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            ae_optimizer.step()
            ae_loss_val = loss.item()
            step_count += 1
            if step_count == 32:
                ch1_export_state = {k: v.clone() for k, v in ae.state_dict().items()}
                
    if ch1_export_state is None:
        ch1_export_state = {k: v.clone() for k, v in ae.state_dict().items()}

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
            preds = attack_head(z_tensor).squeeze()
            pseudo_labels = torch.ones(len(z_tensor))
            loss = F.binary_cross_entropy(preds, pseudo_labels, weight=mse_weights)
            loss.backward()
            head_optimizer.step()

    return z_buffer, len(benign_flows), len(high_mse_flows), ae_loss_val, ch1_export_state
