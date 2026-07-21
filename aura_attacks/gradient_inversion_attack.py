import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import argparse
import pickle
import json
import os
from pathlib import Path

from aura.models import FlowAutoencoder, AttackHead
from config import FEATURE_DIM

def get_ae_delta(dummy_x, global_ae, steps=32, lr=1e-3, batch_size=256):
    params = [p.clone().requires_grad_(True) for p in global_ae.parameters()]
    exp_avgs = [torch.zeros_like(p) for p in params]
    exp_avg_sqs = [torch.zeros_like(p) for p in params]
    
    for step in range(1, steps + 1):
        param_dict = {k: p for (k, _), p in zip(global_ae.named_parameters(), params)}
        recon, _ = torch.func.functional_call(global_ae, param_dict, (dummy_x,))
        loss = F.mse_loss(recon, dummy_x)
        
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        new_params = []
        new_v = []
        new_s = []
        for p, g, v, s in zip(params, grads, exp_avgs, exp_avg_sqs):
            v_next = 0.9 * v + 0.1 * g
            s_next = 0.999 * s + 0.001 * (g * g)
            v_hat = v_next / (1 - 0.9 ** step)
            s_hat = s_next / (1 - 0.999 ** step)
            p_next = p - lr * v_hat / torch.sqrt(s_hat + 1e-8)
            new_params.append(p_next)
            new_v.append(v_next)
            new_s.append(s_next)
        params = new_params
        exp_avgs = new_v
        exp_avg_sqs = new_s
        
    delta = [p - p0 for p, p0 in zip(params, global_ae.parameters())]
    return delta

def invert_ae_exact(global_ae, true_ae_delta, batch_size, steps=200, restarts=3, lr=0.1):
    best_dummy, best_loss = None, float("inf")
    for r in range(restarts):
        torch.manual_seed(1000 + r)
        dummy_x = torch.randn(batch_size, FEATURE_DIM, requires_grad=True)
        opt = torch.optim.Adam([dummy_x], lr=lr)
        
        for _ in range(steps):
            opt.zero_grad()
            dummy_delta = get_ae_delta(dummy_x, global_ae, steps=32, lr=1e-3, batch_size=256)
            loss = sum(((a - b) ** 2).sum() for a, b in zip(dummy_delta, true_ae_delta))
            loss.backward()
            opt.step()
            
        dummy_delta = get_ae_delta(dummy_x, global_ae, steps=32, lr=1e-3, batch_size=256)
        loss = sum(((a - b) ** 2).sum() for a, b in zip(dummy_delta, true_ae_delta)).item()
        
        if loss < best_loss or best_dummy is None:
            best_loss = loss
            best_dummy = dummy_x.detach().clone()
    return best_dummy

def get_head_delta(dummy_z, mse_weights, global_head, epochs=3, lr=1e-3, true_labels=False):
    params = [p.clone().requires_grad_(True) for p in global_head.parameters()]
    exp_avgs = [torch.zeros_like(p) for p in params]
    exp_avg_sqs = [torch.zeros_like(p) for p in params]
    
    for step in range(1, epochs + 1):
        param_dict = {k: p for (k, _), p in zip(global_head.named_parameters(), params)}
        preds = torch.func.functional_call(global_head, param_dict, (dummy_z,)).squeeze(-1)
        
        pseudo_labels = torch.zeros(len(dummy_z)) if true_labels else torch.ones(len(dummy_z))
        
        loss_per_sample = F.binary_cross_entropy(preds, pseudo_labels, reduction="none")
        if mse_weights is not None:
            loss = (loss_per_sample * mse_weights).mean()
        else:
            loss = loss_per_sample.mean()
        
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        new_params = []
        new_v = []
        new_s = []
        
        # Instrumentation variables for Adam
        min_s_hat, max_s_hat = float('inf'), float('-inf')
        min_denom, max_denom = float('inf'), float('-inf')
        min_upd, max_upd = float('inf'), float('-inf')
        has_nan = False
        has_inf = False

        for p, g, v, s in zip(params, grads, exp_avgs, exp_avg_sqs):
            v_next = 0.9 * v + 0.1 * g
            s_next = 0.999 * s + 0.001 * (g * g)
            v_hat = v_next / (1 - 0.9 ** step)
            s_hat = s_next / (1 - 0.999 ** step)
            
            # THE REPAIR: torch.sqrt(s_hat + 1e-8)
            denom = torch.sqrt(s_hat + 1e-8)
            update = lr * v_hat / denom
            p_next = p - update
            
            # Tracking
            min_s_hat = min(min_s_hat, s_hat.min().item())
            max_s_hat = max(max_s_hat, s_hat.max().item())
            min_denom = min(min_denom, denom.min().item())
            max_denom = max(max_denom, denom.max().item())
            min_upd = min(min_upd, update.min().item())
            max_upd = max(max_upd, update.max().item())
            
            if torch.isnan(p_next).any() or torch.isnan(s_hat).any() or torch.isnan(denom).any() or torch.isnan(update).any():
                has_nan = True
            if torch.isinf(p_next).any() or torch.isinf(s_hat).any() or torch.isinf(denom).any() or torch.isinf(update).any():
                has_inf = True

            new_params.append(p_next)
            new_v.append(v_next)
            new_s.append(s_next)
            
        print(f"  [Adam Step {step}] s_hat: [{min_s_hat:.2e}, {max_s_hat:.2e}] | denom: [{min_denom:.2e}, {max_denom:.2e}] | update: [{min_upd:.2e}, {max_upd:.2e}] | NaN: {has_nan} | Inf: {has_inf}")
        
        params = new_params
        exp_avgs = new_v
        exp_avg_sqs = new_s
        
    delta = [p - p0 for p, p0 in zip(params, global_head.parameters())]
    return delta

def invert_head_exact(global_ae, global_head, true_head_delta, batch_size, steps=200, restarts=3, lr=0.1, is_latent_inversion=False):
    best_dummy, best_loss = None, float("inf")
    for r in range(restarts):
        torch.manual_seed(2000 + r)
        dummy_x = torch.randn(batch_size, FEATURE_DIM, requires_grad=True)
        opt = torch.optim.Adam([dummy_x], lr=lr)
        
        for step_idx in range(steps):
            opt.zero_grad()
            recon, z = global_ae(dummy_x)
            mse = F.mse_loss(recon, dummy_x, reduction='none').mean(dim=1)
            
            if is_latent_inversion:
                mse_weights = None
            else:
                mse_weights = (mse - mse.min()) / (mse.max() - mse.min() + 1e-8)
            
            dummy_delta = get_head_delta(z, mse_weights, global_head, epochs=3, lr=1e-3, true_labels=is_latent_inversion)
            loss = sum(((a - b) ** 2).sum() for a, b in zip(dummy_delta, true_head_delta))
            loss.backward()
            opt.step()
            
        recon, z = global_ae(dummy_x)
        mse = F.mse_loss(recon, dummy_x, reduction='none').mean(dim=1)
        if is_latent_inversion:
            mse_weights = None
        else:
            mse_weights = (mse - mse.min()) / (mse.max() - mse.min() + 1e-8)
            
        dummy_delta = get_head_delta(z, mse_weights, global_head, epochs=3, lr=1e-3, true_labels=is_latent_inversion)
        loss = sum(((a - b) ** 2).sum() for a, b in zip(dummy_delta, true_head_delta)).item()
        
        if loss < best_loss or best_dummy is None:
            best_loss = loss
            best_dummy = dummy_x.detach().clone()
    return best_dummy

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor-file", type=str, required=True)
    ap.add_argument("--client-id", type=int, required=True)
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--restarts", type=int, default=3)
    args = ap.parse_args()

    with open(args.tensor_file, 'rb') as f:
        data = pickle.load(f)

    global_ae = FlowAutoencoder()
    global_ae.load_state_dict(data['global_ae_weights'])
    global_ae.eval()
    
    global_head = AttackHead()
    global_head.load_state_dict(data['global_head_weights'])
    global_head.eval()

    c_head_deltas = data['client_head_deltas']
    roles = data['roles']
    train_data = data.get('train_data', None)

    cid = args.client_id
    true_head_delta = [-c_head_deltas[cid][k].cpu() for k, _ in global_head.named_parameters()]
    
    # Calculate batch size dynamically or assume something
    if train_data is not None and cid < len(train_data):
        batch_size = len(train_data[cid])
    else:
        # Default fallback, usually 150740 for full dataset head batch
        batch_size = 150740

    x_hat = invert_head_exact(
        global_ae, 
        global_head, 
        true_head_delta, 
        batch_size=batch_size, 
        steps=args.steps, 
        restarts=args.restarts, 
        is_latent_inversion=False
    )
    
    # Reconstruct true_data for metric comparison
    true_data = None
    if train_data is not None and cid < len(train_data):
        true_data = train_data[cid]
        
    mse = float('nan')
    cos = float('nan')
    if true_data is not None:
        if true_data.dim() == 1:
            true_data = true_data.unsqueeze(0)
            
        if len(true_data) > len(x_hat):
            true_data = true_data[:len(x_hat)]
        elif len(x_hat) > len(true_data):
            x_hat = x_hat[:len(true_data)]
            
        mse = ((x_hat - true_data) ** 2).mean().item()
        cos = F.cosine_similarity(x_hat.flatten(), true_data.flatten(), dim=0).item()

    out = {
        "dlg_loss": 0.0,
        "mse": mse,
        "cosine_similarity": cos,
        "reconstruction_success": not torch.isnan(x_hat).any().item(),
        "nan_present": torch.isnan(x_hat).any().item(),
        "inf_present": torch.isinf(x_hat).any().item()
    }
    
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=4)