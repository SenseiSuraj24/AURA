import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import joblib
from pathlib import Path

AURA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AURA_ROOT))

import config as cfg
from aura.data_loader import CICIDSDataLoader
from aura.split_manager import get_canonical_split
from aura.models import FlowAutoencoder, AttackHead
from scripts.benchmark_byzantine import generate_client_data
from scripts.experiments.byzantine_deception_experiment import _run_latent_inversion_byzantine

def flatten(d): return torch.cat([v.flatten() for v in d.values()]).cpu()
def cos(a, b): return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

class SnapshotAdam(torch.optim.Adam):
    def __init__(self, params, ae_model, base_state, target_steps, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.ae_model = ae_model
        self.base_state = base_state
        self.target_steps = target_steps
        self.step_count = 0
        self.snapshots = {}
    def step(self, closure=None):
        loss = super().step(closure)
        self.step_count += 1
        if self.step_count in self.target_steps:
            current_state = self.ae_model.state_dict()
            self.snapshots[self.step_count] = flatten({k: current_state[k].clone() - self.base_state[k] for k in current_state})
        return loss

def evaluate_attack(attack_mode):
    print(f"\nEvaluating {attack_mode.upper()}...")
    loader = CICIDSDataLoader()
    scaler = joblib.load(os.path.join(cfg.MODELS_DIR, "scaler.joblib"))
    all_windows = list(loader.stream_graphs(scaler))
    calib, train, test, attack = get_canonical_split(all_windows, test_fraction=0.20)
    
    def get_benign(windows):
        flows = [g['edge_attr'][l==0] for g, l in windows if (l==0).any()]
        return torch.cat(flows) if flows else torch.empty(0)
    
    root_data = get_benign(calib)[:cfg.FLTRUST_ROOT_SAMPLES]
    bundle_path = os.path.join(cfg.MODELS_DIR, "aura_bundle.pth")
    saved_state = torch.load(bundle_path, map_location='cpu', weights_only=True)
    global_ae = {k.replace('autoencoder.', ''): v for k, v in saved_state.items() if k.startswith('autoencoder.')}
    global_head = AttackHead().state_dict()
    
    target_steps = [8, 16, 32]
    seeds = list(range(10))
    results = {cp: {'hon': [], 'byz': []} for cp in ['Step 8', 'Step 16', 'Step 32', 'Final']}
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Server Root
        root_ae = FlowAutoencoder()
        root_ae.load_state_dict(global_ae)
        root_opt = torch.optim.Adam(root_ae.parameters(), lr=1e-3)
        actual_bs = min(cfg.AE_BATCH_SIZE, len(root_data))
        root_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(root_data), batch_size=actual_bs, shuffle=True)
        for (b,) in root_loader:
            root_opt.zero_grad()
            r, _ = root_ae(b)
            loss = F.mse_loss(r, b)
            loss.backward()
            root_opt.step()
        root_final_delta = flatten({k: root_ae.state_dict()[k].clone() - global_ae[k] for k in global_ae})
        
        num_clients = 5
        for i in range(num_clients):
            is_byz = (i == 0)
            c_train, _ = generate_client_data(i, is_byz, False, num_clients)
            
            cae = FlowAutoencoder()
            cae.load_state_dict(global_ae)
            chead = AttackHead()
            chead.load_state_dict(global_head, strict=False)
            
            c_opt = SnapshotAdam(cae.parameters(), cae, global_ae, set(target_steps), lr=1e-3)
            h_opt = torch.optim.Adam(chead.parameters(), lr=1e-3)
            
            if is_byz and attack_mode == 'latent_inversion':
                ae_d, _, _, _, _ = _run_latent_inversion_byzantine(cae, chead, c_train, c_opt, h_opt, global_ae, global_head, cfg.CH2_MSE_SPLIT_THRESHOLD, head_epochs=3)
                flat_d = flatten(ae_d)
                for step_name in results.keys():
                    results[step_name]['byz'].append(cos(root_final_delta, flat_d))
            else:
                from aura.local_training import run_two_pass_local_training
                run_two_pass_local_training(cae, chead, c_train, c_opt, h_opt, mse_threshold=cfg.CH2_MSE_SPLIT_THRESHOLD, head_epochs=3, batch_size=256)
                final_d = flatten({k: cae.state_dict()[k].clone() - global_ae[k] for k in global_ae})
                

                
                if 8 in c_opt.snapshots: results['Step 8']['byz' if is_byz else 'hon'].append(cos(root_final_delta, c_opt.snapshots[8]))
                if 16 in c_opt.snapshots: results['Step 16']['byz' if is_byz else 'hon'].append(cos(root_final_delta, c_opt.snapshots[16]))
                if 32 in c_opt.snapshots: results['Step 32']['byz' if is_byz else 'hon'].append(cos(root_final_delta, c_opt.snapshots[32]))
                results['Final']['byz' if is_byz else 'hon'].append(cos(root_final_delta, final_d))

    summary = []
    from sklearn.metrics import roc_auc_score
    for step_name, data in results.items():
        hon = np.array(data['hon'])
        byz = np.array(data['byz'])
        mean_hon = np.mean(hon)
        std_hon = np.std(hon)
        mean_byz = np.mean(byz)
        sep = mean_hon - mean_byz
        y_true = np.array([1]*len(byz) + [0]*len(hon))
        y_score = np.concatenate([byz, hon])
        auc = roc_auc_score(y_true, -y_score)
        
        # Calculate thresholds strictly for metrics
        mean_score = np.mean(y_score)
        preds = y_score < mean_score
        tp = np.sum(preds[:len(byz)])
        fp = np.sum(preds[len(byz):])
        fn = np.sum(~preds[:len(byz)])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        summary.append({
            'name': step_name, 'sep': sep, 'auc': auc, 'mean_hon': mean_hon,
            'mean_byz': mean_byz, 'std_hon': std_hon,
            'precision': precision, 'recall': recall, 'f1': f1
        })
    return summary

if __name__ == '__main__':
    attacks = ['none', 'latent_inversion', 'true_labelflip']
    all_res = {}
    for a in attacks:
        all_res[a] = evaluate_attack(a)
        
    print("\n" + "="*80)
    print(f"{'Attack':<20} | {'Checkpoint':<10} | {'Sep':<8} | {'AUC':<7} | {'Prec':<5} | {'Rec':<5} | {'F1':<5}")
    print("-" * 80)
    for a in attacks:
        for s in all_res[a]:
            print(f"{a:<20} | {s['name']:<10} | {s['sep']:>8.4f} | {s['auc']:>7.4f} | {s['precision']:>5.2f} | {s['recall']:>5.2f} | {s['f1']:>5.2f}")

