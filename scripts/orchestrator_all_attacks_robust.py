import os
import subprocess
import glob
import pickle
import numpy as np
import time
from sklearn.metrics import roc_auc_score

def patch_production_code():
    # local_training.py
    local_train_path = "aura/local_training.py"
    with open(local_train_path, "r", encoding="utf-8") as f:
        lt_code = f.read()
    
    if "step8_state" not in lt_code:
        lt_new = lt_code.replace(
            "    step16_state = None",
            "    step8_state = None\n    step16_state = None\n    step32_state = None"
        ).replace(
            "            if step_count == 16:",
            "            if step_count == 8:\n                step8_state = {k: v.clone() for k, v in ae.state_dict().items()}\n            if step_count == 16:"
        ).replace(
            "                step16_state = {k: v.clone() for k, v in ae.state_dict().items()}",
            "                step16_state = {k: v.clone() for k, v in ae.state_dict().items()}\n            if step_count == 32:\n                step32_state = {k: v.clone() for k, v in ae.state_dict().items()}"
        ).replace(
            "    if step16_state is None:",
            "    if step8_state is None:\n        step8_state = {k: v.clone() for k, v in ae.state_dict().items()}\n    if step16_state is None:"
        ).replace(
            "        step16_state = {k: v.clone() for k, v in ae.state_dict().items()}",
            "        step16_state = {k: v.clone() for k, v in ae.state_dict().items()}\n    if step32_state is None:\n        step32_state = {k: v.clone() for k, v in ae.state_dict().items()}"
        ).replace(
            "    return z_buffer, len(benign_flows), len(high_mse_flows), ae_loss_val, step16_state",
            "    return z_buffer, len(benign_flows), len(high_mse_flows), ae_loss_val, step8_state, step16_state, step32_state, {k: v.clone() for k, v in ae.state_dict().items()}"
        )
        with open(local_train_path, "w", encoding="utf-8") as f:
            f.write(lt_new)
            
    # benchmark_byzantine.py
    bench_path = "scripts/benchmark_byzantine.py"
    with open(bench_path, "r", encoding="utf-8") as f:
        bm_code = f.read()
        
    if "step8_state" not in bm_code and "'step8':" not in bm_code:
        bm_new = bm_code.replace(
            "    z_buffer, n_benign, n_high_mse, _, step16_state = run_two_pass_local_training(",
            "    z_buffer, n_benign, n_high_mse, _, step8, step16, step32, final_state = run_two_pass_local_training("
        )
        
        # Replace ae_delta construction robustly
        import re
        bm_new = re.sub(
            r"    ae_delta = \{k: step16_state\[k\] - global_ae_weights\[k\].*?for k in step16_state\}",
            "    ae_delta = {\n        'step8': {k: step8[k] - global_ae_weights[k] for k in step8},\n        'step16': {k: step16[k] - global_ae_weights[k] for k in step16},\n        'step32': {k: step32[k] - global_ae_weights[k] for k in step32},\n        'final': {k: final_state[k] - global_ae_weights[k] for k in final_state}\n    }",
            bm_new, flags=re.DOTALL
        )
        with open(bench_path, "w", encoding="utf-8") as f:
            f.write(bm_new)

def revert_production_code():
    subprocess.run("git checkout aura/local_training.py scripts/benchmark_byzantine.py", shell=True)

def run_benchmarks():
    attacks = ["none", "latent_inversion", "true_labelflip"]
    seeds = list(range(10))
    
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    
    for f in glob.glob("saved_models/exported_tensors_*.pkl"):
        os.remove(f)
        
    procs = []
    for attack in attacks:
        for seed in seeds:
            cmd = f"python scripts/benchmark_byzantine.py --mode dc_fltrust --attack-mode {attack} --rounds 12 --seed {seed} --export-tensors"
            p = subprocess.Popen(cmd, shell=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            procs.append(p)
            
    print("Waiting for all 30 processes to finish...")
    for p in procs:
        p.wait()
        
    for attack in attacks:
        print(f"\n=======================================================")
        print(f"--- Running Attack Mode: {attack.upper()} ---")
        print(f"=======================================================")
        
        import torch
        import torch.nn.functional as F_func
        def _flat(d): return torch.cat([v.flatten() for v in d.values()])
        def _cos(t1, t2):
            if t1.norm() == 0 or t2.norm() == 0: return 0.0
            return F_func.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()
            
        checkpoints = ["final", "step8", "step16", "step32"]
        results = {cp: {'hon': [], 'byz': []} for cp in checkpoints}
        
        for f in glob.glob(f"saved_models/exported_tensors_{attack}_*.pkl"):
            try:
                d = pickle.load(open(f, 'rb'))
                r_ae_delta = d['root_ae_delta']
                c_ae_deltas = d.get('client_ae_deltas_all_steps', d['client_ae_deltas'])
                roles = d['roles']
                
                t_root = _flat(r_ae_delta)
                
                for client_d, role in zip(c_ae_deltas, roles):
                    for cp in checkpoints:
                        t_client = _flat(client_d[cp])
                        score = _cos(t_root, t_client)
                        if role == 'byzantine':
                            results[cp]['byz'].append(score)
                        else:
                            results[cp]['hon'].append(score)
            except Exception as e:
                pass
                
        for cp in checkpoints:
            hon = np.array(results[cp]['hon'])
            byz = np.array(results[cp]['byz'])
            if len(hon) == 0 or len(byz) == 0: continue
            
            mean_hon = np.mean(hon)
            std_hon = np.std(hon)
            mean_byz = np.mean(byz)
            sep = mean_hon - mean_byz
            
            y_true = np.array([1]*len(byz) + [0]*len(hon))
            y_score = np.concatenate([byz, hon])
            auc = roc_auc_score(y_true, -y_score)
            
            print(f"Attack: {attack:16} | Checkpoint: {cp:6} | Hon Mean: {mean_hon:.4f} (±{std_hon:.4f}) | Byz Mean: {mean_byz:.4f} | Sep: {sep:+.4f} | AUC: {auc:.4f}")

if __name__ == "__main__":
    run_benchmarks()
