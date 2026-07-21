import os
import subprocess
import glob
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
import time

def modify_step(target_step):
    file_path = "aura/local_training.py"
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    with open(file_path, "w") as f:
        for line in lines:
            if "if step_count ==" in line and "step16_state" in line:
                f.write(f"            if step_count == {target_step}:\n")
            else:
                f.write(line)

def run_benchmarks():
    steps = [8, 16, 32, 9999] # 9999 for Final
    seeds = list(range(10))
    results = {}
    
    for step in steps:
        modify_step(step)
        print(f"--- Running for Step {step} ---")
        
        # Clear old tensors
        for f in glob.glob("saved_models/exported_tensors_*.pkl"):
            os.remove(f)
            
        # Spawn 10 processes in parallel
        procs = []
        for seed in seeds:
            print(f"Starting Seed {seed} in background...")
            cmd = f"python scripts/benchmark_byzantine.py --mode dc_fltrust --attack-mode latent_inversion --rounds 12 --seed {seed} --export-tensors"
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            procs.append(p)
            
        print("Waiting for all seeds to finish (approx 10 minutes)...")
        start_time = time.time()
        for p in procs:
            p.wait()
        print(f"Completed in {time.time() - start_time:.1f} seconds")
            
        # Collect results
        byz = []
        hon = []
        for f in glob.glob("saved_models/exported_tensors_*.pkl"):
            try:
                d = pickle.load(open(f, 'rb'))
                scores = d['metadata']['benchmark_ch1']
                roles = d['roles']
                for s, r in zip(scores, roles):
                    if r == 'byzantine': byz.append(s)
                    else: hon.append(s)
            except Exception as e:
                pass
                
        if len(byz) > 0 and len(hon) > 0:
            mean_hon = np.mean(hon)
            mean_byz = np.mean(byz)
            sep = mean_hon - mean_byz
            labels = [1]*len(byz) + [0]*len(hon)
            preds = byz + hon
            auc = roc_auc_score(labels, preds)
            
            # compute fp, fn
            fp = 0
            fn = 0
            mean_score = np.mean(preds)
            for f in glob.glob("saved_models/exported_tensors_*.pkl"):
                try:
                    d = pickle.load(open(f, 'rb'))
                    scores = d['metadata']['benchmark_ch1']
                    roles = d['roles']
                    for s, r in zip(scores, roles):
                        is_byz = (r == 'byzantine')
                        is_flagged = (s < mean_score)
                        if is_flagged and not is_byz: fp += 1
                        if not is_flagged and is_byz: fn += 1
                except:
                    pass
                    
            results[step] = {
                'hon_mean': mean_hon,
                'byz_mean': mean_byz,
                'sep': sep,
                'auc': auc,
                'fp': fp / (len(seeds) * 12),
                'fn': fn / (len(seeds) * 12)
            }
            print(f"Step {step} -> Sep: {sep:.4f}, AUC: {auc:.4f}, FP_avg: {results[step]['fp']:.2f}, FN_avg: {results[step]['fn']:.2f}")
            
    print("\nFINAL RESULTS:")
    for step, r in results.items():
        print(f"Step {step}: Honest={r['hon_mean']:.4f}, Byz={r['byz_mean']:.4f}, Sep={r['sep']:.4f}, AUC={r['auc']:.4f}, FP_avg={r['fp']:.2f}, FN_avg={r['fn']:.2f}")
        
if __name__ == "__main__":
    run_benchmarks()
    modify_step(16)
