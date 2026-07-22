import os
import glob
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

def flatten(delta: dict) -> torch.Tensor:
    return torch.cat([v.flatten() for v in delta.values()])

def signed_cosine(client_flat, root_flat) -> float:
    return F.cosine_similarity(client_flat.unsqueeze(0), root_flat.unsqueeze(0)).item()

def relu_cosine(client_flat, root_flat) -> float:
    return max(0.0, signed_cosine(client_flat, root_flat))

def safe_auc(y_true, scores):
    if len(set(y_true)) <= 1:
        return 0.0
    # Invert scores because lower score = Byzantine (class 1)
    return roc_auc_score(y_true, [-s for s in scores])

def classify_metrics(y_true, y_pred):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    except:
        fp = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1])
        fn = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0])
        
    actual_negatives = len([y for y in y_true if y == 0])
    actual_positives = len([y for y in y_true if y == 1])
    fpr = fp / actual_negatives if actual_negatives > 0 else 0.0
    fnr = fn / actual_positives if actual_positives > 0 else 0.0
    
    return p, r, f1, fp, fn, fpr, fnr

attacks = ['none', 'latent_inversion', 'true_labelflip']

print("FINAL READ-ONLY ARCHITECTURAL VALIDATION\n")
print("========================================")
print("FROZEN PRODUCTION THRESHOLDS: CH1 <= 0.0023, CH2 < 0.12")
print("========================================\n")

for attack in attacks:
    files = glob.glob(f"saved_models/exported_tensors_{attack}_seed_*_round_*.pkl")
    files = [f for f in files if f.endswith("_round_11.pkl") or f.endswith("_round_12.pkl")]
    
    if not files:
        print(f"[{attack.upper()}] No cached tensors found for rounds 11/12.")
        print()
        continue
        
    y_true = []
    ch1_scores = []
    ch2_scores = []
    combined_preds = []
    
    for f in files:
        with open(f, 'rb') as fp:
            d = pickle.load(fp)
            
            roles = d['roles']
            root_ae_flat = flatten(d['root_ae_delta'])
            root_head_flat = flatten(d['root_head_delta'])
            
            for i, role in enumerate(roles):
                is_byz = 1 if role == 'byzantine' else 0
                if attack == 'none':
                    y_true.append(0)
                else:
                    y_true.append(is_byz)
                    
                client_ae_flat = flatten(d['client_ae_deltas'][i])
                client_head = d['client_head_deltas'][i]
                
                ch1 = relu_cosine(client_ae_flat, root_ae_flat)
                if client_head is not None:
                    ch2 = signed_cosine(flatten(client_head), root_head_flat)
                else:
                    ch2 = 0.0
                    
                ch1_scores.append(ch1)
                ch2_scores.append(ch2)
                
                is_ch1_adv = (ch1 <= 0.0023)
                is_ch2_adv = (client_head is not None) and (ch2 < 0.12)
                
                combined_preds.append(1 if (is_ch1_adv or is_ch2_adv) else 0)
                
    print(f"[{attack.upper()}]")
    print(f"Total samples: {len(y_true)} (Honest: {len(y_true)-sum(y_true)}, Byzantine: {sum(y_true)})")
    
    hon_ch1 = [s for y, s in zip(y_true, ch1_scores) if y == 0]
    byz_ch1 = [s for y, s in zip(y_true, ch1_scores) if y == 1]
    
    hon_ch2 = [s for y, s in zip(y_true, ch2_scores) if y == 0]
    byz_ch2 = [s for y, s in zip(y_true, ch2_scores) if y == 1]
    
    def dist_stats(arr):
        if not arr: return ""
        mean, std = np.mean(arr), np.std(arr)
        med = np.median(arr)
        p5, p95 = np.percentile(arr, 5), np.percentile(arr, 95)
        return f"[{min(arr):.4f}, {max(arr):.4f}] (Mean: {mean:.4f} ± {std:.4f}, Med: {med:.4f}, 5th: {p5:.4f}, 95th: {p95:.4f})"

    print("--- DISTRIBUTIONS ---")
    if hon_ch1: print(f"Honest CH1: {dist_stats(hon_ch1)}")
    if byz_ch1: print(f"Byz CH1:    {dist_stats(byz_ch1)}")
    
    if hon_ch2: print(f"Honest CH2: {dist_stats(hon_ch2)}")
    if byz_ch2: print(f"Byz CH2:    {dist_stats(byz_ch2)}")
    
    if attack == 'none':
        p, r, f1, fp, fn, fpr, fnr = classify_metrics(y_true, combined_preds)
        print("--- METRICS (CONTROL GROUP) ---")
        print(f"Combined False Positives: {fp} (FPR: {fpr:.4f})")
    else:
        # We can simulate a continuous combined score for ROC-AUC by using the min of (CH1, modified CH2)
        # However, it's cleaner to just report CH1 ROC and CH2 ROC as architectural capacity.
        # Let's map CH2 such that < -0.1 triggers anomaly.
        # Combined score: min(ch1, ch2 + 0.35) so that 0.25 and -0.1 align.
        combined_scores = [min(c1, c2 + 0.35) for c1, c2 in zip(ch1_scores, ch2_scores)]
        
        auc_ch1 = safe_auc(y_true, ch1_scores)
        auc_ch2 = safe_auc(y_true, ch2_scores)
        auc_comb = safe_auc(y_true, combined_scores)
        
        p, r, f1, fp, fn, fpr, fnr = classify_metrics(y_true, combined_preds)
        
        print("--- ARCHITECTURAL CAPACITY ---")
        print(f"CH1 ROC-AUC:      {auc_ch1:.4f}")
        print(f"CH2 ROC-AUC:      {auc_ch2:.4f}")
        print(f"Combined ROC-AUC: {auc_comb:.4f} (Approximated)")
        
        print("--- PRODUCTION CLASSIFICATION (Frozen Logic) ---")
        print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
        print(f"FP: {fp} (FPR: {fpr:.4f}) | FN: {fn} (FNR: {fnr:.4f})")
        
        print("--- OFFLINE THRESHOLD SWEEP ---")
        best_f1 = -1
        best_metrics = None
        best_t1, best_t2 = 0, 0
        
        # We know CH1 separates True Labelflip, and CH2 separates Latent Inversion
        # Sweep CH1 threshold from 0.0 to 0.15 (since honest mean is 0.11)
        for t1 in np.linspace(0.00, 0.20, 100):
            # Sweep CH2 threshold from -0.8 to 0.2
            for t2 in np.linspace(-0.8, 0.2, 100):
                sweep_preds = []
                for i in range(len(y_true)):
                    is_c1 = (ch1_scores[i] <= t1)
                    is_c2 = (ch2_scores[i] < t2) if ch2_scores[i] != 0.0 else False
                    sweep_preds.append(1 if (is_c1 or is_c2) else 0)
                
                sp, sr, sf1, sfp, sfn, sfpr, sfnr = classify_metrics(y_true, sweep_preds)
                if sf1 > best_f1 or (sf1 == best_f1 and sfpr < best_metrics[5]):
                    best_f1 = sf1
                    best_metrics = (sp, sr, sf1, sfp, sfn, sfpr, sfnr)
                    best_t1, best_t2 = t1, t2
                    
        sp, sr, sf1, sfp, sfn, sfpr, sfnr = best_metrics
        print(f"Best Thresholds: CH1 <= {best_t1:.4f}, CH2 < {best_t2:.4f}")
        print(f"Precision: {sp:.4f} | Recall: {sr:.4f} | F1: {sf1:.4f}")
        print(f"FP: {sfp} (FPR: {sfpr:.4f}) | FN: {sfn} (FNR: {sfnr:.4f})")
        
    print("\n")
