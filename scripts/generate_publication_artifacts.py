import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
JSON_PATH = Path("results/dp_epsilon_sweep.json")
PLOTS_DIR = Path("results/figures")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = Path("results/dp_epsilon_sweep.csv")

def generate_plots():
    if not JSON_PATH.exists():
        print(f"Error: {JSON_PATH} not found.")
        return
        
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
        
    csv_rows = []
    for e in data["sweep"]:
        sigma = e["sigma"]
        n_rounds = e["n_rounds"]
        metrics = e["detection_metrics"]
        last_round = e["round_records"][-1]
        eps0 = last_round["per_client_epsilon"][0]["dp_epsilon_ae"]
        
        # Privacy attacks
        dlg_cos = e.get("privacy_attacks", {}).get("gradient_inversion", {}).get("metrics", {}).get("cosine_similarity", -1.0)
        dlg_mse = e.get("privacy_attacks", {}).get("gradient_inversion", {}).get("metrics", {}).get("mse", -1.0)
        mia_auc = e.get("privacy_attacks", {}).get("mia", {}).get("auc", -1.0)
        
        # Fix missing ROC-AUC fallback
        roc_auc = metrics.get("ROC-AUC", 0.0)
        if roc_auc is None:
            roc_auc = 0.0
            
        csv_rows.append({
            "sigma": sigma,
            "epsilon": eps0,
            "noise_multiplier": sigma,
            "CH1_AUC": roc_auc,
            "CH2_AUC": 0.0,
            "Combined_AUC": 0.0,
            "Precision": metrics.get("Precision", 0.0),
            "Recall": metrics.get("Recall", 0.0),
            "F1": metrics.get("F1", 0.0),
            "DLG_MSE": dlg_mse,
            "DLG_Cosine": dlg_cos,
            "MIA_AUC": mia_auc,
            "Runtime": e.get("elapsed_sec", 0.0)
        })
        
    df = pd.DataFrame(csv_rows)
    df = df.sort_values(by="sigma")
    df.to_csv(CSV_PATH, index=False)
    print(f"Generated CSV at {CSV_PATH}")
    
    # 1. Privacy vs Utility (F1 and DLG Cosine on Dual Axis)
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()
    ax1.plot(df['sigma'], df['F1'], 'b-o', label='F1 Score')
    ax2.plot(df['sigma'], df['DLG_Cosine'], 'r-s', label='DLG Cosine Similarity')
    ax1.set_xlabel('DP Noise Multiplier ($\sigma$)')
    ax1.set_ylabel('Utility (F1 Score)', color='b')
    ax2.set_ylabel('Privacy Risk (DLG Cosine Similarity)', color='r')
    ax1.set_title('Privacy vs. Utility Tradeoff')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'privacy_vs_utility.png', dpi=300)
    plt.close()
    
    # 2. Sigma vs Epsilon
    plt.figure(figsize=(8, 6))
    plt.plot(df['sigma'], df['epsilon'], 'k-o')
    plt.xlabel('DP Noise Multiplier ($\sigma$)')
    plt.ylabel('Privacy Budget ($\epsilon$)')
    plt.title('Sigma vs Epsilon (RDP Accountant)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'sigma_vs_epsilon.png', dpi=300)
    plt.close()
    
    # 3. Sigma vs F1
    plt.figure(figsize=(8, 6))
    plt.plot(df['sigma'], df['F1'], 'b-o')
    plt.xlabel('DP Noise Multiplier ($\sigma$)')
    plt.ylabel('F1 Score')
    plt.title('Utility Impact: Sigma vs F1')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'sigma_vs_f1.png', dpi=300)
    plt.close()
    
    # 4. Sigma vs ROC (AUC)
    plt.figure(figsize=(8, 6))
    plt.plot(df['sigma'], df['CH1_AUC'], 'g-o')
    plt.xlabel('DP Noise Multiplier ($\sigma$)')
    plt.ylabel('CH1 AUC')
    plt.title('Utility Impact: Sigma vs CH1 ROC AUC')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'sigma_vs_roc.png', dpi=300)
    plt.close()
    
    # 5. Sigma vs DLG Reconstruction Error (MSE)
    plt.figure(figsize=(8, 6))
    plt.plot(df['sigma'], df['DLG_MSE'], 'r-o')
    plt.xlabel('DP Noise Multiplier ($\sigma$)')
    plt.ylabel('DLG Reconstruction MSE')
    plt.title('Privacy: Sigma vs DLG MSE')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'sigma_vs_dlg_mse.png', dpi=300)
    plt.close()
    
    # 6. Sigma vs DLG Cosine Similarity
    plt.figure(figsize=(8, 6))
    plt.plot(df['sigma'], df['DLG_Cosine'], 'r-s')
    plt.xlabel('DP Noise Multiplier ($\sigma$)')
    plt.ylabel('DLG Cosine Similarity')
    plt.title('Privacy: Sigma vs DLG Cosine Similarity')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'sigma_vs_dlg_cosine.png', dpi=300)
    plt.close()
    
    print("Generated all plots successfully in results/figures/")
    
    # Generate Markdown Table
    md = df.to_markdown(index=False)
    with open("results/dp_evaluation_table.md", "w") as f:
        f.write("# DP Evaluation Table\n\n")
        f.write(md)
        f.write("\n")
    print("Generated markdown table.")

if __name__ == '__main__':
    generate_plots()
