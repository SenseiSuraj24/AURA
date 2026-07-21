import sys
import re

def patch():
    # 1. benchmark_byzantine.py
    bench_path = "scripts/benchmark_byzantine.py"
    with open(bench_path, "r", encoding="utf-8") as f:
        bm_code = f.read()

    bm_code = bm_code.replace(
        "    z_buffer, n_benign, n_high_mse, _, step16_state = run_two_pass_local_training(\n        ae, attack_head, all_flows,\n        ae_optimizer, head_optimizer,\n        mse_threshold=mse_threshold_high,\n        head_epochs=head_epochs,\n        batch_size=batch_size\n    )\n    \n    assert n_benign > 0 or n_high_mse > 0, \"FATAL: No flows processed in two-pass training\"\n    logger.info(f\"Two-pass: benign={n_benign}, high_mse={n_high_mse}, z_buffer={sum(len(z) for z in z_buffer)}\")\n    \n    # Export Step-16 state for CH1\n    ae_delta = {k: step16_state[k] - global_ae_weights[k]\n                for k in step16_state}",
        "    z_buffer, n_benign, n_high_mse, _, step16_state, step8_state, step32_state, final_state = run_two_pass_local_training(\n        ae, attack_head, all_flows,\n        ae_optimizer, head_optimizer,\n        mse_threshold=mse_threshold_high,\n        head_epochs=head_epochs,\n        batch_size=batch_size\n    )\n    \n    assert n_benign > 0 or n_high_mse > 0, \"FATAL: No flows processed in two-pass training\"\n    logger.info(f\"Two-pass: benign={n_benign}, high_mse={n_high_mse}, z_buffer={sum(len(z) for z in z_buffer)}\")\n    \n    # Export Step-16 state for CH1\n    ae_delta = {k: step16_state[k] - global_ae_weights[k]\n                for k in step16_state}\n    ae_delta_all = {\n        'step8': {k: step8_state[k] - global_ae_weights[k] for k in step8_state},\n        'step16': ae_delta,\n        'step32': {k: step32_state[k] - global_ae_weights[k] for k in step32_state},\n        'final': {k: final_state[k] - global_ae_weights[k] for k in final_state}\n    }"
    )

    bm_code = bm_code.replace(
        "    return ae_delta, head_delta, z_buffer, n_benign, n_high_mse",
        "    return ae_delta, head_delta, z_buffer, n_benign, n_high_mse, ae_delta_all"
    )

    bm_code = bm_code.replace(
        "            c_ae_deltas = []\n            c_head_deltas = []",
        "            c_ae_deltas = []\n            c_ae_deltas_all_steps = []\n            c_head_deltas = []"
    )

    bm_code = bm_code.replace(
        "                if is_byzantine and attack_mode == 'latent_inversion':\n                    ae_delta, head_delta, z_buffer, n_benign, n_attack = _run_latent_inversion_byzantine(\n                        client.model.autoencoder, client.model.attack_head, client.train_data, ae_opt, head_opt,\n                        g_ae_w, g_head_w, mse_threshold_high=cfg.CH2_MSE_SPLIT_THRESHOLD, head_epochs=3\n                    )\n                elif is_byzantine and attack_mode == 'true_labelflip':\n                    ae_delta, head_delta, z_buffer, n_benign, n_attack = _run_true_labelflip_byzantine(\n                        client.model.autoencoder, client.model.attack_head, client.train_data, ae_opt, head_opt,\n                        g_ae_w, g_head_w, mse_threshold_high=cfg.CH2_MSE_SPLIT_THRESHOLD, head_epochs=3\n                    )\n                else:\n                    ae_delta, head_delta, z_buffer, n_benign, n_attack = _run_local_training_dual(\n                        client.model.autoencoder, client.model.attack_head, client.train_data, ae_opt, head_opt,\n                        g_ae_w, g_head_w, mse_threshold_high=cfg.CH2_MSE_SPLIT_THRESHOLD, head_epochs=3\n                    )\n                    \n                c_ae_deltas.append(ae_delta)",
        "                if is_byzantine and attack_mode == 'latent_inversion':\n                    ae_delta, head_delta, z_buffer, n_benign, n_attack, ae_delta_all = _run_latent_inversion_byzantine(\n                        client.model.autoencoder, client.model.attack_head, client.train_data, ae_opt, head_opt,\n                        g_ae_w, g_head_w, mse_threshold_high=cfg.CH2_MSE_SPLIT_THRESHOLD, head_epochs=3\n                    )\n                elif is_byzantine and attack_mode == 'true_labelflip':\n                    ae_delta, head_delta, z_buffer, n_benign, n_attack, ae_delta_all = _run_true_labelflip_byzantine(\n                        client.model.autoencoder, client.model.attack_head, client.train_data, ae_opt, head_opt,\n                        g_ae_w, g_head_w, mse_threshold_high=cfg.CH2_MSE_SPLIT_THRESHOLD, head_epochs=3\n                    )\n                else:\n                    ae_delta, head_delta, z_buffer, n_benign, n_attack, ae_delta_all = _run_local_training_dual(\n                        client.model.autoencoder, client.model.attack_head, client.train_data, ae_opt, head_opt,\n                        g_ae_w, g_head_w, mse_threshold_high=cfg.CH2_MSE_SPLIT_THRESHOLD, head_epochs=3\n                    )\n                    \n                c_ae_deltas.append(ae_delta)\n                c_ae_deltas_all_steps.append(ae_delta_all)"
    )

    bm_code = bm_code.replace(
        "                    \"roles\": roles,",
        "                    \"client_ae_deltas_all_steps\": c_ae_deltas_all_steps,\n                    \"roles\": roles,"
    )

    with open(bench_path, "w", encoding="utf-8") as f:
        f.write(bm_code)

if __name__ == "__main__":
    patch()
