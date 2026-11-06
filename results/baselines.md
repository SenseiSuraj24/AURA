# Wave 1.5 baselines (NF-UNSW-NB15-v3)

Split hash `7648fbc56fe2`, train fraction 0.3, test flows (de-duplicated): 246,537. Thresholds: calibration benign P99.

## Detectors — TTL OFF (primary)

Feature dim 76; dropped: none

| Detector | AUROC | PR-AUC | Recall @ calib FPR 1% | Test FPR |
|---|---|---|---|---|
| isolation_forest | 0.916 | 0.411 | 0.066 | 0.012 |
| lightgbm_supervised | 1.000 | 1.000 | 1.000 | 0.010 |
| ae_central | 0.990 ± 0.001 | 0.876 ± 0.006 | 0.736 ± 0.015 | 0.011 ± 0.001 |
| ae_fedavg | 0.988 ± 0.001 | 0.855 ± 0.005 | 0.708 ± 0.030 | 0.012 ± 0.000 |
| ae_local (hospital, own test flows) | 0.987 ± 0.001 | 0.725 ± 0.023 | 0.744 ± 0.030 | 0.011 ± 0.000 |
| ae_local (bank, own test flows) | — | — | — | 0.011 ± 0.001 |
| ae_local (university, own test flows) | 0.990 ± 0.002 | 0.863 ± 0.017 | 0.754 ± 0.018 | 0.012 ± 0.000 |
| ae_local (isp, own test flows) | 0.967 ± 0.009 | 0.687 ± 0.100 | 0.286 ± 0.286 | 0.012 ± 0.001 |
| ae_local (retail, own test flows) | 0.965 ± 0.003 | 0.864 ± 0.022 | 0.201 ± 0.074 | 0.014 ± 0.001 |

Shortcuts: MAX_TTL rule AUROC 0.9991; top single features in this feature space: {'min_ip_pkt_len': 0.9964, 'server_tcp_flags_psh': 0.9687, 'shortest_flow_pkt': 0.9488, 'dst_to_src_avg_throughput': 0.931, 'src_to_dst_second_bytes': 0.9307, 'out_bytes': 0.8823, 'out_pkts': 0.8815, 'retransmitted_out_pkts': 0.8775}

## Detectors — TTL ON

Feature dim 78; dropped: none

| Detector | AUROC | PR-AUC | Recall @ calib FPR 1% | Test FPR |
|---|---|---|---|---|
| isolation_forest | 0.920 | 0.408 | 0.071 | 0.013 |
| lightgbm_supervised | 1.000 | 1.000 | 1.000 | 0.005 |
| ae_central | 0.994 ± 0.001 | 0.902 ± 0.018 | 0.785 ± 0.007 | 0.011 ± 0.000 |
| ae_fedavg | 0.995 ± 0.001 | 0.921 ± 0.012 | 0.835 ± 0.036 | 0.012 ± 0.001 |
| ae_local (hospital, own test flows) | 0.994 ± 0.001 | 0.845 ± 0.023 | 0.825 ± 0.011 | 0.010 ± 0.001 |
| ae_local (bank, own test flows) | — | — | — | 0.011 ± 0.001 |
| ae_local (university, own test flows) | 0.994 ± 0.002 | 0.914 ± 0.018 | 0.853 ± 0.011 | 0.011 ± 0.000 |
| ae_local (isp, own test flows) | 0.968 ± 0.002 | 0.673 ± 0.019 | 0.119 ± 0.016 | 0.012 ± 0.000 |
| ae_local (retail, own test flows) | 0.970 ± 0.001 | 0.880 ± 0.009 | 0.200 ± 0.032 | 0.014 ± 0.001 |

Shortcuts: MAX_TTL rule AUROC 0.9991; top single features in this feature space: {'min_ttl': 0.9991, 'max_ttl': 0.9991, 'min_ip_pkt_len': 0.9966, 'server_tcp_flags_psh': 0.9687, 'shortest_flow_pkt': 0.9488, 'dst_to_src_avg_throughput': 0.931, 'src_to_dst_second_bytes': 0.9307, 'out_bytes': 0.8823}

## Detectors — TTL OFF + generator shortcuts removed

Feature dim 74; dropped: ['MIN_IP_PKT_LEN', 'SHORTEST_FLOW_PKT']

| Detector | AUROC | PR-AUC | Recall @ calib FPR 1% | Test FPR |
|---|---|---|---|---|
| isolation_forest | 0.908 | 0.386 | 0.076 | 0.013 |
| lightgbm_supervised | 1.000 | 1.000 | 1.000 | 0.012 |
| ae_central | 0.981 ± 0.001 | 0.784 ± 0.014 | 0.576 ± 0.026 | 0.012 ± 0.001 |

Shortcuts: MAX_TTL rule AUROC 0.9991; top single features in this feature space: {'server_tcp_flags_psh': 0.9687, 'dst_to_src_avg_throughput': 0.931, 'src_to_dst_second_bytes': 0.9307, 'out_bytes': 0.8823, 'out_pkts': 0.8815, 'retransmitted_out_pkts': 0.8775, 'retransmitted_out_bytes': 0.8712, 'num_pkts_up_to_128_bytes': 0.8682}

## Zero-day: supervised LightGBM trained without the class (TTL off)

| Held-out class | Test flows | Recall @ calib FPR 1% | AUROC vs benign |
|---|---|---|---|
| Analysis | 124 | 1.000 | 1.0000 |
| Backdoor | 1025 | 1.000 | 1.0000 |
| DoS | 1261 | 1.000 | 1.0000 |
| Exploits | 8098 | 1.000 | 1.0000 |
| Fuzzers | 6465 | 1.000 | 1.0000 |
| Generic | 1280 | 1.000 | 1.0000 |
| Reconnaissance | 2567 | 1.000 | 1.0000 |
| Shellcode | 350 | 1.000 | 1.0000 |
| Worms | 30 | 1.000 | 1.0000 |

## Zero-day: supervised LightGBM trained without the class (TTL off + generator shortcuts removed)

| Held-out class | Test flows | Recall @ calib FPR 1% | AUROC vs benign |
|---|---|---|---|
| Analysis | 124 | 1.000 | 1.0000 |
| Backdoor | 1025 | 1.000 | 1.0000 |
| DoS | 1261 | 1.000 | 1.0000 |
| Exploits | 8098 | 1.000 | 1.0000 |
| Fuzzers | 6465 | 1.000 | 1.0000 |
| Generic | 1280 | 1.000 | 1.0000 |
| Reconnaissance | 2567 | 1.000 | 1.0000 |
| Shellcode | 350 | 1.000 | 1.0000 |
| Worms | 30 | 1.000 | 1.0000 |

## Federation value: AttackHead on AE latents, non-IID class hold-out (TTL off)

Hidden classes per org: {'hospital': ['Analysis', 'Generic', 'Reconnaissance'], 'university': ['Analysis', 'Reconnaissance', 'Worms'], 'isp': ['Exploits', 'Reconnaissance', 'Shellcode'], 'retail': ['Backdoor', 'Fuzzers', 'Generic']}

| Training | Recall on HIDDEN classes | Recall on seen classes | Test FPR |
|---|---|---|---|
| local | 0.132 | 0.209 | 0.011 |
| fedavg | 0.195 | 0.279 | 0.011 |
| central | 0.344 | 0.323 | 0.012 |
