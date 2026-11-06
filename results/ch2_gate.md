# Channel 2 gate, phase A

Frozen AE at G0; 20 rounds, 20 head steps per round, 3 seeds, one attacker per run.
Thresholds: 5th percentile of honest scores on clean runs from a separate seed. Scored on a held-out half of the calibration split.

AE alone: AUROC 0.994, recall 0.743 at 1% FPR, recall 0.961 at 2% FPR.

## Clean federation: head as a detector

| Labels | AUROC | Recall at 1% FPR | Share of AE misses caught | AE + head recall | AE + head FPR |
|---|---|---|---|---|---|
| one_class | 0.355 | 0.000 ± 0.000 | 0.000 | 0.743 | 0.009 |
| pseudo | 0.850 | 0.126 ± 0.142 | 0.020 | 0.749 | 0.018 |
| labelled_1 | 0.866 | 0.001 ± 0.001 | 0.000 | 0.743 | 0.018 |
| labelled_10 | 0.894 | 0.156 ± 0.119 | 0.020 | 0.749 | 0.018 |
| labelled_100 | 0.901 | 0.198 ± 0.141 | 0.034 | 0.752 | 0.018 |

## Against a client relabelling its main class as benign

Lower target recall means more damage. Compare each filter to `none` on the same row group.

| Labels | Scoring | Attack | Target recall | Attacker excluded | Honest excluded |
|---|---|---|---|---|---|
| one_class | none | suppress_class | 0.000 ± 0.000 | 0.00 | 0.000 |
| one_class | none | suppress_blend_50 | 0.000 ± 0.000 | 0.00 | 0.000 |
| one_class | none | suppress_blend_25 | 0.000 ± 0.000 | 0.00 | 0.000 |
| one_class | cosine | suppress_class | 0.000 ± 0.000 | 1.00 | 0.050 |
| one_class | cosine | suppress_blend_50 | 0.000 ± 0.000 | 0.97 | 0.050 |
| one_class | cosine | suppress_blend_25 | 0.000 ± 0.000 | 0.03 | 0.037 |
| one_class | root_loss | suppress_class | 0.000 ± 0.000 | 0.30 | 0.092 |
| one_class | root_loss | suppress_blend_50 | 0.000 ± 0.000 | 0.12 | 0.087 |
| one_class | root_loss | suppress_blend_25 | 0.000 ± 0.000 | 0.00 | 0.046 |
| pseudo | none | suppress_class | 0.082 ± 0.099 | 0.00 | 0.000 |
| pseudo | none | suppress_blend_50 | 0.054 ± 0.058 | 0.00 | 0.000 |
| pseudo | none | suppress_blend_25 | 0.068 ± 0.074 | 0.00 | 0.000 |
| pseudo | cosine | suppress_class | 0.175 ± 0.092 | 0.13 | 0.037 |
| pseudo | cosine | suppress_blend_50 | 0.168 ± 0.103 | 0.10 | 0.025 |
| pseudo | cosine | suppress_blend_25 | 0.183 ± 0.088 | 0.13 | 0.033 |
| pseudo | root_loss | suppress_class | 0.117 ± 0.066 | 0.57 | 0.058 |
| pseudo | root_loss | suppress_blend_50 | 0.085 ± 0.095 | 0.08 | 0.075 |
| pseudo | root_loss | suppress_blend_25 | 0.087 ± 0.096 | 0.00 | 0.083 |
| labelled_1 | none | suppress_class | 0.000 ± 0.000 | 0.00 | 0.000 |
| labelled_1 | none | suppress_blend_50 | 0.000 ± 0.000 | 0.00 | 0.000 |
| labelled_1 | none | suppress_blend_25 | 0.000 ± 0.000 | 0.00 | 0.000 |
| labelled_1 | cosine | suppress_class | 0.117 ± 0.017 | 0.08 | 0.033 |
| labelled_1 | cosine | suppress_blend_50 | 0.109 ± 0.036 | 0.08 | 0.013 |
| labelled_1 | cosine | suppress_blend_25 | 0.104 ± 0.082 | 0.08 | 0.000 |
| labelled_1 | root_loss | suppress_class | 0.000 ± 0.000 | 0.00 | 0.183 |
| labelled_1 | root_loss | suppress_blend_50 | 0.000 ± 0.000 | 0.00 | 0.179 |
| labelled_1 | root_loss | suppress_blend_25 | 0.000 ± 0.000 | 0.00 | 0.179 |
| labelled_10 | none | suppress_class | 0.102 ± 0.081 | 0.00 | 0.000 |
| labelled_10 | none | suppress_blend_50 | 0.099 ± 0.078 | 0.00 | 0.000 |
| labelled_10 | none | suppress_blend_25 | 0.081 ± 0.080 | 0.00 | 0.000 |
| labelled_10 | cosine | suppress_class | 0.145 ± 0.103 | 0.08 | 0.212 |
| labelled_10 | cosine | suppress_blend_50 | 0.150 ± 0.106 | 0.08 | 0.217 |
| labelled_10 | cosine | suppress_blend_25 | 0.157 ± 0.111 | 0.08 | 0.225 |
| labelled_10 | root_loss | suppress_class | 0.122 ± 0.090 | 0.00 | 0.146 |
| labelled_10 | root_loss | suppress_blend_50 | 0.128 ± 0.092 | 0.00 | 0.146 |
| labelled_10 | root_loss | suppress_blend_25 | 0.126 ± 0.092 | 0.00 | 0.146 |
| labelled_100 | none | suppress_class | 0.118 ± 0.086 | 0.00 | 0.000 |
| labelled_100 | none | suppress_blend_50 | 0.127 ± 0.091 | 0.00 | 0.000 |
| labelled_100 | none | suppress_blend_25 | 0.127 ± 0.091 | 0.00 | 0.000 |
| labelled_100 | cosine | suppress_class | 0.148 ± 0.105 | 0.00 | 0.146 |
| labelled_100 | cosine | suppress_blend_50 | 0.156 ± 0.101 | 0.00 | 0.142 |
| labelled_100 | cosine | suppress_blend_25 | 0.152 ± 0.109 | 0.00 | 0.150 |
| labelled_100 | root_loss | suppress_class | 0.174 ± 0.078 | 0.00 | 0.163 |
| labelled_100 | root_loss | suppress_blend_50 | 0.202 ± 0.054 | 0.00 | 0.163 |
| labelled_100 | root_loss | suppress_blend_25 | 0.198 ± 0.060 | 0.00 | 0.158 |

## Other attacks: share of attacker rounds excluded

| Labels | Scoring | Label flip | Random labels | Noise-trained |
|---|---|---|---|---|
| one_class | cosine | 1.00 | 1.00 | 0.87 |
| one_class | root_loss | 0.27 | 0.27 | 0.15 |
| pseudo | cosine | 0.00 | 0.00 | 0.00 |
| pseudo | root_loss | 1.00 | 0.98 | 1.00 |
| labelled_1 | cosine | 0.22 | 0.03 | 0.08 |
| labelled_1 | root_loss | 0.28 | 0.37 | 0.33 |
| labelled_10 | cosine | 0.98 | 0.92 | 1.00 |
| labelled_10 | root_loss | 0.40 | 0.15 | 0.28 |
| labelled_100 | cosine | 0.98 | 0.92 | 1.00 |
| labelled_100 | root_loss | 0.42 | 0.12 | 0.28 |

## Honest exclusion on clean federations

| Labels | Scoring | Honest excluded |
|---|---|---|
| one_class | cosine | 0.027 |
| one_class | root_loss | 0.027 |
| pseudo | cosine | 0.057 |
| pseudo | root_loss | 0.067 |
| labelled_1 | cosine | 0.027 |
| labelled_1 | root_loss | 0.137 |
| labelled_10 | cosine | 0.180 |
| labelled_10 | root_loss | 0.120 |
| labelled_100 | cosine | 0.117 |
| labelled_100 | root_loss | 0.127 |
