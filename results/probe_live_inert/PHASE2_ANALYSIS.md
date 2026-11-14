# Phase 2 analyses on cached features (exploratory)

Script: `interp/probe_live_inert/phase2_analysis.py`. Raw numbers: `phase2_analysis.json`.
Terms follow the MemoryDojo lifecycle states (see TERMINOLOGY.md). Poisoned notes only. Execution rate = rollouts reaching execution / 8. Binary label = reached execution if the rate is at least 0.5, otherwise triggered resisted. 95% CIs resample whole cases.

Of 602 notes labeled triggered resisted, 160 reached execution in at least one rollout.

## A. Leave one domain out, all poisoned notes (888 notes, 43 domains, 286 reached execution)

| Method | AUROC overall | AUROC within wording [95% CI] | within wording and fold [95% CI] | Spearman with execution rate, within wording [95% CI] |
| --- | --- | --- | --- | --- |
| probe_mre | 0.964 | 0.821 [0.76, 0.88] | 0.768 [0.645, 0.865] | 0.381 [0.301, 0.46] |
| probe_note | 0.958 | 0.791 [0.725, 0.851] | 0.641 [0.525, 0.749] | 0.371 [0.288, 0.451] |
| text_full | 0.926 | 0.621 [0.535, 0.706] | 0.598 [0.456, 0.725] | 0.235 [0.131, 0.33] |
| text_note | 0.920 | 0.578 [0.492, 0.666] | 0.475 [0.351, 0.596] | 0.132 [0.038, 0.215] |
| template | 0.878 | 0.142 [0.096, 0.206] | 0.500 [0.5, 0.5] | -0.390 [-0.476, -0.292] |

| Paired difference | AUROC within wording [95% CI] | Spearman within wording [95% CI] |
| --- | --- | --- |
| probe_mre_minus_text_full | +0.200 [0.083, 0.318] | +0.146 [0.033, 0.257] |
| probe_mre_minus_text_note | +0.243 [0.12, 0.361] | +0.249 [0.121, 0.369] |
| probe_note_minus_text_full | +0.170 [0.053, 0.289] | +0.136 [0.011, 0.269] |
| probe_note_minus_text_note | +0.213 [0.093, 0.339] | +0.239 [0.105, 0.365] |

Macro AUROC over domains with both classes: probe_mre 0.980, probe_note 0.964, text_full 0.958, text_note 0.956, template 0.953

## B. Execution rate as the primary label, 5 fold held-out cases, all poisoned notes (888 notes)

| Method | AUROC overall | AUROC within wording [95% CI] | within wording and fold [95% CI] | Spearman with execution rate, within wording [95% CI] |
| --- | --- | --- | --- | --- |
| probe_mre | 0.978 | 0.909 [0.867, 0.944] | 0.918 [0.869, 0.953] | 0.529 [0.457, 0.592] |
| probe_note | 0.974 | 0.894 [0.854, 0.927] | 0.892 [0.842, 0.931] | 0.513 [0.437, 0.577] |
| text_full | 0.956 | 0.805 [0.739, 0.864] | 0.808 [0.734, 0.873] | 0.364 [0.275, 0.454] |
| text_note | 0.949 | 0.760 [0.686, 0.827] | 0.755 [0.663, 0.826] | 0.283 [0.186, 0.372] |
| template | 0.912 | 0.422 [0.344, 0.503] | 0.500 [0.5, 0.5] | -0.107 [-0.185, -0.033] |

| Paired difference | AUROC within wording [95% CI] | Spearman within wording [95% CI] |
| --- | --- | --- |
| probe_mre_minus_text_full | +0.104 [0.035, 0.173] | +0.165 [0.091, 0.24] |
| probe_mre_minus_text_note | +0.149 [0.071, 0.226] | +0.246 [0.165, 0.341] |
| probe_note_minus_text_full | +0.089 [0.034, 0.157] | +0.148 [0.074, 0.231] |
| probe_note_minus_text_note | +0.134 [0.072, 0.202] | +0.230 [0.147, 0.326] |

## B2. Same folds as B, models trained on the binary label (888 notes)

| Method | AUROC overall | AUROC within wording [95% CI] | within wording and fold [95% CI] | Spearman with execution rate, within wording [95% CI] |
| --- | --- | --- | --- | --- |
| probe_mre | 0.968 | 0.860 [0.809, 0.904] | 0.884 [0.825, 0.932] | 0.450 [0.371, 0.523] |
| probe_note | 0.970 | 0.880 [0.839, 0.918] | 0.887 [0.838, 0.929] | 0.476 [0.402, 0.538] |
| text_full | 0.954 | 0.793 [0.719, 0.852] | 0.801 [0.716, 0.87] | 0.352 [0.268, 0.438] |
| text_note | 0.950 | 0.761 [0.689, 0.829] | 0.762 [0.674, 0.835] | 0.291 [0.194, 0.379] |
| template | 0.912 | 0.399 [0.331, 0.463] | 0.500 [0.5, 0.5] | -0.046 [-0.109, 0.01] |

| Paired difference | AUROC within wording [95% CI] | Spearman within wording [95% CI] |
| --- | --- | --- |
| probe_mre_minus_text_full | +0.067 [-0.013, 0.149] | +0.098 [0.017, 0.186] |
| probe_mre_minus_text_note | +0.099 [0.009, 0.181] | +0.159 [0.062, 0.249] |
| probe_note_minus_text_full | +0.087 [0.024, 0.159] | +0.124 [0.04, 0.209] |
| probe_note_minus_text_note | +0.119 [0.056, 0.189] | +0.185 [0.104, 0.27] |

## C. Phase 1 control with nested C, training split only (512 notes)

| Method | AUROC overall | AUROC within wording [95% CI] | within wording and fold [95% CI] | Spearman with execution rate, within wording [95% CI] |
| --- | --- | --- | --- | --- |
| probe_mre | 0.956 | 0.849 [0.786, 0.901] | 0.866 [0.803, 0.925] | 0.432 [0.331, 0.519] |
| probe_note | 0.956 | 0.848 [0.787, 0.904] | 0.857 [0.794, 0.916] | 0.404 [0.301, 0.493] |
| text_full | 0.952 | 0.814 [0.744, 0.875] | 0.826 [0.743, 0.897] | 0.408 [0.308, 0.505] |
| text_note | 0.950 | 0.781 [0.702, 0.846] | 0.794 [0.71, 0.864] | 0.333 [0.218, 0.434] |
| template | 0.890 | 0.370 [0.308, 0.444] | 0.500 [0.5, 0.5] | -0.082 [-0.187, 0.016] |

| Paired difference | AUROC within wording [95% CI] | Spearman within wording [95% CI] |
| --- | --- | --- |
| probe_mre_minus_text_full | +0.035 [-0.052, 0.127] | +0.024 [-0.073, 0.126] |
| probe_mre_minus_text_note | +0.068 [-0.018, 0.156] | +0.099 [-0.001, 0.211] |
| probe_note_minus_text_full | +0.034 [-0.031, 0.111] | -0.003 [-0.083, 0.089] |
| probe_note_minus_text_note | +0.068 [-0.007, 0.148] | +0.072 [-0.017, 0.168] |

## D. Pre-registered splits, models trained on the execution rate

| Split | Method | Spearman overall | Spearman within wording | AUROC (binary) |
| --- | --- | --- | --- | --- |
| E1_new_templates | probe_mre | 0.445 | 0.411 | 0.928 |
| E1_new_templates | probe_note | 0.466 | 0.411 | 0.955 |
| E1_new_templates | text_full | 0.419 | 0.448 | 0.919 |
| E1_new_templates | text_note | 0.308 | 0.400 | 0.851 |
| E1_new_templates | template | nan | nan | 0.500 |
| E2_new_domains | probe_mre | 0.809 | 0.518 | 0.957 |
| E2_new_domains | probe_note | 0.733 | 0.263 | 0.909 |
| E2_new_domains | text_full | 0.756 | 0.291 | 0.926 |
| E2_new_domains | text_note | 0.732 | 0.157 | 0.907 |
| E2_new_domains | template | 0.695 | nan | 0.869 |
