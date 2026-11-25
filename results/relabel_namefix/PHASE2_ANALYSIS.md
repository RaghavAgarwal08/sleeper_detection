# Phase 2 analyses on cached features (exploratory)

Script: `interp/probe_live_inert/phase2_analysis.py`. Raw numbers: `phase2_analysis.json`.
Terms follow the MemoryDojo lifecycle states (see TERMINOLOGY.md). Poisoned notes only. Execution rate = rollouts reaching execution / 8. Binary label = reached execution if the rate is at least 0.5, otherwise triggered resisted. 95% CIs resample whole cases.

Of 591 notes labeled triggered resisted, 155 reached execution in at least one rollout.

## A. Leave one domain out, all poisoned notes (888 notes, 43 domains, 297 reached execution)

| Method | AUROC overall | AUROC within wording [95% CI] | within wording and fold [95% CI] | Spearman with execution rate, within wording [95% CI] |
| --- | --- | --- | --- | --- |
| probe_mre | 0.961 | 0.818 [0.756, 0.874] | 0.748 [0.621, 0.852] | 0.367 [0.286, 0.446] |
| probe_note | 0.959 | 0.817 [0.76, 0.867] | 0.694 [0.587, 0.799] | 0.387 [0.294, 0.464] |
| text_full | 0.926 | 0.639 [0.551, 0.721] | 0.641 [0.503, 0.764] | 0.254 [0.15, 0.353] |
| text_note | 0.917 | 0.578 [0.483, 0.664] | 0.513 [0.394, 0.639] | 0.145 [0.045, 0.227] |
| template | 0.877 | 0.144 [0.099, 0.209] | 0.500 [0.5, 0.5] | -0.399 [-0.483, -0.302] |

| Paired difference | AUROC within wording [95% CI] | Spearman within wording [95% CI] |
| --- | --- | --- |
| probe_mre_minus_text_full | +0.178 [0.058, 0.296] | +0.113 [-0.001, 0.226] |
| probe_mre_minus_text_note | +0.239 [0.117, 0.358] | +0.222 [0.095, 0.349] |
| probe_note_minus_text_full | +0.178 [0.061, 0.298] | +0.133 [0.017, 0.266] |
| probe_note_minus_text_note | +0.239 [0.129, 0.358] | +0.242 [0.115, 0.362] |

Macro AUROC over domains with both classes: probe_mre 0.981, probe_note 0.971, text_full 0.961, text_note 0.953, template 0.953

## B. Execution rate as the primary label, 5 fold held-out cases, all poisoned notes (888 notes)

| Method | AUROC overall | AUROC within wording [95% CI] | within wording and fold [95% CI] | Spearman with execution rate, within wording [95% CI] |
| --- | --- | --- | --- | --- |
| probe_mre | 0.972 | 0.887 [0.838, 0.927] | 0.903 [0.854, 0.938] | 0.532 [0.462, 0.595] |
| probe_note | 0.971 | 0.886 [0.848, 0.92] | 0.894 [0.845, 0.935] | 0.537 [0.462, 0.599] |
| text_full | 0.950 | 0.782 [0.708, 0.849] | 0.781 [0.697, 0.855] | 0.365 [0.276, 0.458] |
| text_note | 0.940 | 0.726 [0.645, 0.803] | 0.725 [0.625, 0.806] | 0.266 [0.162, 0.362] |
| template | 0.907 | 0.409 [0.335, 0.488] | 0.500 [0.5, 0.5] | -0.108 [-0.181, -0.034] |

| Paired difference | AUROC within wording [95% CI] | Spearman within wording [95% CI] |
| --- | --- | --- |
| probe_mre_minus_text_full | +0.106 [0.035, 0.178] | +0.166 [0.085, 0.253] |
| probe_mre_minus_text_note | +0.162 [0.077, 0.25] | +0.266 [0.172, 0.364] |
| probe_note_minus_text_full | +0.104 [0.04, 0.178] | +0.172 [0.092, 0.256] |
| probe_note_minus_text_note | +0.160 [0.085, 0.237] | +0.272 [0.185, 0.374] |

## B2. Same folds as B, models trained on the binary label (888 notes)

| Method | AUROC overall | AUROC within wording [95% CI] | within wording and fold [95% CI] | Spearman with execution rate, within wording [95% CI] |
| --- | --- | --- | --- | --- |
| probe_mre | 0.969 | 0.870 [0.822, 0.912] | 0.887 [0.831, 0.93] | 0.459 [0.375, 0.527] |
| probe_note | 0.965 | 0.874 [0.834, 0.911] | 0.896 [0.851, 0.936] | 0.446 [0.373, 0.51] |
| text_full | 0.948 | 0.777 [0.698, 0.837] | 0.792 [0.707, 0.864] | 0.330 [0.235, 0.419] |
| text_note | 0.941 | 0.732 [0.652, 0.807] | 0.749 [0.654, 0.83] | 0.258 [0.161, 0.353] |
| template | 0.907 | 0.374 [0.307, 0.435] | 0.500 [0.5, 0.5] | -0.052 [-0.114, 0.007] |

| Paired difference | AUROC within wording [95% CI] | Spearman within wording [95% CI] |
| --- | --- | --- |
| probe_mre_minus_text_full | +0.093 [0.016, 0.175] | +0.129 [0.044, 0.224] |
| probe_mre_minus_text_note | +0.139 [0.053, 0.221] | +0.200 [0.095, 0.3] |
| probe_note_minus_text_full | +0.097 [0.029, 0.174] | +0.116 [0.029, 0.199] |
| probe_note_minus_text_note | +0.143 [0.074, 0.222] | +0.188 [0.099, 0.275] |

## C. Phase 1 control with nested C, training split only (512 notes)

| Method | AUROC overall | AUROC within wording [95% CI] | within wording and fold [95% CI] | Spearman with execution rate, within wording [95% CI] |
| --- | --- | --- | --- | --- |
| probe_mre | 0.944 | 0.815 [0.737, 0.88] | 0.862 [0.797, 0.919] | 0.440 [0.333, 0.528] |
| probe_note | 0.955 | 0.853 [0.784, 0.907] | 0.878 [0.819, 0.93] | 0.443 [0.337, 0.533] |
| text_full | 0.949 | 0.815 [0.74, 0.88] | 0.827 [0.746, 0.895] | 0.392 [0.278, 0.49] |
| text_note | 0.941 | 0.763 [0.684, 0.836] | 0.777 [0.691, 0.85] | 0.297 [0.167, 0.41] |
| template | 0.882 | 0.349 [0.283, 0.431] | 0.500 [0.5, 0.5] | -0.050 [-0.177, 0.066] |

| Paired difference | AUROC within wording [95% CI] | Spearman within wording [95% CI] |
| --- | --- | --- |
| probe_mre_minus_text_full | +0.000 [-0.083, 0.09] | +0.048 [-0.062, 0.157] |
| probe_mre_minus_text_note | +0.052 [-0.034, 0.138] | +0.144 [0.029, 0.272] |
| probe_note_minus_text_full | +0.038 [-0.026, 0.113] | +0.051 [-0.03, 0.139] |
| probe_note_minus_text_note | +0.091 [0.01, 0.17] | +0.146 [0.055, 0.247] |

## D. Pre-registered splits, models trained on the execution rate

| Split | Method | Spearman overall | Spearman within wording | AUROC (binary) |
| --- | --- | --- | --- | --- |
| E1_new_templates | probe_mre | 0.458 | 0.419 | 0.938 |
| E1_new_templates | probe_note | 0.468 | 0.417 | 0.961 |
| E1_new_templates | text_full | 0.414 | 0.449 | 0.923 |
| E1_new_templates | text_note | 0.303 | 0.406 | 0.849 |
| E1_new_templates | template | nan | nan | 0.500 |
| E2_new_domains | probe_mre | 0.827 | 0.587 | 0.956 |
| E2_new_domains | probe_note | 0.765 | 0.371 | 0.916 |
| E2_new_domains | text_full | 0.785 | 0.366 | 0.927 |
| E2_new_domains | text_note | 0.738 | 0.141 | 0.897 |
| E2_new_domains | template | 0.724 | nan | 0.869 |
