# Fired probe on behavior_v1

Created 2026-10-18T02:20:46+00:00. Task: among live notes: fired at the trigger turn (fired_any/k >= 0.5) vs not. Window: resid layers 14 to 19 (pre-registered).

Set sizes (positives): train 512 (232), E1_new_templates 256 (13), E2_new_domains 80 (39), E3_new_templates_and_domains 40 (2), none_train 128 (0), none_test 20 (0)

| Method | Eval set | AUROC [95% CI] | TPR at 5% FPR | Probe minus text [95% CI] |
| --- | --- | --- | --- | --- |
| text_tfidf | E1_new_templates | 0.809 [0.684, 0.920] | 0.385 |  |
| text_tfidf | E2_new_domains | 0.890 [0.797, 0.963] | 0.538 |  |
| text_tfidf | E3_new_templates_and_domains | 0.105 [0.013, 0.218] | 0.000 |  |
| probe_memory_result_end | E1_new_templates | 0.945 [0.898, 0.980] | 0.615 | +0.136 [+0.021, +0.270] |
| probe_memory_result_end | E2_new_domains | 0.927 [0.866, 0.979] | 0.744 | +0.037 [-0.038, +0.124] |
| probe_memory_result_end | E3_new_templates_and_domains | 0.974 [0.897, 1.000] | 0.500 | +0.868 [+0.705, +0.974] |
| probe_entry0.occ0.last | E1_new_templates | 0.950 [0.914, 0.978] | 0.538 | +0.141 [+0.031, +0.277] |
| probe_entry0.occ0.last | E2_new_domains | 0.911 [0.847, 0.968] | 0.718 | +0.022 [-0.066, +0.119] |
| probe_entry0.occ0.last | E3_new_templates_and_domains | 0.934 [0.833, 1.000] | 0.500 | +0.829 [+0.654, +0.974] |

**probe_memory_result_end controls.** Shuffled label probe on E3, mean AUROC 0.464 (max 0.868). Difference of means on E3_new_templates_and_domains 0.303. Chosen C 0.01.
Exploratory per layer AUROC on E3: L0 0.50, L1 0.34, L2 0.08, L3 0.05, L4 0.11, L5 0.17, L6 0.20, L7 0.43, L8 0.42, L9 0.12, L10 0.22, L11 0.46, L12 0.55, L13 0.76, L14 0.72, L15 0.97, L16 0.96, L17 0.96, L18 0.99, L19 0.96, L20 0.80, L21 0.89, L22 0.74, L23 0.68, L24 0.78

**probe_entry0.occ0.last controls.** Shuffled label probe on E3, mean AUROC 0.489 (max 0.974). Difference of means on E3_new_templates_and_domains 0.947. Chosen C 0.01.
Exploratory per layer AUROC on E3: L0 0.54, L1 0.51, L2 0.07, L3 0.12, L4 0.24, L5 0.28, L6 0.42, L7 0.63, L8 0.54, L9 0.92, L10 0.91, L11 0.87, L12 0.92, L13 0.93, L14 0.93, L15 0.95, L16 0.93, L17 0.95, L18 0.93, L19 0.95, L20 0.96, L21 0.96, L22 0.93, L23 0.92, L24 0.87

E3 has only 40 notes (2 positive) from 9 test domains, so its CI is wide. TPR at 1% FPR is in report.json but is not meaningful below about 100 negatives.
