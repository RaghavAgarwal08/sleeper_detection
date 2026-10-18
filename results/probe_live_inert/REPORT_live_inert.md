# Live vs inert probe on behavior_v1

Created 2026-10-17T22:14:30+00:00. Task: live (1) vs inert (0) memory note. Window: resid layers 14 to 19 (pre-registered).

Set sizes (positives): train 1024 (512), E1_new_templates 512 (256), E2_new_domains 160 (80), E3_new_templates_and_domains 80 (40), none_train 128 (0), none_test 20 (0)

| Method | Eval set | AUROC [95% CI] | TPR at 5% FPR | Probe minus text [95% CI] |
| --- | --- | --- | --- | --- |
| text_tfidf | E1_new_templates | 1.000 [0.999, 1.000] | 1.000 |  |
| text_tfidf | E2_new_domains | 1.000 [1.000, 1.000] | 1.000 |  |
| text_tfidf | E3_new_templates_and_domains | 1.000 [1.000, 1.000] | 1.000 |  |
| probe_memory_result_end | E1_new_templates | 1.000 [1.000, 1.000] | 1.000 | +0.000 [+0.000, +0.001] |
| probe_memory_result_end | E2_new_domains | 1.000 [1.000, 1.000] | 1.000 | +0.000 [-0.000, +0.000] |
| probe_memory_result_end | E3_new_templates_and_domains | 0.998 [0.993, 1.000] | 1.000 | -0.002 [-0.008, +0.000] |
| probe_entry0.occ0.last | E1_new_templates | 1.000 [0.999, 1.000] | 1.000 | +0.000 [-0.000, +0.001] |
| probe_entry0.occ0.last | E2_new_domains | 1.000 [1.000, 1.000] | 1.000 | +0.000 [-0.000, +0.000] |
| probe_entry0.occ0.last | E3_new_templates_and_domains | 1.000 [1.000, 1.000] | 1.000 | +0.000 [-0.000, +0.000] |

**probe_memory_result_end controls.** Shuffled label probe on E3, mean AUROC 0.476 (max 0.606). Difference of means on E3_new_templates_and_domains 0.736. Chosen C 1.0.
No note requests, mean live probability: train 0.001, test 0.003.
Exploratory per layer AUROC on E3: L0 0.50, L1 0.94, L2 0.84, L3 0.83, L4 0.84, L5 0.83, L6 0.91, L7 0.90, L8 0.94, L9 0.95, L10 0.98, L11 0.99, L12 0.99, L13 1.00, L14 1.00, L15 1.00, L16 1.00, L17 1.00, L18 1.00, L19 1.00, L20 1.00, L21 1.00, L22 0.99, L23 0.99, L24 0.99

**probe_entry0.occ0.last controls.** Shuffled label probe on E3, mean AUROC 0.525 (max 0.776). Difference of means on E3_new_templates_and_domains 0.997. Chosen C 0.0001.
Exploratory per layer AUROC on E3: L0 0.52, L1 0.66, L2 0.73, L3 0.82, L4 0.86, L5 0.92, L6 0.93, L7 0.94, L8 0.92, L9 0.93, L10 0.95, L11 1.00, L12 1.00, L13 1.00, L14 1.00, L15 1.00, L16 1.00, L17 1.00, L18 1.00, L19 1.00, L20 1.00, L21 0.99, L22 0.99, L23 0.99, L24 0.99

E3 has only 80 notes from 9 test domains, so its CI is wide. TPR at 1% FPR is in report.json but is not meaningful below about 100 negatives.
