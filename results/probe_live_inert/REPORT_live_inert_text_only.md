# Live vs inert probe on behavior_v1

Created 2026-10-17T17:28:25+00:00. Task: live (1) vs inert (0) memory note. Window: resid layers 14 to 19 (pre-registered).

Set sizes (positives): train 1024 (512), E1_new_templates 512 (256), E2_new_domains 160 (80), E3_new_templates_and_domains 80 (40), none_train 128 (0), none_test 20 (0)

| Method | Eval set | AUROC [95% CI] | TPR at 5% FPR | Probe minus text [95% CI] |
| --- | --- | --- | --- | --- |
| text_tfidf | E1_new_templates | 1.000 [0.999, 1.000] | 1.000 |  |
| text_tfidf | E2_new_domains | 1.000 [1.000, 1.000] | 1.000 |  |
| text_tfidf | E3_new_templates_and_domains | 1.000 [1.000, 1.000] | 1.000 |  |

E3 has only 80 notes from 9 test domains, so its CI is wide. TPR at 1% FPR is in report.json but is not meaningful below about 100 negatives.
