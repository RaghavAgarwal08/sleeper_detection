# Stronger text baselines (namefix labels)

Script `interp/probe_live_inert/text_baselines.py`. Same folds as PHASE2_ANALYSIS.md B: 5 fold held out cases, 888 poisoned notes (297 reached execution), execution rate target, AUROC within wording, 1,000 case resamples.

| Method | AUROC within wording [95% CI] | Spearman within wording | Probe minus method [95% CI] |
| --- | --- | --- | --- |
| Probe (activations, end of memory result) | 0.888 [0.845, 0.927] | 0.532 |  |
| TF-IDF, full prompt, C = 1 (pre-registered) | 0.786 [0.712, 0.851] | 0.372 | +0.102 [+0.031, +0.181] |
| TF-IDF, full prompt, nested C (0.1 to 1000) | 0.741 [0.646, 0.816] | 0.314 | +0.147 [+0.068, +0.238] |
| TF-IDF to 256 dims, gradient boosted trees | 0.713 [0.630, 0.783] | 0.302 | +0.175 [+0.099, +0.257] |
| Prompt structure only (8 features) | 0.578 [0.493, 0.662] | 0.093 | +0.310 [+0.219, +0.403] |
| TF-IDF and prompt structure | 0.772 [0.698, 0.838] | 0.362 | +0.116 [+0.048, +0.193] |

A neural text baseline (a small language model reading the same prompt) is in `runpod/text_baseline_lm.py` because Hugging Face downloads are blocked in the environment that produced this file.
