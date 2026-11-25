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
| Other model reading the text: qwen2.5-7b-instruct, last token(s) | 0.696 [0.607, 0.772] | 0.243 | +0.192 [+0.122, +0.275] |
| Other model reading the text: qwen2.5-7b-instruct, mean token(s) | 0.830 [0.771, 0.875] | 0.425 | +0.058 [+0.010, +0.114] |
| Other model reading the text: qwen3-0.6b, last token(s) | 0.648 [0.565, 0.725] | 0.243 | +0.240 [+0.161, +0.330] |
| Other model reading the text: qwen3-0.6b, mean token(s) | 0.817 [0.754, 0.864] | 0.374 | +0.071 [+0.012, +0.138] |
| Other model reading the text: qwen3-8b, last token(s) | 0.790 [0.733, 0.839] | 0.407 | +0.098 [+0.052, +0.152] |
| Other model reading the text: qwen3-8b, mean token(s) | 0.844 [0.793, 0.887] | 0.449 | +0.044 [-0.001, +0.094] |

Rows starting "Other model reading the text" use features from runpod/text_baseline_lm.py: another language model reads exactly the text TF-IDF sees, averaged over the same relative depth as gpt-oss's window, at the last token or averaged over all tokens; the same probe is then fitted on those features.

## Does gpt-oss's own state add information beyond another model's reading? (`beyond_other_lm.py`)

Conditional probing with Qwen3-8B's mean pooled reading of the prompt as the baseline, same folds and labels.

| Features | Cross entropy (bits) | AUROC within wording |
| --- | --- | --- |
| Qwen3-8B reading the text | 0.4691 | 0.852 |
| gpt-oss-20b's own state (probe) | 0.4346 | 0.901 |
| Both | 0.4309 | 0.905 |

| Added information | Bits per note [95% CI] | AUROC within wording [95% CI] |
| --- | --- | --- |
| gpt-oss's state beyond Qwen3-8B's reading | +0.038 [+0.024, +0.056] | +0.053 [+0.021, +0.091] |
| Qwen3-8B's reading beyond gpt-oss's state | +0.004 [−0.007, +0.014] | +0.005 [−0.017, +0.027] |

A strong language model reading the same text gets close to the probe (0.844 to 0.852 against 0.888 to 0.901; the direct difference of separately fitted models is +0.044 [−0.001, +0.094]). But conditional probing shows gpt-oss's own state carries information about gpt-oss's own behaviour that Qwen3-8B's reading of the text does not, while the reverse is not true. Most of what the probe reads is a good understanding of the text; a smaller part is specific to the model that acts.
