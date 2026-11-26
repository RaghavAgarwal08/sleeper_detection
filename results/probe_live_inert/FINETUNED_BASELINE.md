# Fine tuned text classifiers as the full prompt baseline (PLAN.md section 10)

888 poisoned notes (297 reached execution), corrected labels, the probe's 5 outer folds over cases (`folds_heldout_cases.json`), execution rate target, AUROC within wording, 95% intervals over 1,000 case resamples.

| Method | AUROC within wording [95% CI] | Probe minus method [95% CI] |
| --- | --- | --- |
| Probe on gpt-oss-20b's state (end of memory result) | 0.888 [0.845, 0.927] |  |
| TF-IDF, full prompt (pre-registered) | 0.786 [0.712, 0.851] | +0.102 [+0.031, +0.181] |
| Frozen Qwen3-8B reading the text, mean pooled, linear probe | 0.844 [0.793, 0.887] | +0.044 [-0.001, +0.094] |
| ModernBERT-large, fine tuned | 0.763 [0.688, 0.828] | +0.125 [+0.056, +0.207] |
| Qwen3-0.6B, fine tuned | 0.776 [0.709, 0.835] | +0.112 [+0.048, +0.179] |
| Qwen3-8B, fine tuned with LoRA | 0.766 [0.698, 0.823] | +0.122 [+0.068, +0.184] |

**H8a** (probe beats the strongest fine tuned classifier, Qwen3-0.6B, fine tuned): +0.112 [+0.048, +0.179], **holds**.

| Conditional probing | Bits per note [95% CI] | AUROC within wording [95% CI] |
| --- | --- | --- |
| fine tuned score+gpt-oss state over fine tuned score | +0.0503 [+0.0245, +0.0783] | +0.101 [+0.055, +0.153] |
| fine tuned score+gpt-oss state over gpt-oss state | -0.0143 [-0.0352, +0.0031] | -0.027 [-0.059, +0.003] |

**H8b** (gpt-oss-20b's state adds information beyond the strongest fine tuned classifier): **holds**.

## What this means (written after seeing the results)

* **Both pre-registered tests hold.** The probe beats every fine tuned classifier by +0.11 to +0.13 AUROC within wording, and gpt-oss-20b's state adds +0.050 bits per note [+0.025, +0.078] beyond the best one, while the classifier adds nothing beyond the state (−0.014 [−0.035, +0.003]).
* **But fine tuning did not produce a stronger baseline.** All three fine tuned classifiers (0.763 to 0.776) are slightly below fixed TF-IDF (0.786) and clearly below a frozen Qwen3-8B reading the text with a linear probe (0.844, `TEXT_BASELINES.md`). With about 570 training notes per fold, the classifiers learn which wording a note uses (overall AUROC 0.95, against 0.92 for the wording's mean rate alone) but not the differences between notes of the same wording, which is what AUROC within wording measures. Validation loss usually bottomed out after one or two epochs.
* **So the strongest text baseline is still the frozen Qwen3-8B reading** (probe minus it +0.044 [−0.001, +0.094]; gpt-oss's state beyond it +0.038 bits [+0.024, +0.056], `beyond_other_lm.json`). The paper should report both: the fine tuned classifiers as the pre-registered H8 baseline, and the frozen reading as the strongest text baseline we found.
* Settings were fixed in advance and not tuned (one seed, no retraining on the full training fold after choosing the epoch). More data or tuning could make the fine tuned baseline stronger; that would be a new, separately pre-registered test.
