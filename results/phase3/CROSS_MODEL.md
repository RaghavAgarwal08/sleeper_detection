# Cross-model matrix: whose state predicts whose behaviour

888 poisoned notes; reached execution: gpt-oss-20b 297, Qwen3-8B 128; correlation of the two models' execution rates 0.646. AUROC within wording, 5 fold held out cases, 95% CI over cases.

| Features | predicts gpt-oss-20b | predicts Qwen3-8B |
| --- | --- | --- |
| TF-IDF of the prompt | 0.786 [0.712, 0.851] | 0.727 [0.629, 0.812] |
| gpt-oss-20b's own state | 0.901 [0.860, 0.937] | 0.717 [0.634, 0.787] |
| Qwen3-8B's own state | 0.787 [0.721, 0.841] | 0.737 [0.653, 0.805] |
| Qwen3-8B reading plain text | 0.852 [0.802, 0.893] | 0.774 [0.703, 0.833] |

| Conditional probing | Bits per note [95% CI] | AUROC within wording [95% CI] |
| --- | --- | --- |
| gpt-oss-20b: gptoss_state beyond qwen_state | +0.0609 [+0.0450, +0.0790] | +0.112 [+0.076, +0.151] |
| gpt-oss-20b: qwen_state beyond gptoss_state | -0.0012 [-0.0120, +0.0090] | -0.002 [-0.022, +0.017] |
| qwen3-8b: gptoss_state beyond qwen_state | +0.0011 [-0.0060, +0.0080] | +0.022 [-0.020, +0.059] |
| qwen3-8b: qwen_state beyond gptoss_state | +0.0133 [+0.0050, +0.0230] | +0.042 [-0.017, +0.096] |
