# Cross-model matrix: whose state predicts whose behaviour

States read at: last_prompt.

888 poisoned notes; reached execution: gpt-oss-20b 297, qwen3-8b 128, llama3.1-8b 68; correlations of execution rates: gpt-oss-20b vs qwen3-8b 0.646, gpt-oss-20b vs llama3.1-8b 0.173, qwen3-8b vs llama3.1-8b 0.152. AUROC within wording, 5 fold held out cases, 95% CI over cases.

| Features | predicts gpt-oss-20b | predicts qwen3-8b | predicts llama3.1-8b |
| --- | --- | --- | --- |
| TF-IDF of the prompt | 0.786 [0.712, 0.851] | 0.727 [0.629, 0.812] | 0.711 [0.646, 0.774] |
| gpt-oss-20b's state | 0.863 [0.795, 0.912] | 0.624 [0.549, 0.706] | 0.695 [0.624, 0.761] |
| Qwen3-8B's state | 0.834 [0.785, 0.874] | 0.810 [0.739, 0.865] | 0.691 [0.622, 0.756] |
| Llama-3.1-8B's state | 0.717 [0.637, 0.790] | 0.630 [0.540, 0.707] | 0.762 [0.695, 0.823] |

| Conditional probing | Bits per note [95% CI] | AUROC within wording [95% CI] |
| --- | --- | --- |
| gpt-oss-20b: gptoss_state beyond qwen_state | +0.0236 [+0.0070, +0.0370] | +0.022 [-0.011, +0.050] |
| gpt-oss-20b: qwen_state beyond gptoss_state | -0.0021 [-0.0140, +0.0090] | -0.006 [-0.032, +0.017] |
| gpt-oss-20b: gptoss_state beyond llama_state | +0.0987 [+0.0740, +0.1240] | +0.127 [+0.081, +0.173] |
| gpt-oss-20b: llama_state beyond gptoss_state | -0.0130 [-0.0250, -0.0020] | -0.019 [-0.044, +0.005] |
| qwen3-8b: qwen_state beyond gptoss_state | +0.0440 [+0.0270, +0.0640] | +0.166 [+0.071, +0.248] |
| qwen3-8b: gptoss_state beyond qwen_state | -0.0034 [-0.0100, +0.0030] | -0.019 [-0.051, +0.016] |
| qwen3-8b: qwen_state beyond llama_state | +0.0605 [+0.0450, +0.0770] | +0.148 [+0.084, +0.213] |
| qwen3-8b: llama_state beyond qwen_state | -0.0043 [-0.0120, +0.0030] | -0.032 [-0.066, +0.001] |
| llama3.1-8b: llama_state beyond gptoss_state | +0.0194 [+0.0090, +0.0320] | +0.070 [+0.029, +0.112] |
| llama3.1-8b: gptoss_state beyond llama_state | +0.0042 [-0.0060, +0.0150] | +0.002 [-0.035, +0.040] |
| llama3.1-8b: llama_state beyond qwen_state | +0.0184 [+0.0060, +0.0320] | +0.064 [+0.014, +0.118] |
| llama3.1-8b: qwen_state beyond llama_state | +0.0009 [-0.0100, +0.0110] | -0.007 [-0.042, +0.029] |

## H7 (pre-registered in PLAN.md section 7; applies to models added after gpt-oss-20b and Qwen3-8B)

| Pair (behaviour of the first) | (a) own beyond other, interval above zero | (b) other beyond own, not above zero | Holds |
| --- | --- | --- | --- |
| gpt-oss-20b vs qwen3-8b | True | True | True |
| gpt-oss-20b vs llama3.1-8b | True | True | True |
| qwen3-8b vs gpt-oss-20b | True | True | True |
| qwen3-8b vs llama3.1-8b | True | True | True |
| llama3.1-8b vs gpt-oss-20b | True | True | True |
| llama3.1-8b vs qwen3-8b | True | True | True |
