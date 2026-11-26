# Cross-model matrix: whose state predicts whose behaviour

888 poisoned notes; reached execution: gpt-oss-20b 297, qwen3-8b 128, llama3.1-8b 68; correlations of execution rates: gpt-oss-20b vs qwen3-8b 0.646, gpt-oss-20b vs llama3.1-8b 0.173, qwen3-8b vs llama3.1-8b 0.152. AUROC within wording, 5 fold held out cases, 95% CI over cases.

| Features | predicts gpt-oss-20b | predicts qwen3-8b | predicts llama3.1-8b |
| --- | --- | --- | --- |
| TF-IDF of the prompt | 0.786 [0.712, 0.851] | 0.727 [0.629, 0.812] | 0.711 [0.646, 0.774] |
| gpt-oss-20b's state | 0.901 [0.860, 0.937] | 0.717 [0.634, 0.787] | 0.671 [0.595, 0.742] |
| Qwen3-8B's state | 0.787 [0.721, 0.841] | 0.737 [0.653, 0.805] | 0.602 [0.514, 0.688] |
| Llama-3.1-8B's state | 0.770 [0.707, 0.821] | 0.661 [0.581, 0.736] | 0.661 [0.552, 0.758] |
| Qwen3-8B reading plain text | 0.852 [0.802, 0.893] | 0.774 [0.703, 0.833] | 0.708 [0.605, 0.804] |

| Conditional probing | Bits per note [95% CI] | AUROC within wording [95% CI] |
| --- | --- | --- |
| gpt-oss-20b: gptoss_state beyond qwen_state | +0.0609 [+0.0450, +0.0790] | +0.112 [+0.076, +0.151] |
| gpt-oss-20b: qwen_state beyond gptoss_state | -0.0012 [-0.0120, +0.0090] | -0.002 [-0.022, +0.017] |
| gpt-oss-20b: gptoss_state beyond llama_state | +0.0891 [+0.0700, +0.1090] | +0.132 [+0.084, +0.180] |
| gpt-oss-20b: llama_state beyond gptoss_state | +0.0049 [-0.0040, +0.0130] | +0.001 [-0.019, +0.020] |
| qwen3-8b: qwen_state beyond gptoss_state | +0.0133 [+0.0050, +0.0230] | +0.042 [-0.017, +0.096] |
| qwen3-8b: gptoss_state beyond qwen_state | +0.0011 [-0.0060, +0.0080] | +0.022 [-0.020, +0.059] |
| qwen3-8b: qwen_state beyond llama_state | +0.0265 [+0.0150, +0.0390] | +0.088 [+0.017, +0.161] |
| qwen3-8b: llama_state beyond qwen_state | +0.0036 [-0.0030, +0.0090] | +0.012 [-0.022, +0.047] |
| llama3.1-8b: llama_state beyond gptoss_state | +0.0062 [-0.0060, +0.0170] | +0.007 [-0.056, +0.060] |
| llama3.1-8b: gptoss_state beyond llama_state | +0.0037 [-0.0120, +0.0180] | +0.017 [-0.046, +0.079] |
| llama3.1-8b: llama_state beyond qwen_state | +0.0131 [+0.0060, +0.0200] | +0.053 [+0.006, +0.096] |
| llama3.1-8b: qwen_state beyond llama_state | +0.0023 [-0.0110, +0.0130] | -0.007 [-0.066, +0.057] |

## H7 (pre-registered in PLAN.md section 7; applies to models added after gpt-oss-20b and Qwen3-8B)

| Pair (behaviour of the first) | (a) own beyond other, interval above zero | (b) other beyond own, not above zero | Holds |
| --- | --- | --- | --- |
| gpt-oss-20b vs qwen3-8b | True | True | True |
| gpt-oss-20b vs llama3.1-8b | True | True | True |
| qwen3-8b vs gpt-oss-20b | True | True | True |
| qwen3-8b vs llama3.1-8b | True | True | True |
| llama3.1-8b vs gpt-oss-20b | False | True | False |
| llama3.1-8b vs qwen3-8b | True | True | True |

## Reading this

* The pre-registered part of H7 is Llama's behaviour (the model added after the hypothesis was written): it holds
  against Qwen3-8B and fails against gpt-oss-20b, where neither model's state adds anything beyond the other.
* Across all six ordered pairs the pattern "own state adds, other's state does not" appears in five. It is strongest
  for gpt-oss-20b, whose probe is also the only one that beats TF-IDF (H1 holds for 1 of 3 models).
* Llama is the hardest case: only 68 notes reached execution, its outcomes barely track the wording, and its execution
  rates correlate only 0.15 to 0.17 with the other two models. Low power is a likely reason for the failed pair.
* The rows marked gptoss_state come from the phase 1 capture; the other states come from run_model.sh. All are the
  window mean at the end of the memory result at the same relative depth.
