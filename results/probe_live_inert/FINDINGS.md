# Linear probes on gpt-oss-20b for planted memory notes

Raghav Agarwal, AI Alignment @ Illinois Fellowship, Interpretability stream. October 2026.
Built on the behavior_v1 dataset and gptoss.py harness in this repo.

## Summary

1. **Live vs inert.** A linear probe on the residual stream separates a pending instruction from a closed record of the
   same action with AUROC 0.998 to 1.000 on held out wordings and domains. A TF-IDF text baseline also scores 1.000, so
   this validates the pipeline but shows nothing beyond the words. The signal reaches the end of the memory tool result
   by layer 1 and is perfectly decodable from about layer 13.
2. **Fired.** Among live notes only, the same kind of probe predicts whether the agent will actually execute the planted
   action at the trigger turn. Holding out whole cases and comparing only notes that share a wording, the probe scores
   AUROC **0.876 [0.818, 0.920]**, against 0.781 for TF-IDF on the note (difference +0.096 [+0.008, +0.185]) and 0.814
   for TF-IDF on the full prompt (difference +0.062 [−0.016, +0.141], not significant).
3. **Wording drives obedience.** The six live note wordings are executed in 1% to 92% of cases. The paper's own
   preamble (template 0) is executed 92% of the time; template 4 only 1%.
4. **Caveat.** The pre-registered primary set E3 has only 2 executed notes out of 40 and cannot be interpreted. All
   controls in section 4 are exploratory, run after the pre-registered report.

## 1. Setup

| Item | Value |
| --- | --- |
| Model | gpt-oss-20b, Hugging Face weights dequantized to bf16 (activations), vLLM 0.30.0 MXFP4 (behaviour) |
| Data | behavior_v1: 148 Sleeper Attack cases × (6 live, 6 inert, 1 no note) = 1,924 trigger turn requests |
| Feature | residual stream averaged over resid layers 14 to 19 (pre-registered in manifest.json) |
| Sites | `memory_result_end` (primary, label free); `entry0.occ0.last`, the last token of the planted note (secondary) |
| Probe | standardise, then L2 logistic regression; C chosen by leave one template out CV on train only |
| Splits | train = templates 0 to 3, train domains; E1 = templates 4 and 5, seen domains; E2 = templates 0 to 3, 9 new domains; E3 = both new |
| Baselines | TF-IDF (word 1 to 2 grams and char 3 to 5 grams) on the note; difference of means; 20 shuffled label probes |
| Uncertainty | 2,000 bootstrap resamples of cases (notes from one case are not independent) |
| Behaviour | 8 samples per live note at T = 1.0, up to 4 agent steps, judged by the case's own success conditions. Fired = executed in at least 4 of 8 |

Prompts are identical between activation capture and behaviour sampling: vLLM was run with
`VLLM_SYSTEM_START_DATE` set to the dataset's build date, and a check request rendered to 962 = 962 tokens.

## 2. Live vs inert (pre-registered)

| Method | E1 new wording | E2 new domains | E3 both new |
| --- | --- | --- | --- |
| TF-IDF note text | 1.000 | 1.000 | 1.000 |
| Probe, end of memory result | 1.000 | 1.000 | 0.998 [0.993, 1.000] |
| Probe, end of note | 1.000 | 1.000 | 1.000 |

Shuffled label probes average 0.48 and 0.53. Requests with no note get a mean live probability of 0.003.
Per layer AUROC at the end of the memory result is 0.50 at layer 0 (the token is identical in both conditions), 0.94 at
layer 1, and 1.00 from layer 13 to 21. Difference of means scores 0.997 at the note but 0.736 at the end of the memory
result: the direction is dominant at the note and becomes one of many after it is carried forward.
Full report: `REPORT_live_inert.md`.

## 3. Fired (pre-registered report)

| Method | E1 new wording (13 of 256 fired) | E2 new domains (39 of 80) | E3 both new (2 of 40) |
| --- | --- | --- | --- |
| TF-IDF note text | 0.809 [0.684, 0.920] | 0.890 [0.797, 0.963] | 0.105 |
| Probe, end of memory result | 0.945 [0.898, 0.980] | 0.927 [0.866, 0.979] | 0.974 |
| Probe, end of note | 0.950 [0.914, 0.978] | 0.911 [0.847, 0.968] | 0.934 |

Fired rate by template: 0 = 0.92, 1 = 0.07, 2 = 0.10, 3 = 0.74, 4 = 0.01, 5 = 0.10. Because the held out wordings
(4 and 5) are rarely executed, E3 has only 2 positives; a shuffled label probe reached 0.97 there, so E3 numbers are
noise. Full report: `REPORT_fired.md`.

## 4. Controls (exploratory)

Two confounds could explain the probe's lead in section 3. **Case recognition**: E1 contains the same cases as
training with new wording, and the probe sees the whole prompt while the baseline sees only the note. **Template**:
wording alone predicts most of the label. `interp/probe_live_inert/controls.py` addresses both.

**Training split, 5 fold CV holding out whole cases, AUROC only over pairs that share a template** (6,562 pairs):

| Method | Overall | Within template [95% CI] |
| --- | --- | --- |
| Probe, end of memory result | 0.965 | **0.876 [0.818, 0.920]** |
| Probe, end of note | 0.963 | 0.869 [0.809, 0.916] |
| TF-IDF, note | 0.950 | 0.781 [0.708, 0.846] |
| TF-IDF, user request plus note | 0.946 | 0.779 [0.698, 0.853] |
| TF-IDF, full prompt | 0.952 | 0.814 [0.747, 0.876] |
| Template only | 0.890 | 0.5 by construction |

Probe minus note text: +0.096 [+0.008, +0.185]. Probe minus full prompt text: +0.062 [−0.016, +0.141].
Within template, the probe's score also tracks the continuous fire rate better (Spearman 0.38 to 0.61 vs 0.35 to 0.51).

**Held out sets with stronger baselines:**

| Method | E1 | E2 |
| --- | --- | --- |
| Probe, end of memory result | 0.946 | 0.927 |
| TF-IDF, full prompt | 0.890 | 0.901 |
| Same case's fired rate in training | 0.862 | n/a |
| Template's fired rate in training | n/a | 0.869 |

Within template on E1, the full prompt baseline ties the probe (0.933 vs 0.917), so part of the E1 lead is case
recognition. Within case on E1, the probe ranks template 5 above template 4 in all 11 cases where they behave
differently, which matches behaviour (10% vs 1%); every text baseline ranks them the other way. This is effectively a
single comparison between two wordings. Details: `CONTROLS_fired.md`, `controls_fired.json`.

## 5. What this does and does not show

It shows that gpt-oss-20b represents whether a memory note is pending, and that its activations before the trigger turn
carry information about whether it will act on the note, at least as much as any text baseline tried and significantly
more than the note's text alone. It does not yet show that the activations beat a strong reader of the full prompt; that
difference is +0.06 and not significant with 148 cases and 6 wordings. All results are on one model and one attack
family, with tool results after the first step canned.

## 6. Next steps

1. Write many more live note wordings (for example 30), so generalization to new wording is tested on many wordings
   rather than 2. This is also what would make E3 usable.
2. Add cases or attack families to narrow the confidence intervals.
3. Pre-register TF-IDF on the full prompt (or a small LLM reading the prompt) as the main baseline.
4. Predict the continuous fire rate instead of a 0.5 threshold, and consider k = 16 to reduce label noise.
5. Test whether the probe direction is causal by steering along it at the trigger turn (hooks exist in gptoss.py).

## Reproduce

See `interp/probe_live_inert/README.md`. Section 4 runs on a laptop in about 2 minutes from the files in this folder.
GPU steps ran on one RunPod A100 SXM 80 GB: activation capture takes about 10 minutes and sampling the 888 live notes
at k = 8 well under an hour.
