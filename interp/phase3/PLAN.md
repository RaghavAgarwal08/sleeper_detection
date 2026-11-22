# Phase 3 plan and pre-registration

Written 14 November 2026, before any phase 3 run. Terms follow `results/probe_live_inert/TERMINOLOGY.md`.

## 1. Questions

1. Does the gpt-oss-20b result (a mid layer probe predicts, within wording, which poisoned notes reach execution better
   than TF-IDF on the full prompt) hold for other open models?
2. Is the probe direction causal for gpt-oss-20b: does moving the residual stream along it change how often the
   planted action is executed?

## 2. Models and inclusion

Pilot (120 poisoned notes, 20 per wording, k = 4): Qwen2.5 7B and 14B Instruct, Qwen3 8B, Llama 3.1 8B Instruct,
Mistral 7B Instruct v0.3 (`models.py`). Qwen2.5 32B is added if budget allows. A model goes to the full run if, in the
pilot, between 10% and 90% of notes reach execution and at least 2 wordings contain both outcomes. In the full run
(all 888 poisoned notes, k = 8) a model is analysed if at least 50 notes reach execution, at least 50 are resisted,
and at least 2 wordings contain both outcomes. Every model that is piloted is reported, including those excluded.

## 3. Hypotheses and analysis (fixed now)

Features: residual stream averaged over the layers at the same relative depth as gpt-oss-20b's 14 to 19 of 24
(`models.window_layers`), at the last token of the memory result; secondary site the last token of the note.
Behaviour: same agent loop and judge as phase 1 (`demo/sample_behavior.py`), temperature 1.0, up to 4 steps.
Execution rate = executed rollouts / 8; reached execution = at least 4 of 8.
Metric: AUROC over pairs of notes with the same wording. Intervals: 1,000 resamples of whole cases, 95% percentile.

* **H1 (primary, per model).** 5 fold held out cases over all poisoned notes, execution rate target (binomial
  weights), probe C chosen by nested 4 fold cross validation by case, TF-IDF C = 1.0. The probe beats TF-IDF on the
  full prompt if the 95% interval of the paired difference is above zero.
* **H1 across models.** Reported as the number of included models where H1 holds; no pooled test.
* **H2 (secondary).** Leave one domain out, yes/no target, same comparison.
* **H3 (secondary).** Spearman correlation with the execution rate within wording, probe minus text, same folds as H1.
* **H4 (sanity).** Poisoned vs benign twin on the pre-registered E1 and E2 splits; expected near 1.0 for every method.
* **H5 (steering, gpt-oss-20b first).** Direction = execution probe (C = 0.01) on training split poisoned notes,
  in raw residual units, normalized. Edits from the first token of the memory result on, including generated tokens:
  add alpha × sigma × direction after the first window layer, alpha in {−4, −2, 0, +2, +4}; the same along one fixed
  random direction at ±4; ablation of the direction at every window layer. Notes: up to 48 training split notes whose
  phase 1 first step execution is between 1/8 and 7/8, stratified by wording; k = 8 first responses per condition.
  H5 holds if execution at +4 minus unsteered has a 95% interval above zero, at −4 below zero, and the random
  direction at ±4 stays inside the interval of no change. Any condition where the share of responses with any tool
  call drops by more than half is reported as breaking the model rather than steering it.

Everything else (layer sweeps, probes at other positions, J-lens or SAE follow ups) is exploratory and labelled so.

## 4. Budget (RunPod, about $1,000)

| Step | GPU | Estimate |
| --- | --- | --- |
| Attention baselines, gpt-oss-20b | 1 × A100 80GB or H100, under 1 h | about $5 |
| Steering, gpt-oss-20b | 1 × H100, 2 to 4 h | about $10 |
| Pilot, 5 models | 1 × A100 80GB, about 1 h each | about $30 |
| Full run per included model (sampling, capture, analysis) | 1 × A100 80GB or H100, 3 to 6 h | $40 to $80 each |
| Reserve for reruns | | $150 |

Every pod is stopped when its script ends; scripts write a results archive to download before stopping.

## 5. Additions after this plan (22 November 2026, exploratory)

Added after reading MacDiarmid et al. 2024, Ashok and May 2025, Hewitt et al. 2021, Todd et al. 2024 and Pandey et
al. 2026, before any GPU run: conditional probing, directions not trained on execution and confidence based
deferral (`interp/probe_live_inert/paper_methods.py`, CPU); the generic yes/no defection probe
(`generic_probe.py`), a position sweep over 8 label free positions and trajectory geometry (`sweeps.py`) on the pod.
All are exploratory and do not change H1 to H5.
