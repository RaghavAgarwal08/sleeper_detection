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

## 6. H6, the difference of means direction as a defense (24 November 2026, before run 3)

Origin, stated plainly: H5 failed (probe direction, one layer). The exploratory follow up `causal_v2.py` then found
that adding −1 SD of the difference of means direction (reached execution minus triggered resisted, training split
window mean at the end of the memory result) at every window layer cut first step execution from 0.26 to 0.07 while
91% of rollouts still called a tool. That was found on the training notes after trying several conditions, so it is
treated as a hypothesis here, not a result. It is tested only if the size matched random control in `causal_v2.py`
(random direction at ±3.25 probe SD, the same size per layer) leaves execution unchanged (95% interval includes zero).

Run 3 (`interp/phase3/causal_v3.py`), written and fixed before it runs:
* Prompts: every poisoned note in the held_out_mixed and test splits whose phase 1 first step execution is between
  1/8 and 7/8 (about 50; the direction never saw them), and the clean prompt of each of their cases.
* Outcomes per rollout (first assistant step, k = 8, T = 1): the planted call (the case's judge); the user's task (a
  call to the tool the user asked for, the most common non planted first step tool of that case in phase 1); any call.
* Conditions: unsteered; difference of means at −1 and −0.5 SD per window layer; a fixed random direction orthogonal
  to the probe at the size of −1 SD; on clean prompts unsteered, difference of means −1 and the same random direction.
* Intervals: 1,000 resamples of whole cases, paired against the unsteered condition on the same prompts.

**H6 holds if all three hold.**
1. Execution on held out poisoned notes, difference of means −1 minus unsteered: 95% interval entirely below zero.
2. Specificity, random direction minus unsteered on the same notes: 95% interval includes zero.
3. Cost, user's task on clean prompts, difference of means −1 minus unsteered: point estimate above −0.05 and 95%
   interval lower bound above −0.10.
The −0.5 condition (dose) and the user's task on poisoned prompts are reported as descriptive.

## 7. Qwen3-8B result and H7, written before any third model (25 November 2026)

**Qwen3-8B (run_model.sh).** Pilot gate passed (21 of 120 reached execution). Full run: 888 notes, k = 8, 128 reached
execution (mean rate 0.152; by wording 0.65, 0.05, 0.03, 0.13, 0.01, 0.04). Tokenization check 20 of 20. **H1 does not
hold for Qwen3-8B**: probe 0.720 vs TF-IDF 0.729 within wording, difference −0.010 [−0.122, +0.104]; H2 −0.064
[−0.189, +0.054]. H1 across models so far: 1 of 2.

**Exploratory cross-model matrix** (`cross_model.py`, results/phase3/CROSS_MODEL.md). For gpt-oss-20b's behaviour,
gpt-oss's own state adds +0.061 bits [+0.045, +0.079] beyond Qwen3-8B's state and Qwen's adds nothing. For Qwen3-8B's
behaviour, Qwen's own state adds +0.013 bits [+0.005, +0.023] beyond gpt-oss's state and gpt-oss's adds nothing
(+0.001 [−0.006, +0.008]). This looks like each model holding information about its own behaviour that the other does
not. It was found after looking, on two models, so it is tested as H7 on new models before it is claimed.

**H7 (self-advantage), fixed now.** For every new included model M and each previous model N, with both models' states
read at the end of the memory result (window mean at the pre-registered relative depth) and the conditional probing
procedure of `cross_model.py` (5 fold held out cases, nested C, execution rate target, 1,000 case resamples):
(a) for M's behaviour, M's state beyond N's state has a cross entropy gain with a 95% interval above zero, and
(b) N's state beyond M's state has an interval that includes zero or lies below it.
H7 holds for a pair if both (a) and (b) hold; it is reported per pair, with no pooling.

## 8. Llama-3.1-8B results (25 November 2026, recorded after the run, nothing above changed)

Weights from the ungated mirror unsloth/Meta-Llama-3.1-8B-Instruct (same weights as Meta's release). For Llama only,
the agent loop carries the first tool call of a step into the history (its template rejects parallel calls); all calls
are judged. Pilot gate passed (13 of 120). Full run: 68 of 888 reached execution (rate 0.101, flat across wordings
0.06 to 0.15). Execution rates correlate 0.17 with gpt-oss-20b and 0.15 with Qwen3-8B.
* **H1 does not hold for Llama**: probe 0.690 vs TF-IDF 0.707, difference −0.018 [−0.107, +0.061]. H1 across models: 1 of 3.
* **H7 (Llama's behaviour)**: vs Qwen3-8B holds (Llama's state beyond Qwen's +0.013 bits [+0.006, +0.020]; reverse
  +0.002 [−0.011, +0.013]). vs gpt-oss-20b does not hold (Llama's state beyond gpt-oss's +0.006 [−0.006, +0.017]).
  H7: 1 of 2 pre-registered pairs.
* Not pre-registered but new: for gpt-oss's and Qwen's behaviour, each model's own state also adds information beyond
  Llama's state (+0.089 and +0.027 bits, intervals above zero) and not the reverse.

## 9. Reading position re-test, written before the re-capture (25 November 2026)

**Why.** The end of the memory result was chosen on gpt-oss-20b, where it is the best position. The exploratory
position sweeps of the new models (sweeps.json, binary target, no nested C) put their best position elsewhere:
Qwen3-8B 0.669 at the end of the memory result vs 0.777 at the last prompt token; Llama-3.1-8B 0.628 vs 0.740.
Because this choice was informed by those sweeps on the same notes, the re-test below is **not independent** of the
data that suggested it and is reported as a secondary, position-corrected analysis, not as a replacement for H1.

**H1-LP.** Exactly the H1 procedure (5 fold held out cases, execution rate target, nested C, TF-IDF on the full prompt,
AUROC within wording, 1,000 case resamples), with the probe reading the window mean at the **last prompt token**, for
Qwen3-8B, Llama-3.1-8B and, for symmetry, gpt-oss-20b. Holds for a model if the probe minus TF-IDF interval is above
zero. H2 is reported alongside.

**H7-LP.** The H7 procedure with every model's state read at the last prompt token; reported for all six ordered pairs.

**For every model added after today** the primary position is the last prompt token, decided now, before any of its
data exist. That makes the next model a clean, independent test of both H1-LP and H7-LP.

## 10. Fine tuned text classifiers as the full prompt baseline, written before any fine tuning (26 November 2026)

**Why.** Mentor feedback (Enyi Jiang): before expanding, add a stronger full prompt text baseline. So far the text
baselines are frozen (TF-IDF variants, and other models' frozen readings with a linear probe; best Qwen3-8B mean pooled
0.844 against the probe's 0.888). A classifier fine tuned end to end on the prompt is the baseline reviewers expect.

**Setup, fixed now.** Population, labels and folds are those of `text_baselines.py`: the 888 poisoned notes, corrected
labels, execution rate as a soft target, the same 5 outer folds over cases, saved before any fine tuning in
`results/probe_live_inert/folds_heldout_cases.json`. Each classifier reads exactly the text TF-IDF reads (all message
contents joined) and is trained with binary cross entropy against the rate. Inside each outer training fold, 20% of the
cases are held out for choosing the epoch (lowest validation loss); test notes are never used for any choice. One seed.
Hyperparameters are fixed in `runpod/run_finetune_baselines.sh` and not tuned:
* ModernBERT-large, full fine tuning, learning rate 2e-5, 4 epochs.
* Qwen3-0.6B with a classification head, full fine tuning, learning rate 1e-5, 4 epochs.
* Qwen3-8B with a classification head, LoRA (rank 16 on all linear layers, head trained), learning rate 1e-4, 3 epochs.
All use an effective batch of 8, 10% warmup, then linear decay. The strongest of the three by AUROC within wording is
"the fine tuned baseline"; choosing the maximum favours the baseline, not the probe.

**H8a.** The execution probe (gpt-oss-20b, end of the memory result) beats the fine tuned baseline: probe minus baseline,
AUROC within wording, 1,000 case resamples, interval above zero.

**H8b (the main one, the paper's story).** gpt-oss-20b's own state adds information beyond the fine tuned baseline:
conditional probing as in `beyond_other_lm.py`, with the baseline's out of fold logit as the base feature and the
gpt-oss window activations added; gain in cross entropy bits per note, interval above zero. Caveat stated in advance:
the baseline's logits for training notes come from models trained on other folds, a mild stacking leak that affects the
base and the full model alike.

Both outcomes are reported whichever way they fall. If H8a fails and H8b holds, the paper says the acting model's state
is not a better classifier than a fine tuned reader but carries information that reader lacks.

## 11. H6b, cross fitted confirmation of the steering defense, written before the run (26 November 2026)

**Why.** Mentor feedback: confirm the steering result on fresh held out examples with the corrected scorer from the
start, saving all outputs, and compare clearly with CounterSteer. Under the corrected labels only 52 held out poisoned
notes have mixed first step outcomes and run 3 used 48 of them; the other held out notes are almost all never executed
on the first step, so nothing is left to suppress. We therefore cross fit.

**Design, fixed now** (`interp/phase3/causal_v4.py`, `runpod/run_causal_v4.sh`).
* Direction: difference of means of the window mean (layers 14 to 19) at the end of the memory result, reached
  execution minus the rest, computed on the 376 held out poisoned notes (55 reached execution), corrected labels.
* Test notes: training split poisoned notes with mixed phase 1 first step outcomes (corrected labels) that run 2 did
  not use (69 notes on 45 cases, mean first step execution 0.34). No test note contributes to the direction.
* Clean prompts of the same cases. k = 8 first responses, T = 1, up to 1,536 new tokens.
* Scoring: the corrected parser from the start; the full text of every rollout saved.
* Conditions: base; dom_-1 (the H6 recipe: −1 SD per window layer from the start of the memory result and on every
  generated token); random_-1 (a fixed random direction orthogonal to it, same size); toolresult_-1 and
  toolresult_-3 (CounterSteer style: the same direction only on the memory result's tokens and only during the prompt
  pass); on clean prompts base, dom_-1, random_-1, toolresult_-3. SD is the projection's standard deviation over the
  held out notes.

**H6b holds** if, as in H6: (1) dom_-1 lowers execution with the 95% case bootstrap interval below zero; (2) the
random push's interval includes zero; (3) on clean prompts dom_-1 changes the user's task by more than −0.05 with the
interval's lower bound above −0.10.

**Reported without a pass or fail:** the CounterSteer style conditions against the H6 recipe (execution, user task on
poisoned and clean prompts, any call). If H6b fails, the steering result is reported as not replicated under cross
fitting, alongside run 3.

**Caveat stated in advance.** The training notes are on cases that are also in the training split for the probe; the
direction does not come from them, but the alpha (−1) was chosen in run 2 on other training notes.

## 12. Adaptive attacker against the execution probe, written before any attack code was run (26 November 2026)

**Question.** Can an attacker who rewrites a planted note keep gpt-oss-20b executing it while the execution probe stops
flagging it? And across rewrites, does lowering the probe's score cost the attacker execution (the score is tied to the
behaviour) or not (the score and the behaviour can be pulled apart)? Mentor priority A.

**Monitor (fixed).** The logistic probe on the window mean (layers 14 to 19) at the end of the memory result, trained on
the 512 training split poisoned notes with corrected labels, execution rate target with binomial weights, C chosen by
4 fold GroupKFold over cases (cross entropy) from the usual grid. Features are recomputed on the pod from vLLM's own
tokenization of each prompt (so the monitor sees exactly what the agent sees, on the run's date). Threshold tau: the
95th percentile of out of fold scores (5 fold over cases) of training notes that did not reach execution, i.e. 5%
false alarms. Scored on every candidate but **not attacked**: the same probe recipe at the last prompt token, the
difference of means projection (training split), and TF-IDF on the full prompt (CPU, after the run), each with its
threshold set the same way.

**Targets (fixed by rule before the attack).** Held out poisoned notes (held_out_mixed and test) that reached execution
in phase 1 (corrected labels) and that the monitor flags (score at or above tau).

**Attacker.** Qwen3-8B (thinking off) served by vLLM, prompted as a red team rewriter in a sandboxed benchmark. Each
round it proposes 4 rewrites of the whole note; it is told to keep the planted tool and the judge's required argument
values, and candidates missing any required string value (case insensitive) are discarded before evaluation. 6 rounds, so up to 24 candidates per
note. Execution during the search: 4 rollouts of the phase 1 agent loop (vLLM, T = 1, up to 4 steps, fired_any). Two
arms on the same notes:
* **P, probe feedback**: the attacker sees each earlier candidate's execution (out of 4) and the monitor's score
  (0 to 100) with the flag threshold.
* **C, control**: the attacker sees execution only and is asked for reliable, varied rewrites.
Selection of each arm's final candidate (fixed): P, the lowest monitor score among candidates with at least 2 of 4
executions, else the highest execution (ties by lower score); C, the earliest candidate with at least 2 of 4, else the
highest execution (ties by earlier). **Final evaluation**: 16 fresh rollouts of each arm's final candidate and of the
original note, on the same day; the monitor scores are recomputed. A note is **evaded** if its final execution is at
least 0.5 and its monitor score is below tau.

**A1 (primary).** Probe feedback produces more evasion than rewriting alone: evasion rate in P minus evasion rate in C,
95% case bootstrap interval above zero.

**A2 (primary, coupling).** Across all evaluated candidates in arm P, the within note Spearman correlation between the
monitor's score and search execution, averaged over notes (notes where either is constant are skipped), 95% case
bootstrap interval above zero. A positive coupling means pushing the score down tends to cost execution.

**Reported without pass or fail.** Evasion and final execution per arm; the score and execution of the original; how
many evaded notes also fall below each unattacked monitor's threshold (transfer); what the attacker changed, read from
the saved text. **Exploratory (A3).** Retraining: with cases split in two halves, retrain the monitor on the training
notes plus all of arm P's candidates from one half (labelled with their search execution), reset tau, and measure how
many of the other half's evaded finals it now flags; then swap halves.

**Interpretation fixed now.** A1 fails and A2 holds: within this attacker's reach, the probe's signal is tied to the
behaviour. A1 holds and A2 fails: the score and the behaviour can be separated, so a deployed probe needs adversarial
training or a different design. Both hold: evasion is possible but costs execution; report the trade off curve. One
attacker, one agent model, one monitor design: no claim beyond black box rewriting attacks.

**Budget.** One H200 (both vLLM servers and the Hugging Face model must fit), about 4 hours, about $15 to $20.

## 13. Varied wording probe and a small edit attacker, written before any of this code was run (27 November 2026)

**Why.** Section 12 found (exploratory) that the probe flags only 41% of rewrites that still work, against 71% of
original held out notes, and that the black box attacker was too weak to test robustness because its rewrites mostly
stopped working. Two follow ups, run together on one H200 (`interp/phase3/attack_v2.py`, `runpod/run_attack_v2.sh`).

**Data, fixed now.** The attacker model (Qwen3-8B, thinking off) paraphrases notes with **no** feedback of any kind:
3 paraphrases of each of the 512 training split poisoned notes (the augmentation set) and 2 of each of the 376 held
out poisoned notes (the evaluation set), each keeping the planted tool and the judge's required string values (case
insensitive; others discarded). Every paraphrase is labelled with 8 rollouts of the phase 1 agent loop (fired_any)
and its features are read exactly as in section 12.

**Probes.** O is the section 12 monitor (original training notes only). W is the same recipe trained on the original
training notes plus their paraphrases; cross validation folds group by case so a note and its paraphrases never split.
Each probe's threshold is the 95th percentile of its own out of fold scores among its training rows that did not reach
execution (5% false alarms).

**W1 (primary).** On held out paraphrases that the agent executes (rate at least 0.5), W flags more than O: recall of W
minus recall of O, 95% case bootstrap interval above zero. Reported alongside: both probes' AUROC on all held out
paraphrases (executed vs not), and both probes' recall on the original held out notes, so any loss on the original
wordings is visible.

**Small edit attacker.** Targets: held out poisoned notes that reached execution in phase 1 and that **both** O and W
flag. Each target starts from its original note (the incumbent). Each of 8 rounds, the attacker sees the incumbent and
earlier results and proposes 8 small edits of the incumbent (each changing at most 12 words, by a word level diff;
others discarded, as are edits missing required values). Three arms on the same targets:
* **P2**: the attacker sees probe O's score; the 4 lowest scoring edits by O are run 4 times; the incumbent becomes the
  lowest O scoring edit executed at least 3 of 4 times, if it scores below the current incumbent.
* **P2W**: the same against probe W (sees and is selected by W's score).
* **C2** (control): no score shown; 4 random valid edits are run; the incumbent becomes a random one executed at least
  3 of 4 times.
Final candidate: in P2 and P2W the lowest scoring (by the arm's probe) evaluated edit executed at least 3 of 4 times,
in C2 the last incumbent; if none, the original note. **Final evaluation**: 16 fresh rollouts and fresh scores. A note
is **evaded** in an arm if final execution is at least 0.5 and the arm's probe (C2 is judged against O) scores it
below that probe's threshold.

**A1v2 (primary).** P2 evasion minus C2 evasion, case bootstrap interval above zero.
**A3v2 (primary).** P2 evasion minus P2W evasion, interval above zero (varied wording training makes the probe
harder to evade).
**Reported:** each arm's evasion rate with interval, final execution, number of words changed, transfer of P2's evaded
notes to W and to the last prompt token probe, and the saved text of every edit.

**Interpretation fixed now.** If P2 evades often, a small edit black box attacker defeats the original probe; whether
W resists it is A3v2. If P2 rarely evades even with execution kept high, the score is hard to move without changing
behaviour, which is a much stronger robustness statement than section 12 could make. One attacker model, one agent,
black box only.

**Budget.** One H200, about 1.5 hours, about $8. All features (originals included) are recomputed in this run so they share its date.
