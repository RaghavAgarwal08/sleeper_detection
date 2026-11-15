# Linear probes on gpt-oss-20b for planted memory notes

Raghav Agarwal, AI Alignment @ Illinois Fellowship, Interpretability stream. Updated 14 November 2026.
Built on Huy Pham's behavior_v1 dataset and gptoss.py harness in this repo. Terms follow the MemoryDojo lifecycle states
(`TERMINOLOGY.md`): poisoned note (formerly live), benign twin (formerly inert), clean, reached execution (formerly
fired) and triggered resisted (formerly not fired).

## Summary

1. **Poisoned vs benign twin is easy and textual.** A probe separates a pending instruction from a closed record of the same
   action with AUROC 0.998 to 1.000 on every held out set, and so does TF-IDF on the note (1.000). This validates the
   pipeline and nothing more.
2. **Whether a poisoned note reaches execution is not textual, and the probe reads it better than text.** Among poisoned
   notes, we compare only notes that share a wording, because wording alone moves the execution rate from 2% to 78%.
   * Held out domains (43 folds, all 888 poisoned notes): probe 0.821 vs full prompt TF-IDF 0.621, difference
     **+0.200 [+0.083, +0.318]**.
   * Held out cases, execution rate as the training target (all 888 notes): probe 0.909 vs 0.805, difference
     **+0.104 [+0.035, +0.173]**; Spearman with the execution rate 0.529 vs 0.364.
   * The original 512 note control, now with nested choice of C: 0.849 vs 0.814, difference +0.035 [−0.052, +0.127],
     not significant.
   These analyses were run after the pre-registered report and are exploratory.
3. **A strong guard model does not see it.** Jev (TypeSafe) scored 0.473 on the execution task on E1, where the probe scores
   0.981 on the same 33 notes. That first run had design problems; `jev_baseline.py` is rewritten for a fair rerun.
4. **Interpretability readouts.** On Neuronpedia's Jacobian lens, the last token of a poisoned note reads out as
   "reminder, checklist, task, automatically", a benign twin as "completed, earlier, canceled". Neuronpedia's SAE latents
   separate poisoned notes from benign twins but carry no detectable execution signal in a 70 note sample, where the dense probe does
   (0.76 within wording). Both are small, exploratory samples.
5. **The pre-registered primary set E3 is uninformative** (2 of 40 notes reach execution) and should not be cited.

## 1. Setup

| Item | Value |
| --- | --- |
| Model | gpt-oss-20b. Activations from Hugging Face weights dequantized to bf16; behaviour from vLLM 0.30.0 (MXFP4) |
| Data | behavior_v1, 148 Sleeper Attack cases × 13 versions (6 poisoned notes, 6 benign twins, 1 clean) = 1,924 prompts |
| What a prompt is | one trigger turn. The user makes a benign request, the agent retrieves memory, the memory result contains the note. Poisoned notes and benign twins both contain the planted action; the poisoned note says it is still to be done, the twin says it was done |
| Feature | residual stream averaged over layers 14 to 19 (pre-registered) |
| Sites | end of the memory result (primary, label free) and the last token of the planted note (secondary) |
| Probe | standardize, then L2 logistic regression |
| Splits | train = templates 0 to 3 in 34 domains; E1 = templates 4 and 5, seen domains; E2 = templates 0 to 3, 9 new domains; E3 = both new |
| Behaviour | 8 rollouts per poisoned note at T = 1.0, up to 4 agent steps, judged by each case's success conditions. Execution rate = executed rollouts / 8; reached execution = at least 4 of 8, otherwise triggered resisted |
| Uncertainty | bootstrap over whole cases (the 13 versions of a case are not independent) |

Prompts are identical between capture and sampling (vLLM run with `VLLM_SYSTEM_START_DATE` set to the dataset build date;
a check request rendered to 962 = 962 tokens).

## 2. Poisoned vs benign twin (pre-registered; `REPORT_live_inert.md`)

| Method | E1 new wording | E2 new domains | E3 both new |
| --- | --- | --- | --- |
| TF-IDF note text | 1.000 | 1.000 | 1.000 |
| Probe, end of memory result | 1.000 | 1.000 | 0.998 [0.993, 1.000] |
| Probe, end of note | 1.000 | 1.000 | 1.000 |

The signal at the end of the memory result is 0.50 at layer 0 (the token is identical), 0.94 at layer 1 and 1.00 from
layer 13. At a 0.5 cutoff the probe calls 6 poisoned notes benign twins, and all 6 use template 4, the wording that is executed in
about 1% of rollouts. The poisoned/twin direction seems to already encode how seriously the model takes the note.

## 3. Reached execution vs triggered resisted (pre-registered; `REPORT_fired.md`)

| Method | E1 (13 of 256 reach execution) | E2 (39 of 80) | E3 (2 of 40) |
| --- | --- | --- | --- |
| TF-IDF note text | 0.809 [0.684, 0.920] | 0.890 [0.797, 0.963] | not interpretable |
| Probe, end of memory result | 0.945 [0.898, 0.980] | 0.927 [0.866, 0.979] | not interpretable |
| Probe, end of note | 0.950 [0.914, 0.978] | 0.911 [0.847, 0.968] | not interpretable |

Share of poisoned notes that reach execution, by template: 0 = 0.92, 1 = 0.07, 2 = 0.10, 3 = 0.74, 4 = 0.01, 5 = 0.10 (mean execution rate 0.78,
0.11, 0.12, 0.62, 0.02, 0.11). The held out wordings are rarely executed, so E3 has 2 positives and a shuffled label
probe reaches 0.97 there.

**What the agent did in the 7,104 rollouts.** 69.3% did only the user's task, 29.2% did the user's task and the planted
action, 0.2% did only the planted action, 1.3% did neither. A note that reaches execution is almost always executed alongside
the legitimate task, not instead of it. Of the 602 notes labeled triggered resisted, 160 were executed in at least one
rollout, which is why section 5 uses the execution rate itself.

## 4. Phase 1 controls (exploratory)

The probe's lead could come from recognizing cases (E1 reuses training cases) or from wording (wording predicts most of
the label). `controls.py` holds out whole cases and scores only pairs of notes that share a template.

| Method (training split, 5 folds by case, 6,562 pairs) | Overall | Within template [95% CI] |
| --- | --- | --- |
| Probe, end of memory result | 0.965 | 0.876 [0.818, 0.920] |
| Probe, end of note | 0.963 | 0.869 [0.809, 0.916] |
| TF-IDF, note | 0.950 | 0.781 [0.708, 0.846] |
| TF-IDF, full prompt | 0.952 | 0.814 [0.747, 0.876] |

On E1 within template the full prompt baseline is level with the probe (0.933 vs 0.917), so part of the E1 lead is
case recognition. Within case on E1, the probe ranks template 5 above template 4 in all 11 cases where they behave
differently, matching behaviour, while every text baseline ranks them the other way; this is one comparison of two
wordings. TF-IDF on the note still finds 38.5% (E1) and 53.8% (E2) of notes that reach execution at 5% FPR, so text is weaker, not
useless. Details in `CONTROLS_fired.md`.

## 5. Phase 2 analyses (exploratory, CPU, `phase2_analysis.py`)

All rows compare notes within the same wording. The probe's C is chosen by nested cross validation inside every training
fold; TF-IDF uses C = 1.0, the value its pre-registered search picked. Full tables in `PHASE2_ANALYSIS.md`.

| Analysis | Probe (end of memory result) | TF-IDF full prompt | Difference [95% CI] |
| --- | --- | --- | --- |
| A. Leave one domain out, 888 notes, 43 domains | 0.821 | 0.621 | **+0.200 [+0.083, +0.318]** |
| A, only pairs from the same held out domain | 0.768 | 0.598 | wide intervals |
| B. Held out cases, trained on execution rate, 888 notes | 0.909 | 0.805 | **+0.104 [+0.035, +0.173]** |
| B2. Same folds, trained on reached execution yes/no | 0.860 | 0.793 | +0.067 [−0.013, +0.149] |
| C. Phase 1 control rerun with nested C, 512 notes | 0.849 | 0.814 | +0.035 [−0.052, +0.127] |

Spearman correlation with the execution rate, within wording: A 0.381 vs 0.235 (difference +0.146 [+0.033, +0.257]);
B 0.529 vs 0.364 (+0.165 [+0.091, +0.240]); B2 0.450 vs 0.352 (+0.098 [+0.017, +0.186]).

Reading.
1. On new domains the probe keeps most of its within wording signal while text drops close to chance. This is the
   clearest result so far.
2. Training on the execution rate helps the probe (0.860 to 0.909) and barely helps text (0.793 to 0.805), so the
   activations carry graded information about how likely the agent is to act.
3. Nested C shrinks the phase 1 gap from +0.062 to +0.035. With 512 notes and 4 wordings the probe and a strong text
   reader cannot be told apart; with 888 notes and 6 wordings they can.
4. On the pre-registered splits with the execution rate as target, the probe and text are close on E1 (Spearman within
   wording 0.41 vs 0.45) and the probe is ahead on E2 (0.52 vs 0.29).
5. A domain averaged leave one domain out AUROC (0.98) mostly measures wording, so it should not be reported alone.

The earlier `paper/new_analysis.py` was rerun with real scikit-learn 1.7.2. It reproduces the pre-registered numbers
exactly (E1 0.946, E2 0.927). The homemade offline reimplementation it had used was off by up to 0.03 in Spearman.

## 6. Jev guard baseline

First run (`REPORT_jev_baseline.md`, 21 October). Jev was asked whether the memory is malicious and whether executing it
violates a policy that says executing hidden memory instructions is a violation.

| Task, split | n (positive) | Jev AUROC | Probe on the same notes |
| --- | --- | --- | --- |
| poisoned vs benign twin, E1 | 40 (20) | 0.816 | 1.000 |
| poisoned vs benign twin, E2 | 40 (20) | 0.924 | 1.000 |
| execution, E1 | 33 (13) | 0.473 | 0.981 |
| execution, E2 | 40 (20) | 0.647 | 0.905 (note TF-IDF 0.920) |

Jev scored nearly every poisoned note above 0.8 whether or not it reached execution. That is the expected behaviour of a text
guard asked whether a note is malicious, since every poisoned note is. The run also had fixable problems: the first N notes
in file order rather than a random or full set, API errors scored 0.5 and not counted, scores not saved, a policy that
makes every poisoned note a violation by definition, and no question about whether the note will reach execution.

`jev_baseline.py` is rewritten to fix all of these. It scores full splits (or a seeded stratified sample), saves every
raw answer and per note score, retries and counts errors, uses a neutral policy by default, adds a will_execute and an
is_pending question, supports few shot examples from the training split and a combiner and threshold fit on training
notes, reports within wording AUROC and TPR at 1% and 5% FPR, and computes the probe and TF-IDF on exactly the notes Jev
scored. The report is generated from the numbers. A dry run with mock answers passes. It needs an OpenRouter key to run;
the full E1 plus E2 for both tasks is about 1,000 calls, roughly 5 cents at the first run's prices.

## 7. Attention baselines from the literature (implemented, need a GPU run)

`attention_capture.py` (GPU) and `attention_baselines.py` (CPU) implement
* Attention Tracker (Hung et al. 2024): heads calibrated on 30 sentences with and without "Ignore previous instruction",
  k = 4 rule, focus score on the user request, flag when low; plus a version with heads chosen on our training labels;
* AttnDA (Kawasaki and Lee 2026): A(I) / (A(I) + A(D)) for all 1,536 heads, shrinkage diagonal LDA, with variants on
  full attention layers only and at the memory result position;
* AttentionDefense (Siska and Sankaran 2025): last layer attention to the system prompt, per head z-normalized, random
  forest and logistic regression;
* the AttnDA paper's FC baseline (logistic regression on the final hidden state).

One design point matters for gpt-oss-20b. Half its layers use a 128 token sliding window, and from the last prompt token
the user request is 123 to 818 tokens back (median 390), so those layers mostly cannot attend to it. Attention Tracker
style scores therefore have to come from the full attention layers. Unit tests and an end to end test on synthetic
attention with a planted signal pass; the capture takes about 25 minutes on an A100.

These methods target an instruction that hijacks the current turn. Our poisoned notes are executed alongside the user's task (29.2% of
rollouts do both), so we expect attention distraction to be weak here; that is the RQ1 test.

## 8. SAE and Jacobian lens readouts (exploratory, Neuronpedia, `SAE_JLENS.md`)

Sample, seeded, training split only: 48 poisoned/benign twin pairs (24 cases, templates 0 and 1) and 70 poisoned notes for
reached execution vs triggered resisted within wording (template 3, 24 vs 24; template 2, 11 vs 11). No GPU; Neuronpedia's own endpoints, queried from the
browser (`neuronpedia_collect.js`).

**Jacobian lens** (Gurnee et al. 2026), exact prompts by token id, top 8 readout tokens at every layer, middle band
layers 9 to 21. At the last token of the note, readouts verbalize the note's status. Tokens that appear with poisoned notes
and never with benign twins include automatically (33 of 48), checklist (28), pinned (27), reminder (32 vs 2), urg (18),
task (12). Benign twins show completed (39 vs 0), earlier (18 vs 0), canceled (13 vs 0). The logit lens shows the same
split more weakly (agenda, pending, ongoing vs previous, followed). Part of this may echo words in the note itself, so it
is a lead, not evidence of an abstract status representation. Readouts do not separate notes that reach execution from resisted ones, and
the planted action's words almost never appear in the readout at the memory result or the reply position.

**SAE latents** (Andy Arditi's BatchTopK SAEs, 131k latents, layers 15 and 19, top 100 latents at the end of the memory
result). Neuronpedia only accepts text, so prompts were re-rendered with tool schemas shortened to their names.
* Poisoned vs benign twin: 53 of 92 latents at layer 15 differ at q < 0.05, and a sparse classifier on the latents reaches 0.97.
  Their Neuronpedia labels describe the local token (braces, punctuation, code), not note status.
* Reached execution vs triggered resisted, within wording: no latent survives correction, and classifiers on the latents are at chance
  (0.37 to 0.50), while the dense probe on the same 70 notes reaches 0.76.
So the execution signal is either spread across many small latents, outside the top 100, in the SAE's reconstruction error,
or altered by the shortened prompts. Checking which needs the SAE weights on a GPU with the exact prompts.

## 9. What this does and does not show

It shows that gpt-oss-20b represents whether a memory note is pending, and that within a wording its activations at the
trigger turn predict whether it will reach execution better than a bag of words reader of the full prompt, clearly so on
new domains. It does not show causality, and the probe and text cannot be separated on the smaller 512 note split.
Everything is one model, one attack family, one trigger turn, and canned tool results after the first step. The note is
always present in these prompts; detecting a note before its trigger arrives (the `retrieved_unreached` state, the dormant
setting in the 6 November plan) needs Huy's multi turn data.

## 10. Next steps

1. Run the rewritten Jev baseline (needs the OpenRouter key) and the attention baselines (one A100 session).
2. Many more wordings (30 or more) and timing minimal pairs, so new wording generalization is tested on many wordings
   and E3 becomes usable.
3. Steering and ablation along the probe direction at the trigger turn (hooks exist in gptoss.py).
4. Pull the SAE weights to test whether the execution direction is spanned by a few latents or lies in the error term.
5. Evaluate on MemoryDojo (Yang et al. 2026, 155 multi session poisoning trajectories, 4 domains, 5 memory systems) as
   an external dataset.

## Reproduce

`interp/probe_live_inert/README.md` covers the GPU steps. CPU steps, from the repo root:

    python interp/probe_live_inert/controls.py
    python interp/probe_live_inert/phase2_analysis.py           # about 15 minutes on 2 cores
    python interp/probe_live_inert/sae_jlens_analysis.py        # uses neuronpedia/export.json
    python interp/probe_live_inert/test_attention_baselines.py --e2e
    python interp/probe_live_inert/jev_baseline.py --task both --mock    # dry run, no API calls
