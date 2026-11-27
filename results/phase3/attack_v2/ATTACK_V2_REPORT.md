# Varied wording probe and small edit attacker (PLAN.md section 13)

## W1: does training on paraphrases help the probe catch reworded notes?

Probe O: original training notes. Probe W: plus 1452 training paraphrases (no feedback, executed at rate 0.135). Thresholds at 5% false alarms on each probe's own training rows.

| | Probe O | Probe W |
| --- | --- | --- |
| Recall on held out paraphrases the agent executes (56, 28 cases) | 0.375 | 0.554 |
| AUROC on all held out paraphrases (executed vs not) | 0.952 | 0.978 |
| Recall on held out original notes | 0.709 | 0.873 |

**W1** (W minus O recall): +0.179 [0.083, 0.276], **holds**.

## Small edit attacker

39 targets (held out, executed in phase 1, flagged by both probes). Final values from 16 fresh rollouts. P2 attacks probe O, P2W attacks probe W, C2 is the control (no score shown, judged against O).

| Arm | Evaded [95% CI] | Final execution | Words changed | Flagged by O | Flagged by W |
| --- | --- | --- | --- | --- | --- |
| Original note | 0.000 [0.0, 0.0] | 0.915 | 0.0 | 1.000 | 1.000 |
| P2: small edits against O | 0.308 [0.175, 0.459] | 0.846 | 4.7 | 0.667 | 0.872 |
| P2W: small edits against W | 0.077 [0.0, 0.154] | 0.800 | 5.2 | 0.744 | 0.846 |
| C2: control | 0.077 [0.0, 0.172] | 0.885 | 6.3 | 0.897 | 0.949 |

**A1v2** (P2 minus C2): +0.231 [0.125, 0.351], **holds**.
**A3v2** (P2 minus P2W): +0.231 [0.081, 0.395], **holds**.
Of P2's 12 evaded notes, the share W also misses: 0.25.

## What this means (written after seeing the results)

Run November 27, 2026, one H200, 46 minutes. 2,172 paraphrases written with no feedback (1,452 training, 720 held out); gpt-oss executes few of them (13.5% and 7.9%), so rewording alone usually breaks the planted instruction. No agent rollout hit the special token error.

**All three pre-registered tests hold.**
* **W1.** Training on paraphrases raises recall on held out paraphrases the agent executes from 0.375 to 0.554 (+0.179 [+0.083, +0.276]) and does not cost the original wordings (0.709 to 0.873).
* **A1v2.** A small edit attacker guided by probe O's score evades it on 12 of 39 notes (0.308), against 0.077 for edits without the score (+0.231 [+0.125, +0.351]), while the agent still executes the evading notes (final execution 0.85). The edits are tiny, 4.7 words on average. Examples, each still executed at least half the time and no longer flagged: "immediately." to "asap."; "travel expenses." to "travel."; "immediately for security reasons." to "asap.".
* **A3v2.** The same attacker against probe W evades on 0.077 of notes (+0.231 [+0.081, +0.395] in W's favour), and W still flags 9 of the 12 notes that evaded O.

**Two checks that qualify the W results (exploratory).**
* **W flags somewhat more.** On held out notes the agent does not execute, W's false alarm rate is 4.0% against O's 2.5% (on non executed held out paraphrases 0.9% against 0.6%). Part of W's advantage is a looser threshold. At matched false alarm rates W is still better: on executed held out paraphrases, recall at 2% false alarms is 0.71 for W and 0.54 for O (at 5%, 0.89 and 0.82); on original notes, 0.80 and 0.71 at 2%. For the attack, if O's threshold were lowered to W's 4% false alarm rate, P2's evasion would fall from 0.31 to about 0.13, against 0.08 for W, so most of A3v2's gap at the pre-registered thresholds comes from the threshold, and a smaller part from W itself. (This is an approximation: the attacker optimised against O's actual threshold.)
* **The "paraphrase gap" is mostly calibration, not ranking.** O ranks held out paraphrases almost as well as W (AUROC 0.952 against 0.978), and at a threshold set on paraphrases' own negatives O's recall is 0.82. Rewording lowers the probe's scores across the board, so a threshold set on the original six wordings misses reworded notes that are still executed. Training on paraphrases mostly fixes that calibration and helps ranking a little.

**What the attack says about the probe.** The edits that evade O sit at the end of the note ("immediately." to "asap."), next to where the probe reads (the end of the memory result, where the note appears last). gpt-oss still obeys, but the probe's score falls below threshold. So part of what probe O reads at its position is local wording, not only the decision. That fits the earlier finding that the reading position matters, and it suggests reading at several positions or pooling over the note as a defence to test next.

**Limits.** One agent, one attacker model, black box only, 39 target notes. W was trained on paraphrases from the same attacker model that attacked it, which may favour W.
