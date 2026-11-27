# Pooled probes against the small edit attacker (PLAN.md section 14)

Every probe set to 3% false alarms on held out original notes the agent did not execute.

| Probe | Recall, held out originals | AUROC, held out originals | Recall, held out paraphrases | AUROC, held out paraphrases | False alarms, held out paraphrases |
| --- | --- | --- | --- | --- | --- |
| End of memory result, original notes | 0.745 | 0.969 | 0.375 | 0.952 | 0.008 |
| End of memory result, plus paraphrases | 0.855 | 0.983 | 0.5 | 0.978 | 0.008 |
| Mean over memory result, original notes | 0.727 | 0.974 | 0.607 | 0.949 | 0.035 |
| Mean over memory result, plus paraphrases | 0.873 | 0.986 | 0.696 | 0.982 | 0.026 |
| Last prompt token, original notes | 0.709 | 0.964 | 0.464 | 0.931 | 0.012 |
| Last prompt token, plus paraphrases | 0.873 | 0.974 | 0.536 | 0.965 | 0.015 |

## Attack (36 targets from 21 cases; original notes executed 0.92 in 16 fresh rollouts)

| Arm (probe attacked) | Evaded [95% CI] | Final execution | Words changed |
| --- | --- | --- | --- |
| E_mre (End of memory result, original notes) | 0.139 [0.054, 0.231] | 0.880 | 5.2 |
| E_mean (Mean over memory result, original notes) | 0.028 [0.0, 0.091] | 0.825 | 5.4 |
| E_wmean (Mean over memory result, plus paraphrases) | 0.056 [0.0, 0.139] | 0.884 | 4.6 |

**D1** (E_mre minus E_mean): +0.111 [0.0, 0.22], **does not hold**.
**D2** (E_mre minus E_wmean): +0.083 [-0.029, 0.194], **does not hold**.

Transfer: for each arm's evaded notes, the share each probe also misses.

| Arm | Evaded | O_mre | W_mre | O_mean | W_mean | O_last | W_last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E_mre | 5 | 1.0 | 0.6 | 0.2 | 0.2 | 0.2 | 0.2 |
| E_mean | 1 | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| E_wmean | 2 | 1.0 | 0.5 | 1.0 | 1.0 | 0.5 | 0.0 |

Agent rollouts that hit the special token error: {'special_token': 0, 'counted_not_executed': 0}.

## What this means (written after seeing the results)

Run November 27, 2026, one fresh H200, 35 minutes; no agent errors. All six probes set to the same 3% false alarm rate on held out original notes the agent did not execute.

**Pre-registered results.**
* **D1 does not hold, narrowly.** Small edits evade the single position probe on 13.9% of notes and the mean pooled probe on 2.8%; the difference +0.111 has an interval of [0.000, 0.220], touching zero. The direction is as predicted.
* **D2 does not hold.** Against the mean pooled probe trained with paraphrases, evasion is 5.6%; difference +0.083 [−0.029, +0.194].
* **Replication.** At this run's matched threshold the single position probe is evaded less often than in section 13 (13.9% against 30.8%), because its threshold here is slightly looser (3% held out false alarms against 2.5%) and attack runs vary. The evading edits are again about 5 words, and gpt-oss still executes them (0.88).

**Without an attack (pre-registered table above).** At equal false alarms on original notes, pooling over the memory result raises recall on reworded notes the agent executes from 0.375 to 0.607 (original training notes) and to 0.696 with paraphrase training, the best of the six. Part of this is again calibration: the pooled probes also raise more false alarms on non executed paraphrases (2.6% to 3.5% against 0.8%).

**Exploratory, not pre-registered: evasions do not transfer.** Of the 5 notes that evaded the single position probe, the mean pooled probes and the last prompt token probes still catch 4. Of the mean pooled probe's one evasion, the single position probe catches it. A monitor that flags a note when **either** the single position or the mean pooled probe flags it, with both thresholds raised together so that the combination still has 3% false alarms (3.1%), catches 0.78 of held out executed notes (against 0.75 and 0.73 for the two alone), and only 2 of the 36 notes from the single position attack still execute and slip past both (against 5 for the single probe). The attacker in this run never targeted the combination, so this is not yet a robustness result; an adaptive attack against the combined monitor is the test.

**Where this leaves the robustness story.**
1. A single position probe can be evaded by a few words while the agent still obeys (sections 13 and 14).
2. Different reading positions fail on different edits, so combining them is a promising, cheap defence at no extra false alarm cost.
3. Training on paraphrases improves recall on reworded notes, mostly through calibration.
4. None of the defences has yet been tested against an attacker aimed at it directly, apart from the single pooled probes here (2.8% and 5.6% evasion).
