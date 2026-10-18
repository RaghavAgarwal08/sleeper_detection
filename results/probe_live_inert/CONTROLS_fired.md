# Fired probe: controls (exploratory, run after the pre-registered report)

Data: behavior_v1 live notes, gpt-oss-20b, k = 8 samples per note, fired = fired_any/8 at least 0.5. Features: mean of resid layers 14 to 19 (pre-registered window). Script: interp/probe_live_inert/controls.py (CPU, uses features_window14_19.npz and behavior.jsonl.gz). Raw numbers: controls_fired.json.

Fired rate by template: 0 = 0.92, 1 = 0.07, 2 = 0.10, 3 = 0.74, 4 = 0.01, 5 = 0.10. Wording alone moves obedience from about 1% to about 92%.

## 1. Held out sets, AUROC

| Method | E1 new wording (256, 13 fired) | E2 new domains (80, 39 fired) |
| --- | --- | --- |
| Probe, end of memory result | **0.946** | **0.927** |
| Probe, last token of the note | 0.940 | 0.905 |
| TF-IDF, note text | 0.809 | 0.890 |
| TF-IDF, user request plus note | 0.835 | 0.880 |
| TF-IDF, full prompt | 0.890 | 0.901 |
| Same case's fire rate in training | 0.862 | n/a |
| Template baseline | n/a | 0.869 |

## 2. Within stratum AUROC (only pairs that share a template or a case)

| Comparison | Probe (end of memory result) | TF-IDF full prompt |
| --- | --- | --- |
| E1, within case (template 4 vs 5 of the same case, 11 pairs) | 1.00 | 0.00 |
| E1, within template (1,519 pairs) | 0.917 [0.829, 0.973] | 0.933 [0.871, 0.983] |
| E2, within template (211 pairs) | 0.820 [0.643, 0.956] | 0.678 [0.402, 0.874] |

## 3. Training set, 5 fold CV holding out whole cases, within template (6,562 pairs)

| Method | Overall AUROC | Within template AUROC [95% CI] |
| --- | --- | --- |
| Probe, end of memory result | 0.965 | **0.876 [0.818, 0.920]** |
| Probe, last token of the note | 0.963 | 0.869 [0.809, 0.916] |
| TF-IDF, note text | 0.950 | 0.781 [0.708, 0.846] |
| TF-IDF, user request plus note | 0.946 | 0.779 [0.698, 0.853] |
| TF-IDF, full prompt | 0.952 | 0.814 [0.747, 0.876] |
| Template baseline | 0.890 | 0.5 by construction |

Probe minus TF-IDF note text, within template: +0.096 [+0.008, +0.185].
Probe minus TF-IDF full prompt, within template: +0.062 [-0.016, +0.141].

## Reading

1. The probe is at least as good as every text baseline in every comparison, and clearly beats the note text baseline once template and case are controlled for.
2. Against the strongest text baseline (TF-IDF on the full prompt), the probe's edge of about +0.06 is suggestive but not significant.
3. On unseen wordings, the probe ranks template 5 above template 4 in all 11 cases where they differ, matching behavior (10% vs 1%); every text baseline ranks them the other way. This is effectively one comparison between two wordings, so it needs more templates to confirm.
4. The pre-registered primary set E3 has only 2 fired notes and cannot be interpreted.

Next: more templates (so wording generalization is tested on many wordings, not 2), more cases, predict the continuous fire rate, and make full-prompt TF-IDF the pre-registered baseline.
