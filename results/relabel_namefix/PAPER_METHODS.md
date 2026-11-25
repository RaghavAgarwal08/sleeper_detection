# Methods adapted from four papers (exploratory, CPU, cached gpt-oss-20b features)

Script `interp/probe_live_inert/paper_methods.py`. 888 poisoned notes, 5 fold held out cases (same folds as PHASE2_ANALYSIS.md B), execution rate as the soft target, 1,000 case resamples for 95% intervals.

## 1. Conditional probing (Hewitt et al. 2021)

Each row is a logistic probe on the listed features. Lower cross entropy is better; a constant predictor scores 0.9019 bits.

| Features | Cross entropy (bits per note) | AUROC within wording | AUROC overall |
| --- | --- | --- | --- |
| word | 0.5425 | 0.416 | 0.908 |
| text | 0.4896 | 0.790 | 0.953 |
| word+text | 0.4945 | 0.761 | 0.946 |
| act | 0.4348 | 0.905 | 0.976 |
| word+act | 0.4279 | 0.904 | 0.975 |
| text+act | 0.4406 | 0.886 | 0.971 |
| word+text+act | 0.4445 | 0.878 | 0.970 |
| word+text+act_note | 0.4614 | 0.850 | 0.964 |

| Information added | Bits per note [95% CI] | AUROC within wording [95% CI] |
| --- | --- | --- |
| word+act over word | +0.1146 [+0.087, +0.148] | +0.487 [+0.403, +0.572] |
| text+act over text | +0.0490 [+0.031, +0.070] | +0.096 [+0.054, +0.148] |
| word+text+act over word+text | +0.0501 [+0.033, +0.071] | +0.117 [+0.065, +0.172] |
| word+text+act_note over word+text | +0.0332 [+0.018, +0.049] | +0.089 [+0.039, +0.143] |
| word+text+act over act | -0.0096 [-0.030, +0.009] | -0.027 [-0.069, +0.014] |

`act` = activations at the end of the memory result (mean of layers 14 to 19); `act_note` = at the last token of the note; `word` = wording one hot; `text` = TF-IDF on the full prompt. A positive gain with an interval above zero means the activations carry information about execution that the baseline does not. The wording only model gives every note of a wording the same score within a fold, so its within wording AUROC (and the +0.49 gain over it) only reflects fold to fold shifts; read its bits, not its AUROC.

## 2. Directions never trained on execution (MacDiarmid et al. 2024)

| Direction | AUROC within wording [95% CI] | AUROC overall |
| --- | --- | --- |
| poisoned_vs_twin_probe | 0.691 [0.612, 0.761] | 0.91 |
| poisoned_minus_clean_means | 0.367 [0.278, 0.458] | 0.672 |
| generic_a_helpful_ai | 0.535 [0.454, 0.612] | 0.382 |
| generic_a_harmless_ai | 0.550 [0.468, 0.631] | 0.613 |
| generic_an_honest_ai | 0.537 [0.460, 0.622] | 0.451 |
| generic_doing_something_dangerous | 0.522 [0.435, 0.613] | 0.453 |
| generic_doing_something_sneaky | 0.515 [0.436, 0.596] | 0.547 |
| generic_trying_to_deceive_the_human | 0.535 [0.451, 0.627] | 0.406 |
| generic_mean_of_six | 0.570 [0.487, 0.650] | 0.444 |
| execution_diff_of_means | 0.559 [0.456, 0.655] |  |
| execution_trained_probe_C0.01 | 0.880 [0.830, 0.917] |  |

1,000 random directions (sign chosen after the fact, so this is a generous control): mean 0.559, 95th percentile 0.643, best 0.714.

## 3. Confidence based deferral (Ashok and May 2025)

The detector answers only when its confidence clears a threshold chosen on half the cases to reach the target accuracy; coverage = share of notes it answers on the other half (200 random splits).

| Detector | Target 90%: coverage / accuracy | Target 95% | Target 98% |
| --- | --- | --- | --- |
| probe_mre | 1.00 / 0.919 | 0.90 / 0.950 | 0.72 / 0.979 |
| text_full | 0.99 / 0.910 | 0.70 / 0.949 | 0.42 / 0.973 |
| probe_plus_text (conditional) | 1.00 / 0.916 | 0.87 / 0.949 | 0.69 / 0.978 |

Within wording 3 only (where wording cannot help), accuracy of the most confident predictions:

| Detector | Top 25% | Top 50% | All |
| --- | --- | --- | --- |
| probe_mre | 1.0 | 0.932 | 0.804 |
| text_full | 0.892 | 0.905 | 0.824 |
| probe_plus_text (conditional) | 1.0 | 0.919 | 0.831 |

## 4. Needs the GPU

* Generic yes/no defection probe (MacDiarmid et al.): `interp/phase3/generic_probe.py`, then rerun this script.
* Position and layer sweeps (Ashok and May) and trajectory geometry (Pandey et al.): `interp/phase3/capture.py` (always saves the sweep positions), then `interp/phase3/sweeps.py`.
* Function vectors (Todd et al.): head level activation patching, a later session.
