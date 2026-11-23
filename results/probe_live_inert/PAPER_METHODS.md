# Methods adapted from four papers (exploratory, CPU, cached gpt-oss-20b features)

Script `interp/probe_live_inert/paper_methods.py`. 888 poisoned notes, 5 fold held out cases (same folds as PHASE2_ANALYSIS.md B), execution rate as the soft target, 1,000 case resamples for 95% intervals.

## 1. Conditional probing (Hewitt et al. 2021)

Each row is a logistic probe on the listed features. Lower cross entropy is better; a constant predictor scores 0.8737 bits.

| Features | Cross entropy (bits per note) | AUROC within wording | AUROC overall |
| --- | --- | --- | --- |
| word | 0.5641 | 0.422 | 0.912 |
| text | 0.5203 | 0.811 | 0.958 |
| word+text | 0.5247 | 0.793 | 0.954 |
| act | 0.4724 | 0.908 | 0.978 |
| word+act | 0.4646 | 0.914 | 0.979 |
| text+act | 0.4846 | 0.892 | 0.974 |
| word+text+act | 0.4902 | 0.882 | 0.971 |
| word+text+act_note | 0.4961 | 0.868 | 0.968 |

| Information added | Bits per note [95% CI] | AUROC within wording [95% CI] |
| --- | --- | --- |
| word+act over word | +0.0995 [+0.074, +0.131] | +0.492 [+0.404, +0.580] |
| text+act over text | +0.0357 [+0.022, +0.051] | +0.081 [+0.041, +0.124] |
| word+text+act over word+text | +0.0345 [+0.021, +0.049] | +0.089 [+0.046, +0.130] |
| word+text+act_note over word+text | +0.0286 [+0.016, +0.042] | +0.075 [+0.033, +0.120] |
| word+text+act over act | -0.0178 [-0.035, -0.001] | -0.026 [-0.068, +0.015] |

`act` = activations at the end of the memory result (mean of layers 14 to 19); `act_note` = at the last token of the note; `word` = wording one hot; `text` = TF-IDF on the full prompt. A positive gain with an interval above zero means the activations carry information about execution that the baseline does not. The wording only model gives every note of a wording the same score within a fold, so its within wording AUROC (and the +0.49 gain over it) only reflects fold to fold shifts; read its bits, not its AUROC.

## 2. Directions never trained on execution (MacDiarmid et al. 2024)

| Direction | AUROC within wording [95% CI] | AUROC overall |
| --- | --- | --- |
| poisoned_vs_twin_probe | 0.695 [0.619, 0.762] | 0.914 |
| poisoned_minus_clean_means | 0.364 [0.282, 0.454] | 0.677 |
| generic_a_helpful_ai | 0.532 [0.449, 0.609] | 0.378 |
| generic_a_harmless_ai | 0.554 [0.465, 0.635] | 0.615 |
| generic_an_honest_ai | 0.539 [0.464, 0.623] | 0.452 |
| generic_doing_something_dangerous | 0.520 [0.430, 0.615] | 0.449 |
| generic_doing_something_sneaky | 0.524 [0.442, 0.611] | 0.551 |
| generic_trying_to_deceive_the_human | 0.532 [0.452, 0.621] | 0.403 |
| generic_mean_of_six | 0.566 [0.486, 0.648] | 0.441 |
| execution_diff_of_means | 0.563 [0.462, 0.657] |  |
| execution_trained_probe_C0.01 | 0.885 [0.832, 0.924] |  |

1,000 random directions (sign chosen after the fact, so this is a generous control): mean 0.56, 95th percentile 0.643, best 0.715.

## 3. Confidence based deferral (Ashok and May 2025)

The detector answers only when its confidence clears a threshold chosen on half the cases to reach the target accuracy; coverage = share of notes it answers on the other half (200 random splits).

| Detector | Target 90%: coverage / accuracy | Target 95% | Target 98% |
| --- | --- | --- | --- |
| probe_mre | 1.00 / 0.916 | 0.90 / 0.950 | 0.74 / 0.979 |
| text_full | 0.99 / 0.912 | 0.73 / 0.949 | 0.46 / 0.975 |
| probe_plus_text (conditional) | 1.00 / 0.918 | 0.86 / 0.948 | 0.68 / 0.978 |

Within wording 3 only (where wording cannot help), accuracy of the most confident predictions:

| Detector | Top 25% | Top 50% | All |
| --- | --- | --- | --- |
| probe_mre | 0.919 | 0.919 | 0.784 |
| text_full | 0.865 | 0.865 | 0.791 |
| probe_plus_text (conditional) | 0.946 | 0.878 | 0.851 |

## 4. Needs the GPU

* Generic yes/no defection probe (MacDiarmid et al.): `interp/phase3/generic_probe.py`, then rerun this script.
* Position and layer sweeps (Ashok and May) and trajectory geometry (Pandey et al.): `interp/phase3/capture.py` (always saves the sweep positions), then `interp/phase3/sweeps.py`.
* Function vectors (Todd et al.): head level activation patching, a later session.
