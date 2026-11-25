# Re-analysis on corrected execution labels (November 24, 2026)

**Why.** vLLM's tool call parser sometimes kept gpt-oss-20b's channel tag in the tool name (for example `ShopifyDeleteProduct<|channel|>commentary`), and the phase 1 judge requires an exact name. 522 of 11,214 saved calls had the tag, 188 of them the planted call, so those executions were not counted. `interp/probe_live_inert/relabel_behavior.py` strips the tag and re-judges every saved call, writing `interp/data/behavior_v1/behavior_namefix.jsonl.gz` (the original file is unchanged; original counts are kept in `fired_any_original` and `fired_first_original`).

**Label change.** Mean execution rate 0.294 to 0.318; 135 of 888 poisoned notes change rate (all upward); reached execution 286 to 297 (11 notes flip); correlation with the original rates 0.989.

**Re-analysis.** The same scripts, unchanged, rerun with the corrected file in place of the original. Outputs in this folder; the originals stay where they were.

| Analysis | Original labels | Corrected labels |
| --- | --- | --- |
| **H1** held out cases, probe minus full prompt TF-IDF, AUROC within wording | +0.099 [+0.031, +0.167] | **+0.121 [+0.046, +0.197]** |
| H1 probe / TF-IDF | 0.904 / 0.805 | 0.903 / 0.782 |
| H1 Spearman difference (H3) | +0.165 [+0.093, +0.240] | +0.185 [+0.108, +0.264] |
| **H2** leave one domain out | +0.142 [+0.017, +0.265] | **+0.131 [+0.004, +0.257]** |
| H4 poisoned vs benign twin | 0.9998 / 1.0 | 0.9998 / 1.0 |
| Phase 2 A, leave one domain out (exploratory) | +0.200 [+0.083, +0.318] | +0.178 [+0.058, +0.296] |
| Phase 2 B, held out cases, rate target | +0.104 [+0.035, +0.173] | +0.106 [+0.035, +0.178] |
| Phase 2 B2, held out cases, yes/no target | +0.067 [−0.013, +0.149] | +0.093 [+0.016, +0.175] |
| Phase 2 C, phase 1 control on 512 notes | +0.035 [−0.052, +0.127] | +0.000 [−0.083, +0.090] |
| Conditional probing, activations beyond wording and TF-IDF (bits) | +0.035 [+0.021, +0.049] | +0.050 [+0.033, +0.071] |
| Same, AUROC within wording | +0.089 [+0.046, +0.130] | +0.117 [+0.065, +0.172] |
| Poisoned vs twin direction, execution within wording | 0.695 | 0.691 |
| Generic yes/no directions, mean of six | 0.566 | 0.570 |
| Deferral at 98% target: probe / TF-IDF coverage | 0.74 / 0.46 | 0.72 / 0.42 |

**Conclusion.** Every pre-registered test that held still holds, and most gaps between the probe and the text baseline widen slightly, because the text baseline loses more than the probe when the missed executions are restored. H2's lower interval bound moves close to zero (+0.004). The small phase 1 control (C), which was already inconclusive, is now exactly zero. Going forward the corrected labels are the primary ones; the paper reports both.

Not rerun: the layer sweep (needs the per layer activations on the RunPod volume) and the steering runs (their outcomes come from new generations, see `results/phase3/CAUSAL_REPORT.md`).
