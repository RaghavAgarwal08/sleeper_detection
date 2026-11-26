"""Figure for the reading position re-test (PLAN.md section 9), CPU, seconds.

    python interp/phase3/position_retest_figure.py
Reads analysis.json and analysis_last_prompt.json per model (gpt-oss at the end of the memory result from
results/relabel_namefix/phase3_analysis.json, the corrected labels) and cross_model_last_prompt.json.
Writes results/phase3/figures/position_retest.png and prints the H1 and H2 table.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
P3 = REPO / "results/phase3"
FIG = P3 / "figures"; FIG.mkdir(parents=True, exist_ok=True)
MODELS = [("gpt-oss-20b", "gpt-oss-20b"), ("qwen3-8b", "Qwen3-8B"), ("llama3.1-8b", "Llama-3.1-8B")]
C = {"mre": "#9a9890", "lp": "#2a78d6", "muted": "#6b6a63", "grid": "#e6e5e0", "ink": "#1f1f1e"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": C["muted"], "xtick.color": C["muted"], "ytick.color": C["muted"],
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def load(m, site):
    if site == "mre" and m == "gpt-oss-20b":
        return json.load(open(REPO / "results/relabel_namefix/phase3_analysis.json"))
    return json.load(open(P3 / m / ("analysis.json" if site == "mre" else "analysis_last_prompt.json")))


rows = {}
print("| Model | Test | End of memory result | Last prompt token |")
print("| --- | --- | --- | --- |")
for m, name in MODELS:
    for h, hn in (("H1_heldout_cases_rate", "H1"), ("H2_lodo_binary", "H2")):
        cell = []
        for site in ("mre", "lp"):
            x = load(m, site)[h]["probe_mre_minus_text_full"]
            rows[(m, hn, site)] = (x["auroc_within_wording"], *x["ci"])
            cell.append(f"{x['auroc_within_wording']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]")
        print(f"| {name} | {hn} | {cell[0]} | {cell[1]} |")

cm = json.load(open(P3 / "cross_model_last_prompt.json"))
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.05], "wspace": 0.35})
for k, hn in enumerate(("H1", "H2")):
    a = ax[k]
    for i, (m, name) in enumerate(MODELS):
        for j, site in enumerate(("mre", "lp")):
            v, lo, hi = rows[(m, hn, site)]
            y = i + (j - 0.5) * 0.3
            a.plot([lo, hi], [y, y], color=C[site], lw=2.2, solid_capstyle="round", alpha=0.85)
            a.plot(v, y, "o", color=C[site], ms=6.5, mec="white", mew=1.1,
                   label=("End of memory result" if site == "mre" else "Last prompt token") if i == 0 else None)
    a.axvline(0, color=C["muted"], ls=":", lw=1)
    a.set_yticks(range(3), [n for _, n in MODELS]); a.invert_yaxis()
    a.set_xlabel("Probe minus TF-IDF, AUROC within wording (95% CI)")
    a.set_title({"H1": "H1: held out cases, rate target", "H2": "H2: held out domains, binary"}[hn], loc="left",
                color=C["ink"], fontsize=10)
    a.grid(axis="x", color=C["grid"]); a.set_axisbelow(True)
ax[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2, fontsize=8.5)
# matrix
keys = [m for m, _ in MODELS]
state = {"gpt-oss-20b": "gptoss_state", "qwen3-8b": "qwen_state", "llama3.1-8b": "llama_state"}
M = None
for key in ("matrix", "auroc_matrix"):
    if key in cm:
        M = cm[key]
if M is None:
    raise SystemExit(f"no matrix in cross_model_last_prompt.json; keys {list(cm)}")
grid = np.array([[M[f"{state[r]} -> {c}"]["auroc_within_wording"] for c in keys] for r in keys])
a = ax[2]
im = a.imshow(grid, cmap="Blues", vmin=0.55, vmax=0.9)
for r in range(3):
    for c in range(3):
        a.text(c, r, f"{grid[r, c]:.2f}", ha="center", va="center", fontsize=10,
               color="white" if grid[r, c] > 0.78 else C["ink"], fontweight="bold" if r == c else "normal")
a.set_xticks(range(3), [n for _, n in MODELS], fontsize=8.5); a.set_yticks(range(3), [n for _, n in MODELS], fontsize=8.5)
a.set_xlabel("Whose behaviour is predicted"); a.set_ylabel("Whose state (last prompt token)")
a.set_title("Each model's own state predicts it best", loc="left", color=C["ink"], fontsize=10)
a.spines[:].set_visible(False); a.grid(False)
fig.savefig(FIG / "position_retest.png"); plt.close(fig)
print("wrote", FIG / "position_retest.png")
