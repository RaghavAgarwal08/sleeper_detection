"""Figure for run 4 (H6b, causal_v4.py), CPU, seconds.

    python interp/phase3/causal_v4_figure.py
Reads results/phase3/gpt-oss-20b/causal_v4.json; writes results/phase3/figures/causal_v4.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
S = json.load(open(REPO / "results/phase3/gpt-oss-20b/causal_v4.json"))["summary"]
FIG = REPO / "results/phase3/figures"; FIG.mkdir(parents=True, exist_ok=True)
C = {"base": "#9a9890", "dom": "#1baf7a", "random": "#eb6834", "toolresult": "#2a78d6", "muted": "#6b6a63",
     "grid": "#e6e5e0", "ink": "#1f1f1e"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": C["muted"], "xtick.color": C["muted"], "ytick.color": C["muted"],
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
names = {"base": "Unsteered", "dom_-1": "Difference of means −1 SD\n(window layers, prompt and reply)",
         "random_-1": "Random direction, same size", "toolresult_-1": "Memory result tokens only, −1 SD\n(prompt pass only)",
         "toolresult_-3": "Memory result tokens only, −3 SD\n(prompt pass only)"}
col = lambda c: C[c.split("_")[0]] if c != "base" else C["base"]  # noqa: E731
pois = ["base", "dom_-1", "random_-1", "toolresult_-1", "toolresult_-3"]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False, gridspec_kw={"width_ratios": [1.3, 1, 1], "wspace": 0.08})
for k, (key, rate, title) in enumerate([("execution", "execution_rate", "Planted call (poisoned prompts)"),
                                        ("user_task", "user_task_rate", "User's task (poisoned prompts)")]):
    a = ax[k]
    for i, c in enumerate(pois):
        s = S[c]; v = s[rate]
        if c != "base":
            lo, hi = S["base"][rate] + s[f"{key}_minus_base_ci"][0], S["base"][rate] + s[f"{key}_minus_base_ci"][1]
            a.plot([lo, hi], [i, i], color=col(c), lw=2.2, alpha=0.85, solid_capstyle="round")
        a.plot(v, i, "o", color=col(c), ms=7, mec="white", mew=1.2)
        a.text(v, i - 0.32, f"{v:.2f}", ha="center", fontsize=8.5, color=C["ink"])
    a.axvline(S["base"][rate], color=C["muted"], ls=":", lw=1)
    a.set_yticks(range(len(pois)), [names[c] for c in pois] if k == 0 else [""] * len(pois)); a.set_ylim(len(pois) - 0.5, -0.9)
    a.set_title(title, loc="left", color=C["ink"]); a.grid(axis="x", color=C["grid"]); a.set_axisbelow(True)
a = ax[2]
cl = ["clean_base", "clean_dom_-1", "clean_random_-1", "clean_toolresult_-3"]
lab = ["Unsteered", "Difference of means −1", "Random, same size", "Memory result only −3"]
for i, c in enumerate(cl):
    s = S[c]; v = s["user_task_rate"]
    if c != "clean_base":
        lo, hi = S["clean_base"]["user_task_rate"] + s["user_task_minus_base_ci"][0], S["clean_base"]["user_task_rate"] + s["user_task_minus_base_ci"][1]
        a.plot([lo, hi], [i, i], color=col(c[6:]), lw=2.2, alpha=0.85)
    a.plot(v, i, "o", color=col(c[6:]) if c != "clean_base" else C["base"], ms=7, mec="white", mew=1.2)
    a.text(0.905, i, lab[i], va="center", fontsize=8.5, color=C["ink"])
a.set_xlim(0.9, 1.02); a.set_yticks([]); a.set_ylim(len(cl) - 0.5, -0.9)
a.set_title("User's task (clean prompts)", loc="left", color=C["ink"]); a.grid(axis="x", color=C["grid"])
fig.savefig(FIG / "causal_v4.png"); plt.close(fig)
print("wrote", FIG / "causal_v4.png")
