"""Figure for the causal follow up (interp/phase3/causal_v2.py), CPU, seconds.

    python interp/phase3/causal_v2_figures.py
Reads results/phase3/gpt-oss-20b/causal_v2.json; writes results/phase3/figures/causal_v2.png and prints a table.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
G = REPO / "results/phase3/gpt-oss-20b"
FIG = REPO / "results/phase3/figures"
FIG.mkdir(parents=True, exist_ok=True)
C = {"probe": "#2a78d6", "dom": "#1baf7a", "random": "#eb6834", "patch": "#8a63d2", "twin": "#1f1f1e", "base": "#9a9890",
     "muted": "#6b6a63", "grid": "#e6e5e0", "ink": "#1f1f1e"}
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"font.size": 10, "axes.edgecolor": C["muted"], "xtick.color": C["muted"], "ytick.color": C["muted"],
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": C["grid"],
                     "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight"})

J = json.load(open(G / "causal_v2.json"))
S = J["summary"]
sp, sd = J.get("sigma_probe", 32.5), J.get("sigma_dom", 105.8)


def label(c):
    if c == "base":
        return "Unsteered"
    if c == "twin_prompt":
        return "Benign twin's own prompt"
    if c.startswith("patch_twin_L"):
        return f"Patch twin state at layer {c[12:]}"
    kind, a = c.split("_all_")
    a = float(a)
    size = a * (sd if kind == "dom" else sp)
    name = {"probe": "Probe direction", "dom": "Difference of means", "random": "Random direction"}[kind]
    return f"{name} {a:+g} SD ({size:+.0f}/layer)"


def colour(c):
    for k in ("probe", "dom", "random", "patch", "twin"):
        if c.startswith(k):
            return C[k]
    return C["base"]


order = ["base", "twin_prompt"] + sorted([c for c in S if c.startswith("patch")]) + \
        sorted([c for c in S if c.startswith("random")], key=lambda c: float(c.split("_")[-1])) + \
        sorted([c for c in S if c.startswith("probe")], key=lambda c: float(c.split("_")[-1])) + \
        sorted([c for c in S if c.startswith("dom")], key=lambda c: float(c.split("_")[-1]))
order = [c for c in order if c in S]
base = S["base"]["execution_rate"]
fig, ax = plt.subplots(1, 2, figsize=(11, 0.42 * len(order) + 1.4), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.05},
                       sharey=True)
for i, c in enumerate(order):
    s = S[c]
    lo, hi = base + s["minus_baseline_ci"][0], base + s["minus_baseline_ci"][1]
    if c != "base":
        ax[0].plot([lo, hi], [i, i], color=colour(c), lw=2.2, solid_capstyle="round", alpha=0.8)
    ax[0].plot(s["execution_rate"], i, "o", color=colour(c), ms=7, mec="white", mew=1.2)
    ax[0].text(max(hi, s["execution_rate"]) + 0.01, i, f"{s['execution_rate']:.2f}", va="center", fontsize=8.5, color=C["ink"])
    broken = s["any_tool_call_rate"] < 0.85
    ax[1].barh(i, s["any_tool_call_rate"], color=C["random"] if broken else C["grid"], height=0.6)
    ax[1].text(s["any_tool_call_rate"] + 0.01, i, f"{s['any_tool_call_rate']:.2f}", va="center", fontsize=8.5, color=C["ink"])
ax[0].axvline(base, color=C["muted"], ls=":", lw=1)
ax[0].set_yticks(range(len(order)), [label(c) for c in order]); ax[0].invert_yaxis()
ax[0].set_xlim(-0.01, 0.45); ax[0].set_xlabel("Share of rollouts that execute the planted call (95% CI)")
ax[0].set_title("Execution under each intervention", loc="left", color=C["ink"])
ax[1].tick_params(left=False); ax[1].set_xlim(0, 1.12)
ax[1].set_xlabel("Share with any tool call (orange: below 0.85)")
ax[1].set_title("Does the agent still use tools?", loc="left", color=C["ink"])
fig.savefig(FIG / "causal_v2.png"); plt.close(fig)

print("| Condition | Execution | Minus unsteered [95% CI] | Any tool call |")
print("| --- | --- | --- | --- |")
for c in order:
    s = S[c]
    print(f"| {label(c)} | {s['execution_rate']:.3f} | {s['minus_baseline']:+.3f} [{s['minus_baseline_ci'][0]:+.3f}, "
          f"{s['minus_baseline_ci'][1]:+.3f}] | {s['any_tool_call_rate']:.2f} |")
print("wrote", FIG / "causal_v2.png")
