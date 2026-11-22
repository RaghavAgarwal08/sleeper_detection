"""Figures for PAPER_METHODS_REPORT.md from paper_methods.json and the phase 2 out of fold scores (CPU, seconds).

    python interp/probe_live_inert/paper_methods_figures.py
Writes results/probe_live_inert/figures/pm_*.png and pm_curves.json.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import probe as P  # noqa: E402

RES = REPO / "results/probe_live_inert"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)
C = {"probe": "#2a78d6", "text": "#eb6834", "aqua": "#1baf7a", "gray": "#9a9890", "ink": "#1f1f1e", "muted": "#6b6a63", "grid": "#e6e5e0"}

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink"], "xtick.color": C["muted"],
                     "ytick.color": C["muted"], "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": C["grid"], "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
                     "savefig.bbox": "tight"})
J = json.load(open(RES / "paper_methods.json"))

# 1. conditional probing: cross entropy per feature set and the information added
fs = J["conditional_probing"]["feature_sets"]
order = ["word", "text", "word+text", "act", "word+act", "text+act", "word+text+act"]
names = {"word": "Wording only", "text": "TF-IDF full prompt", "word+text": "Wording + TF-IDF", "act": "Activations only",
         "word+act": "Wording + activations", "text+act": "TF-IDF + activations", "word+text+act": "Wording + TF-IDF + activations"}
fig, ax = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.45})
vals = [fs[k]["ce_bits"] for k in order]
cols = [C["text"] if "act" not in k else C["probe"] for k in order]
ax[0].barh(range(len(order)), vals, color=cols, height=0.62)
ax[0].axvline(J["conditional_probing"]["entropy_bits_constant_predictor"], color=C["muted"], ls=":", lw=1)
ax[0].text(J["conditional_probing"]["entropy_bits_constant_predictor"], -0.55, "constant guess ", color=C["muted"], fontsize=8, va="center", ha="right")
for i, v in enumerate(vals):
    ax[0].text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8.5, color=C["ink"])
ax[0].set_yticks(range(len(order)), [names[k] for k in order]); ax[0].invert_yaxis()
ax[0].set_xlim(0.4, 0.92); ax[0].set_xlabel("Held out cross entropy, bits per note (lower is better)")
ax[0].set_title("Probes on different feature sets", loc="left", color=C["ink"])
g = J["conditional_probing"]["gains"]
gk = ["text+act over text", "word+text+act over word+text", "word+text+act_note over word+text", "word+text+act over act"]
gl = ["Activations beyond TF-IDF", "Activations beyond wording + TF-IDF", "Note-token activations beyond wording + TF-IDF", "TF-IDF + wording beyond activations"]
for i, k in enumerate(gk):
    v, (lo, hi) = g[k]["bits"], g[k]["bits_ci"]
    col = C["probe"] if lo > 0 else C["gray"]
    ax[1].plot([lo, hi], [i, i], color=col, lw=2.5, solid_capstyle="round")
    ax[1].plot(v, i, "o", color=col, ms=8, mec="white", mew=1.5)
    ax[1].text(hi + 0.002, i, f"{v:+.3f}", va="center", fontsize=8.5, color=C["ink"])
ax[1].axvline(0, color=C["muted"], lw=1, ls=":")
ax[1].set_yticks(range(len(gk)), gl); ax[1].invert_yaxis()
ax[1].set_xlabel("Information added, bits per note (95% CI)")
ax[1].set_title("Conditional probing", loc="left", color=C["ink"])
fig.savefig(FIG / "pm_conditional_probing.png"); plt.close(fig)

# 2. directions never trained on execution
d = J["directions"]
rows = [("Trained execution probe (reference)", d["execution_trained_probe_C0.01"], C["probe"]),
        ("Poisoned vs twin direction (no execution labels)", d["poisoned_vs_twin_probe"], C["aqua"]),
        ("Execution, difference of means", d["execution_diff_of_means"], C["gray"]),
        ("Poisoned minus clean, difference of means", d["poisoned_minus_clean_means"], C["gray"])]
fig, ax = plt.subplots(figsize=(9, 3.5))
for i, (lab, r, col) in enumerate(rows):
    lo, hi = r["ci"]
    ax.plot([lo, hi], [i, i], color=col, lw=2.5, solid_capstyle="round")
    ax.plot(r["auroc_within_wording"], i, "o", color=col, ms=8, mec="white", mew=1.5)
    ax.text(hi + 0.01, i, f"{r['auroc_within_wording']:.3f}", va="center", fontsize=8.5, color=C["ink"])
rd = d["random_1000"]
ax.axvspan(0.5, rd["p95"], color=C["grid"], alpha=0.9, lw=0)
ax.axvline(rd["best_of_1000"], color=C["muted"], lw=1, ls="--")
ax.set_ylim(len(rows) - 0.4, -0.9)
ax.text((0.5 + rd["p95"]) / 2, -0.62, "95% of 1,000 random directions", color=C["muted"], fontsize=8, va="center", ha="center")
ax.text(rd["best_of_1000"] + 0.005, -0.62, "best random", color=C["muted"], fontsize=8, va="center", ha="left")
ax.axvline(0.5, color=C["muted"], lw=1, ls=":")
ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.set_xlim(0.25, 1.0)
ax.set_xlabel("AUROC within wording for execution (95% CI)")
ax.set_title("Directions never trained on execution", loc="left", color=C["ink"])
fig.savefig(FIG / "pm_directions.png"); plt.close(fig)

# 3. deferral: coverage at each target accuracy (finer curve than the table), and confidence ranked accuracy in wording 3
labels, _ = P.load_labels(REPO / "interp/data/behavior_v1")
ids = np.array([l["id"] for l in labels])
beh = {b["id"]: b for b in map(json.loads, gzip.open(REPO / "interp/data/behavior_v1/behavior.jsonl.gz", "rt"))}
oof = np.load(RES / "phase2_oof_cases_rate.npz")
opos = {i: k for k, i in enumerate(oof["ids"])}
POP = np.array([k for k, l in enumerate(labels) if l["framing"] == "live" and l["id"] in beh])
y = np.array([beh[ids[k]]["fired_any"] / beh[ids[k]]["k"] >= 0.5 for k in POP]).astype(int)
cases = np.array([labels[k]["case_id"] for k in POP])
tm = np.array([labels[k]["template_id"] for k in POP])
S = {"probe_mre": oof["probe_mre"][[opos[ids[k]] for k in POP]], "text_full": oof["text_full"][[opos[ids[k]] for k in POP]]}
targets = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
uc = np.unique(cases)
curves = {}
for name, s in S.items():
    pred, conf = (s >= 0.5).astype(int), np.maximum(s, 1 - s)
    cv = []
    for t in targets:
        cov = []
        for rep in range(200):
            rr = np.random.default_rng(rep)
            cal = np.isin(cases, rr.choice(uc, len(uc) // 2, replace=False))
            thr = 1.01
            for th in np.unique(conf[cal]):
                m = conf[cal] >= th
                if m.sum() >= 10 and (pred[cal][m] == y[cal][m]).mean() >= t:
                    thr = th
                    break
            cov.append((conf[~cal] >= thr).mean())
        cv.append((float(np.mean(cov)), [float(x) for x in np.percentile(cov, [10, 90])]))
    curves[name] = cv
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for name, col, lab in [("probe_mre", C["probe"], "Probe"), ("text_full", C["text"], "TF-IDF full prompt")]:
    m = [c[0] for c in curves[name]]
    lo = [c[1][0] for c in curves[name]]; hi = [c[1][1] for c in curves[name]]
    ax[0].fill_between([t * 100 for t in targets], lo, hi, color=col, alpha=0.15, lw=0)
    ax[0].plot([t * 100 for t in targets], m, "o-", color=col, lw=2, ms=6, mec="white", mew=1.2, label=lab)
ax[0].set_xlabel("Required accuracy on the notes it answers (%)"); ax[0].set_ylabel("Share of notes answered")
ax[0].set_ylim(0, 1.05); ax[0].legend(loc="lower left")
ax[0].set_title("Answer when confident, defer otherwise", loc="left", color=C["ink"])
w3 = tm == 3
fr = np.linspace(0.1, 1.0, 19)
for name, col, lab in [("probe_mre", C["probe"], "Probe"), ("text_full", C["text"], "TF-IDF full prompt")]:
    s = S[name][w3]; yy = y[w3]
    pred, conf = (s >= 0.5).astype(int), np.maximum(s, 1 - s)
    o = np.argsort(-conf)
    acc = [(pred[o[:max(1, int(f * len(o)))]] == yy[o[:max(1, int(f * len(o)))]]).mean() for f in fr]
    ax[1].plot(fr * 100, acc, color=col, lw=2, label=lab)
ax[1].axhline(yy.mean(), color=C["muted"], ls=":", lw=1)
ax[1].text(100, yy.mean(), " always guess\n executed", color=C["muted"], fontsize=8, va="center")
ax[1].set_xlabel("Most confident share of wording 3 notes (%)"); ax[1].set_ylabel("Accuracy")
ax[1].set_title("Within one wording (wording 3, 148 notes)", loc="left", color=C["ink"]); ax[1].legend(loc="lower left")
fig.savefig(FIG / "pm_deferral.png"); plt.close(fig)
(RES / "pm_curves.json").write_text(json.dumps({"targets": targets, "coverage": curves}, indent=1))
print("figures written", sorted(p.name for p in FIG.glob("pm_*.png")))
