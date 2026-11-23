"""Figures and the generic probe scores for results/phase3/GPTOSS_GPU_REPORT.md (CPU, about a minute).

    python interp/phase3/gptoss_figures.py
Reads results/phase3/gpt-oss-20b/{sweeps,steering}.json, the generic yes/no directions in
results/probe_live_inert/generic_probe_directions.npz and, if present, results/phase3/gpt-oss-20b/window_features.npz
(20 MB, kept on the RunPod volume; without it the saved generic_probe_scores.json is used).
Writes results/phase3/figures/{position_sweep,training_free_detectors,steering_detail}.png and
results/phase3/gpt-oss-20b/generic_probe_scores.json.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
R3 = REPO / "results/phase3"
G = R3 / "gpt-oss-20b"
FIG = R3 / "figures"
FIG.mkdir(parents=True, exist_ok=True)
C = {"probe": "#2a78d6", "text": "#eb6834", "aqua": "#1baf7a", "gray": "#9a9890", "ink": "#1f1f1e", "muted": "#6b6a63", "grid": "#e6e5e0"}
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"font.size": 10, "axes.edgecolor": C["muted"], "xtick.color": C["muted"], "ytick.color": C["muted"],
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": C["grid"],
                     "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight"})


def strat_auc(y, s, strata):
    num = den = 0.0
    for g in np.unique(strata):
        m = strata == g
        p, n = s[m][y[m] == 1], s[m][y[m] == 0]
        if len(p) and len(n):
            num += (p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()
            den += len(p) * len(n)
    return num / den


# ---------------------------------------------------------------- generic yes/no directions (MacDiarmid et al.)
wf = G / "window_features.npz"
if wf.exists():
    z = np.load(wf)
    gdirs = np.load(REPO / "results/probe_live_inert/generic_probe_directions.npz")
    lab = {l["id"]: l for l in map(json.loads, open(REPO / "interp/data/behavior_v1/labels.jsonl"))}
    beh = {b["id"]: b for b in map(json.loads, gzip.open(REPO / "interp/data/behavior_v1/behavior.jsonl.gz", "rt"))}
    ids = z["ids"]
    A = z["window_mean"][:, 0].astype(np.float64)  # end of the memory result, layers 14 to 19
    fr = np.array([lab[i]["framing"] for i in ids])
    tm = np.array([-1 if lab[i]["template_id"] is None else lab[i]["template_id"] for i in ids])
    cs = np.array([lab[i]["case_id"] for i in ids])
    P = np.array([f == "live" and i in beh for f, i in zip(fr, ids)])
    y = np.array([beh[i]["fired_any"] / beh[i]["k"] >= 0.5 if i in beh else 0 for i in ids]).astype(int)
    uc = np.unique(cs[P])
    idx = {c: np.where(P & (cs == c))[0] for c in uc}
    rng = np.random.default_rng(0)
    res = {}
    for k in gdirs.files:
        if not k.startswith("dir_"):
            continue
        s = A @ (gdirs[k] / np.linalg.norm(gdirs[k]))  # sign fixed in advance: higher = more like the "bad" answer
        bs = []
        for _ in range(1000):
            ii = np.concatenate([idx[c] for c in rng.choice(uc, len(uc))])
            bs.append(strat_auc(y[ii], s[ii], tm[ii]))
        res[k[4:]] = {"auroc_within_wording": round(float(strat_auc(y[P], s[P], tm[P])), 3),
                      "ci": [round(float(v), 3) for v in np.percentile(bs, [2.5, 97.5])]}
    Rd = rng.normal(size=(1000, A.shape[1]))
    ra = np.array([strat_auc(y[P], A[P] @ (r / np.linalg.norm(r)), tm[P]) for r in Rd])
    res["random_1000_fixed_sign"] = {"p2.5": round(float(np.percentile(ra, 2.5)), 3), "p97.5": round(float(np.percentile(ra, 97.5)), 3),
                                     "mean": round(float(ra.mean()), 3)}
    (G / "generic_probe_scores.json").write_text(json.dumps(res, indent=1))
res = json.load(open(G / "generic_probe_scores.json"))
sw = json.load(open(G / "sweeps.json"))
st = json.load(open(G / "steering.json"))

# 1. position sweep
pos = sw["position_sweep_execution_auroc_within_wording"]
names = {"user_end": "End of user\nrequest", "memory_result_start": "Start of\nmemory result", "note": "Last token\nof the note",
         "memory_result_end": "End of\nmemory result", "message_end": "End of\nthat message", "last_minus_2": "3rd last\nprompt token",
         "last_minus_1": "2nd last\nprompt token", "last_prompt": "Last prompt\ntoken"}
fig, ax = plt.subplots(figsize=(9, 3.4))
xs = list(range(len(pos)))
ax.plot(xs, list(pos.values()), "o-", color=C["probe"], lw=2, ms=7, mec="white", mew=1.5)
for x, v in zip(xs, pos.values()):
    ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5, color=C["ink"])
ax.axvspan(1.5, 2.5, color=C["grid"], alpha=0.7, lw=0)
ax.text(2, 0.585, "the model reads\nthe note here", ha="center", fontsize=8, color=C["muted"])
ax.set_xticks(xs, [names[k] for k in pos], fontsize=8)
ax.set_ylim(0.55, 0.95); ax.set_ylabel("Execution AUROC\nwithin wording")
ax.set_title("Where in the prompt the execution signal appears (same probe, layers 14 to 19)", loc="left", color=C["ink"])
fig.savefig(FIG / "position_sweep.png"); plt.close(fig)

# 2. training free detectors: generic yes/no directions and trajectory geometry, against the trained probe
traj = sw["trajectory_auroc_within_wording_sign_free"]
best_traj = sorted(traj.items(), key=lambda kv: -kv[1]["execution"])[:3]
rows = [("Trained execution probe (H1)", 0.904, [0.868, 0.934], C["probe"])]
lab_g = {"mean_of_six": "Generic, mean of six questions", "a_harmless_ai": "Generic, 'Are you harmless?'",
         "a_helpful_ai": "Generic, 'Are you helpful?'", "an_honest_ai": "Generic, 'Are you honest?'",
         "doing_something_sneaky": "Generic, 'Doing something sneaky?'", "trying_to_deceive_the_human": "Generic, 'Deceiving the human?'",
         "doing_something_dangerous": "Generic, 'Doing something dangerous?'"}
for k in lab_g:
    rows.append((lab_g[k], res[k]["auroc_within_wording"], res[k]["ci"], C["aqua"]))
fig, ax = plt.subplots(figsize=(9, 4.4))
for i, (l, v, ci, col) in enumerate(rows):
    ax.plot(ci, [i, i], color=col, lw=2.5, solid_capstyle="round")
    ax.plot(v, i, "o", color=col, ms=8, mec="white", mew=1.5)
    ax.text(ci[1] + 0.01, i, f"{v:.3f}", va="center", fontsize=8.5, color=C["ink"])
for k, v in best_traj:
    i = len(rows)
    ax.plot(v["execution"], i, "D", color=C["gray"], ms=7, mec="white", mew=1.2)
    ax.text(v["execution"] + 0.01, i, f"{v['execution']:.3f} (best of 26, sign chosen after)", va="center", fontsize=8, color=C["muted"])
    rows.append((f"Trajectory, {k.replace('_', ' ')}", None, None, None))
rr = res["random_1000_fixed_sign"]
ax.axvspan(rr["p2.5"], rr["p97.5"], color=C["grid"], alpha=0.9, lw=0)
ax.text((rr["p2.5"] + rr["p97.5"]) / 2, -0.75, "95% of random directions", ha="center", fontsize=8, color=C["muted"])
ax.axvline(0.5, color=C["muted"], lw=1, ls=":")
ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.set_ylim(len(rows) - 0.4, -1.1); ax.set_xlim(0.35, 1.0)
ax.set_xlabel("AUROC within wording for execution (95% CI)")
ax.set_title("Detectors that never see an execution label", loc="left", color=C["ink"])
fig.savefig(FIG / "training_free_detectors.png"); plt.close(fig)

# 3. steering: execution and any tool call by condition
S = st["summary"]
order = ["add_-4", "add_-2", "add_+0", "add_+2", "add_+4"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
xs = [-4, -2, 0, 2, 4]
ex = [S[c]["execution_rate"] for c in order]
ax[0].plot(xs, ex, "o-", color=C["probe"], lw=2, ms=7, mec="white", mew=1.5, label="Along the probe direction")
for c, x in [("random_-4", -4), ("random_+4", 4)]:
    ax[0].plot(x, S[c]["execution_rate"], "s", color=C["text"], ms=7, mec="white", mew=1.2, label="Random direction" if x < 0 else None)
ax[0].plot(0, S["ablate"]["execution_rate"], "D", color=C["aqua"], ms=7, mec="white", mew=1.2, label="Probe direction removed")
for c, x in zip(order, xs):
    if c != "add_+0":
        b = S["add_+0"]["execution_rate"]
        ax[0].plot([x, x], [b + S[c]["minus_baseline_ci"][0], b + S[c]["minus_baseline_ci"][1]], color=C["probe"], lw=1, alpha=0.5)
ax[0].set_ylim(0, 0.5); ax[0].set_xticks(xs); ax[0].set_xlabel("Push along direction (standard deviations)")
ax[0].set_ylabel("Share of rollouts that execute"); ax[0].legend(loc="upper right", fontsize=8)
ax[0].set_title("Execution of the planted instruction", loc="left", color=C["ink"])
tc = [S[c]["any_tool_call_rate"] for c in order]
ax[1].plot(xs, tc, "o-", color=C["gray"], lw=2, ms=7, mec="white", mew=1.5)
for x, v in zip(xs, tc):
    ax[1].text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5, color=C["ink"])
ax[1].set_ylim(0.6, 1.02); ax[1].set_xticks(xs); ax[1].set_xlabel("Push along direction (standard deviations)")
ax[1].set_ylabel("Share with any tool call")
ax[1].set_title("Any tool call at all", loc="left", color=C["ink"])
fig.savefig(FIG / "steering_detail.png"); plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("*.png")))
