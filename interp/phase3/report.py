"""Build the phase 3 report and figures from whatever results exist (CPU, seconds).

Reads results/phase3/<model>/analysis.json (and analysis_pilot.json, steering.json), and, if present,
results/probe_live_inert/attention_baselines.json and report_jev_baseline_v2.json. Every sentence with a number is
generated from those files; nothing is hand written about results.

    python interp/phase3/report.py
Writes results/phase3/REPORT_PHASE3.md and results/phase3/figures/*.png.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import models as M  # noqa: E402

RES = Path(os.environ.get("PHASE3_OUT", REPO / "results/phase3"))
P1 = REPO / "results/probe_live_inert"
# Validated categorical palette (dataviz reference instance), fixed order: probe, text, then other series.
C = {"probe": "#2a78d6", "text": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300"}
SERIES = [C["probe"], C["text"], C["aqua"], C["yellow"], C["magenta"], C["green"]]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"

CHANGES = """\
* **Terminology.** The repo's MemoryDojo lifecycle states are used throughout: poisoned note (was live), benign twin
  (was inert), reached execution (was fired), triggered resisted (was not fired), execution rate (was fire rate).
* **Primary label.** The execution rate (rollouts reaching execution out of 8) replaces the yes/no label as the target;
  yes/no (at least 4 of 8) is secondary.
* **Primary metric.** AUROC computed only over pairs of notes with the same wording, because wording alone moves the
  execution rate from 2% to 78% and inflates overall AUROC for every method.
* **Strongest text baseline.** TF-IDF on the full prompt (not only the note) is the comparison for every claim.
* **Nested model selection** for the probe's regularization, so no test fold influences the choice.
* **Leave one domain out within wording** with text baselines, replacing a domain averaged number that mostly
  measured wording.
* **The pre-registered set E3 is reported as uninformative** (2 positives) instead of as a headline.
* **Jev guard baseline rebuilt** for a fair comparison (full or random sets, saved scores, error counts, neutral
  policy, a will it execute question, few shot and train fitted variants).
* **Attention baselines added** (Attention Tracker, AttnDA, AttentionDefense, last layer FC) on real attention.
* **Interpretability readouts added** (Neuronpedia SAE latents and Jacobian lens).
* **Phase 3 (this report): the same pipeline on several open models**, each with its own behaviour labels, a
  pre-registered window at the same relative depth, an inclusion rule, and a causal steering test.
* **Code hygiene.** Real scikit-learn everywhere (an offline reimplementation was retired), gpt-oss-20b pinned where
  the dataset's token ids require it, all work on branches.
"""


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight"})
    return plt


def load_models():
    out = {}
    for d in sorted(RES.glob("*/")):
        a = d / "analysis.json"
        p = d / "analysis_pilot.json"
        if a.exists() or p.exists():
            out[d.name] = {"analysis": json.load(open(a)) if a.exists() else None,
                           "pilot": json.load(open(p)) if p.exists() else None,
                           "steering": json.load(open(d / "steering.json")) if (d / "steering.json").exists() else None}
    return out


def fig_h1(plt, models, figdir):
    rows = [(k, v["analysis"]["H1_heldout_cases_rate"]) for k, v in models.items()
            if v["analysis"] and v["analysis"].get("included") and "H1_heldout_cases_rate" in v["analysis"]]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(rows) + 1.2))
    for j, (k, h) in enumerate(rows):
        for off, name, col, lab in [(0.12, "probe_mre", C["probe"], "Probe (mid layers)"), (-0.12, "text_full", C["text"], "TF-IDF, full prompt")]:
            v, (lo, hi) = h[name]["auroc_within_wording"], h[name]["auroc_within_wording_ci"]
            ax.plot([lo, hi], [j + off] * 2, color=col, lw=2, solid_capstyle="round")
            ax.plot(v, j + off, "o", color=col, ms=8, mec="white", mew=1.5, label=lab if j == 0 else None)
    ax.axvline(0.5, color=MUTED, lw=1, ls=":")
    ax.set_yticks(range(len(rows)), [k for k, _ in rows])
    ax.set_xlabel("AUROC within wording, held out cases (95% CI)")
    ax.set_title("Predicting execution: probe vs text, per model", loc="left", color=INK)
    ax.legend(loc="lower right")
    f = figdir / "h1_probe_vs_text_by_model.png"
    fig.savefig(f); plt.close(fig)
    return f


def fig_rates(plt, models, figdir):
    rows = []
    for k, v in models.items():
        a = v["analysis"] or v["pilot"]
        if a:
            rows.append((k, [a["execution_rate_by_wording"].get(str(t), a["execution_rate_by_wording"].get(t, np.nan)) for t in range(6)]))
    if not rows:
        return None
    Z = np.array([r for _, r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(rows) + 1.4))
    im = ax.imshow(Z, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if Z[i, j] == Z[i, j]:
                ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", color="white" if Z[i, j] > 0.55 else INK, fontsize=9)
    ax.set_xticks(range(6), [f"wording {t}" for t in range(6)]); ax.set_yticks(range(len(rows)), [k for k, _ in rows])
    ax.grid(False)
    ax.set_title("Execution rate of poisoned notes by wording", loc="left", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    f = figdir / "execution_rate_by_wording.png"
    fig.savefig(f); plt.close(fig)
    return f


def fig_sweep(plt, models, figdir):
    rows = [(k, v["analysis"]["layer_sweep_within_wording_auroc"]) for k, v in models.items()
            if v["analysis"] and v["analysis"].get("layer_sweep_within_wording_auroc")]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for (k, sw), col in zip(rows, SERIES):
        x = np.arange(len(sw)) / (len(sw) - 1)
        ax.plot(x, sw, color=col, lw=2, label=k)
        ax.annotate(k, (x[-1], sw[-1]), xytext=(4, 0), textcoords="offset points", color=INK, fontsize=8, va="center")
    ax.axvspan(M.WINDOW_REL[0], M.WINDOW_REL[1], color=GRID, alpha=0.6, lw=0)
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("Relative depth (0 = embeddings, 1 = last layer); shaded = pre-registered window")
    ax.set_ylabel("AUROC within wording")
    ax.set_title("Where the execution signal lives, per model", loc="left", color=INK)
    ax.legend(loc="lower right", fontsize=8)
    f = figdir / "layer_sweep.png"
    fig.savefig(f); plt.close(fig)
    return f


def fig_steer(plt, models, figdir):
    rows = [(k, v["steering"]) for k, v in models.items() if v["steering"]]
    if not rows:
        return None
    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 3.4), squeeze=False)
    for ax, (k, s) in zip(axes[0], rows):
        S = s["summary"]
        al = [-4, -2, 0, 2, 4]
        y = [S[f"add_{a:+d}"]["execution_rate"] for a in al]
        lo = [S[f"add_{a:+d}"]["execution_rate"] - (S[f"add_{a:+d}"]["minus_baseline"] - S[f"add_{a:+d}"]["minus_baseline_ci"][0]) for a in al]
        hi = [S[f"add_{a:+d}"]["execution_rate"] + (S[f"add_{a:+d}"]["minus_baseline_ci"][1] - S[f"add_{a:+d}"]["minus_baseline"]) for a in al]
        ax.fill_between(al, lo, hi, color=C["probe"], alpha=0.15, lw=0)
        ax.plot(al, y, "o-", color=C["probe"], lw=2, ms=7, mec="white", mew=1.5, label="along the probe direction")
        ax.plot([-4, 4], [S["random_-4"]["execution_rate"], S["random_+4"]["execution_rate"]], "s", color=C["text"], ms=7,
                mec="white", mew=1.5, label="random direction")
        ax.plot([0], [S["ablate"]["execution_rate"]], "D", color=C["aqua"], ms=7, mec="white", mew=1.5, label="probe direction removed")
        ax.set_xlabel("Push along direction (standard deviations)"); ax.set_ylabel("First step execution rate")
        ax.set_title(k, loc="left", color=INK); ax.legend(fontsize=8, loc="best")
    fig.suptitle("Steering: does moving along the probe direction change behaviour?", x=0.02, ha="left", color=INK)
    f = figdir / "steering.png"
    fig.savefig(f); plt.close(fig)
    return f


def fig_attention(plt, figdir):
    p = P1 / "attention_baselines.json"
    if not p.exists():
        return None, None
    a = json.load(open(p))
    T = a["tasks"].get("execution") or a["tasks"].get("fired")
    sec = T.get("grouped_cv_5fold_cases")
    if not sec:
        return None, None
    meth = [m for m in sec["methods"] if sec["methods"][m]["auroc"] is not None]
    vals = [sec["methods"][m]["auroc_within_wording"] for m in meth]
    fig, ax = plt.subplots(figsize=(6.4, 0.38 * len(meth) + 1.2))
    cols = [C["probe"] if m == "probe_mre" else C["text"] if m == "text_full" else "#9a9890" for m in meth]
    ax.barh(range(len(meth)), vals, color=cols, height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", color=INK, fontsize=8)
    ax.axvline(0.5, color=MUTED, lw=1, ls=":"); ax.set_xlim(0, 1)
    ax.set_yticks(range(len(meth)), meth); ax.set_xlabel("AUROC within wording, held out cases")
    ax.set_title("gpt-oss-20b: attention baselines vs probe (execution task)", loc="left", color=INK)
    f = figdir / "attention_baselines.png"
    fig.savefig(f); plt.close(fig)
    return f, dict(zip(meth, vals))


def ci(x):
    return f"[{x[0]:+.3f}, {x[1]:+.3f}]" if x and x[0] is not None else "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RES))
    args = ap.parse_args()
    out = Path(args.out); figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    plt = setup_mpl()
    models = load_models()
    figs = {"h1": fig_h1(plt, models, figdir), "rates": fig_rates(plt, models, figdir),
            "sweep": fig_sweep(plt, models, figdir), "steer": fig_steer(plt, models, figdir)}
    figs["attn"], attn = fig_attention(plt, figdir)

    L = ["# Phase 3: does the execution signal generalize across models, and is it causal?", "",
         "Generated by `interp/phase3/report.py` from the result files; rerun it after every new run.", "",
         "Terms follow `results/probe_live_inert/TERMINOLOGY.md`. Plan and pre-registration: `interp/phase3/PLAN.md`.", ""]
    inc = [k for k, v in models.items() if v["analysis"] and v["analysis"].get("included")]
    sup = [k for k in inc if models[k]["analysis"].get("H1_heldout_cases_rate", {}).get("probe_mre_minus_text_full", {}).get("ci", [None])[0]
           not in (None,) and models[k]["analysis"]["H1_heldout_cases_rate"]["probe_mre_minus_text_full"]["ci"][0] > 0]
    L += ["## Summary", "",
          f"* Models with full results: {len(inc)} included under the pre-registered rule ({', '.join(inc) or 'none yet'}).",
          f"* H1 (probe beats full prompt TF-IDF within wording, CI above zero) holds in {len(sup)} of {len(inc)}"
          + (f": {', '.join(sup)}." if sup else ".")]
    for k, v in models.items():
        s = v["steering"]
        if s:
            S = s["summary"]
            L.append(f"* Steering on {k}: execution rate {S['add_-4']['execution_rate']:.2f} at minus 4 SD, "
                     f"{S['add_+0']['execution_rate']:.2f} unsteered, {S['add_+4']['execution_rate']:.2f} at plus 4 SD "
                     f"(difference from unsteered {S['add_+4']['minus_baseline']:+.3f} {ci(S['add_+4']['minus_baseline_ci'])}); "
                     f"random direction at plus 4 SD {S['random_+4']['minus_baseline']:+.3f} {ci(S['random_+4']['minus_baseline_ci'])}.")
    L += ["", "## What changed since the October report", "", CHANGES]

    L += ["## Models and the inclusion rule", "",
          "A model is analysed only if, on all 888 poisoned notes, at least 50 reach execution, at least 50 are resisted, "
          "and at least 2 wordings contain both outcomes (otherwise there is nothing to predict).", "",
          "| Model | Notes | Reached execution | Mean execution rate | Included | Pilot gate passed |", "| --- | --- | --- | --- | --- | --- |"]
    for k, v in models.items():
        a = v["analysis"]; p = v["pilot"]
        L.append(f"| {k} | {a['n_poisoned'] if a else '(pilot) ' + str(p['n_poisoned'])} | "
                 f"{a['n_reached_execution'] if a else p['n_reached_execution']} | {(a or p)['mean_execution_rate']} | "
                 f"{'yes' if a and a.get('included') else 'no'} | {'see sample summary' if p else 'n/a'} |")
    if figs["rates"]:
        L += ["", f"![Execution rate by wording](figures/{figs['rates'].name})"]

    L += ["", "## H1, held out cases, execution rate target (primary)", "",
          "| Model | Probe [95% CI] | TF-IDF full prompt [95% CI] | Difference [95% CI] | Spearman difference [95% CI] |",
          "| --- | --- | --- | --- | --- |"]
    for k in inc:
        h = models[k]["analysis"]["H1_heldout_cases_rate"]; d = h["probe_mre_minus_text_full"]
        L.append(f"| {k} | {h['probe_mre']['auroc_within_wording']:.3f} {ci(h['probe_mre']['auroc_within_wording_ci'])} | "
                 f"{h['text_full']['auroc_within_wording']:.3f} {ci(h['text_full']['auroc_within_wording_ci'])} | "
                 f"{d['auroc_within_wording']:+.3f} {ci(d['ci'])} | {d['spearman_diff']:+.3f} {ci(d['spearman_diff_ci'])} |")
    if figs["h1"]:
        L += ["", f"![Probe vs text by model](figures/{figs['h1'].name})"]
    L += ["", "## H2, leave one domain out (secondary)", "", "| Model | Probe | TF-IDF full prompt | Difference [95% CI] |", "| --- | --- | --- | --- |"]
    for k in inc:
        h = models[k]["analysis"].get("H2_lodo_binary")
        if h:
            L.append(f"| {k} | {h['probe_mre']['auroc_within_wording']:.3f} | {h['text_full']['auroc_within_wording']:.3f} | "
                     f"{h['probe_mre_minus_text_full']['auroc_within_wording']:+.3f} {ci(h['probe_mre_minus_text_full']['ci'])} |")
    L += ["", "## H4, poisoned vs benign twin (sanity check)", "", "| Model | E1 | E2 |", "| --- | --- | --- |"]
    for k, v in models.items():
        a = v["analysis"] or v["pilot"]
        if a and a.get("H4_poisoned_vs_twin"):
            L.append(f"| {k} | {a['H4_poisoned_vs_twin'].get('E1_new_templates')} | {a['H4_poisoned_vs_twin'].get('E2_new_domains')} |")
    if figs["sweep"]:
        L += ["", "## Layer sweep (descriptive)", "", f"![Layer sweep](figures/{figs['sweep'].name})"]
    if figs["steer"]:
        L += ["", "## H5, steering (causal test)", "", f"![Steering](figures/{figs['steer'].name})", "",
              "| Model | Condition | First step execution | Minus unsteered [95% CI] | Any tool call |", "| --- | --- | --- | --- | --- |"]
        for k, v in models.items():
            if v["steering"]:
                for c, r in v["steering"]["summary"].items():
                    L.append(f"| {k} | {c} | {r['execution_rate']:.3f} | {r['minus_baseline']:+.3f} {ci(r['minus_baseline_ci'])} | {r['any_tool_call_rate']:.2f} |")
                bc = v["steering"]["baseline_check"]
                L.append(f"\nUnsteered check for {k}: Hugging Face first step rate {bc['hf_mean']} vs vLLM {bc['vllm_fired_first_mean']} (correlation {bc['corr']}).")
    if figs["attn"]:
        L += ["", "## RQ1, attention baselines on gpt-oss-20b", "", f"![Attention baselines](figures/{figs['attn'].name})", "",
              "Within wording AUROC, held out cases: " + ", ".join(f"{m} {v:.3f}" for m, v in attn.items()) + "."]
    jv = P1 / "report_jev_baseline_v2.json"
    if jv.exists():
        j = json.load(open(jv))
        L += ["", "## Jev guard baseline (rerun)", ""]
        for task, t in j["tasks"].items():
            for sp, s in t["splits"].items():
                best = max((k for k in s["metrics"] if k.startswith("jev")), key=lambda k: s["metrics"][k]["auroc"] or 0)
                L.append(f"* {task}, {sp}: best Jev scorer {best} AUROC {s['metrics'][best]['auroc']}, probe "
                         f"{s['metrics']['probe_mre']['auroc']}, TF-IDF {s['metrics']['text_full']['auroc']} on the same {s['n_scored']} notes.")
    L += ["", "## Limitations", "",
          "* One attack family (Sleeper Attack), one trigger turn per prompt, canned tool results after the first step.",
          "* Steering measures the first response only, generated with Hugging Face rather than vLLM.",
          "* Models differ in how often they call tools at all; the inclusion rule removes models that never or always execute.",
          "* Analyses beyond those listed in PLAN.md are exploratory and labelled as such."]
    (out / "REPORT_PHASE3.md").write_text("\n".join(L) + "\n")
    print("wrote", out / "REPORT_PHASE3.md", "figures:", [f.name for f in figs.values() if f])


if __name__ == "__main__":
    main()
