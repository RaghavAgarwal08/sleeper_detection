"""Phase 2 analyses on the cached behavior_v1 features (CPU only, no GPU).

Addresses the review of the phase 1 results:
  A. Leave-one-domain-out (LODO) evaluated WITHIN wording and next to text and wording baselines. A plain LODO AUROC
     mixes the six wordings, and wording alone predicts most of the label, so it mostly measures wording.
  B. Fire rate (0 to 8 of 8 rollouts) as the PRIMARY label. Models are fit with binomial sample weights
     (each note counts as a positive with weight r and a negative with weight 1 - r, r = fired/8). Reported:
     Spearman correlation with the fire rate, pooled and within wording, plus within-wording AUROC on the binary
     label (fired = at least 4 of 8) as a secondary metric.
  C. Nested cross-validation for the probe's regularization C, removing the small optimistic bias of choosing C
     on folds that are later used for testing.

Every comparison resamples whole cases (the 13 versions of a case are not independent) for 95% CIs, and every
probe vs text difference is a paired bootstrap on the same resampled cases.

Text baselines use the C chosen by the pre-registered leave-one-template-out procedure on the training split
(C = 1.0 for both TF-IDF models), held fixed, because nesting the TF-IDF search over 43 folds is too slow on a CPU.
The probe's C is nested (inner GroupKFold by case on the training folds only).

    python interp/probe_live_inert/phase2_analysis.py           # about 10 to 15 minutes on a laptop
Writes results/probe_live_inert/phase2_analysis.json and PHASE2_ANALYSIS.md.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import probe as P  # noqa: E402

DATA = REPO / "interp/data/behavior_v1"
RES = REPO / "results/probe_live_inert"
N_BOOT = 1000
SEED = 0
C_GRID = P.C_GRID
TEXT_C = 1.0

# ------------------------------------------------------------------------------------------------ data
labels, manifest = P.load_labels(DATA)
sets = P.eval_sets(labels, manifest)
ids = np.array([l["id"] for l in labels])
bpath = DATA / "behavior.jsonl"
beh_lines = open(bpath) if bpath.exists() else gzip.open(str(bpath) + ".gz", "rt")
beh = {b["id"]: b for b in map(json.loads, beh_lines)}
live = np.array([l["framing"] == "live" and l["id"] in beh for l in labels])
rate = np.array([beh[i]["fired_any"] / beh[i]["k"] if i in beh else np.nan for i in ids])
y = (rate >= 0.5).astype(int)
cases = np.array([l["case_id"] for l in labels])
tmpl = np.array([-1 if l["template_id"] is None else int(l["template_id"]) for l in labels])
dom = np.array([l["domain"] for l in labels])

feat = np.load(RES / "features_window14_19.npz")
pos = {i: k for k, i in enumerate(feat["ids"])}
X = {s: feat["X"][j][[pos[i] for i in ids]].astype(np.float32) for j, s in enumerate(feat["sites"])}
reqs = {r["id"]: r for r in map(json.loads, open(DATA / "requests.jsonl"))}
TXT = {
    "text_note": [l["note_text"] or "" for l in labels],
    "text_full": ["\n".join((m.get("content") or "") for m in reqs[i]["request"]["messages"]) for i in ids],
}
rng = np.random.default_rng(SEED)


# --------------------------------------------------------------------------------------------- metrics
def strat_auc(yy, ss, strata):
    """AUROC over positive/negative pairs that share a stratum (ties count one half)."""
    num = den = 0.0
    for g in np.unique(strata):
        m = strata == g
        p, n = ss[m][yy[m] == 1], ss[m][yy[m] == 0]
        if len(p) and len(n):
            num += (p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()
            den += len(p) * len(n)
    return num / den if den else np.nan


def within_spearman(rr, ss, strata):
    """Mean Spearman correlation inside each stratum (strata with a constant target are skipped)."""
    vals = []
    for g in np.unique(strata):
        m = strata == g
        if m.sum() > 2 and np.ptp(rr[m]) > 0 and np.ptp(ss[m]) > 0:
            vals.append(spearmanr(ss[m], rr[m]).statistic)
    return float(np.mean(vals)) if vals else np.nan


def boot(fn, mask, B=N_BOOT):
    """Case-resampled bootstrap of fn(idx) over the notes in mask; returns (low, high) 95% interval."""
    uc = np.unique(cases[mask])
    by = {c: np.where(mask & (cases == c))[0] for c in uc}
    v = []
    for _ in range(B):
        idx = np.concatenate([by[c] for c in rng.choice(uc, len(uc))])
        r = fn(idx)
        if r == r:
            v.append(r)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


# ----------------------------------------------------------------------------------------------- models
def binomial_xy(mask):
    """Duplicate each note into a positive and a negative copy weighted by its fire rate."""
    idx = np.where(mask)[0]
    rows = np.concatenate([idx, idx])
    yy = np.concatenate([np.ones(len(idx), int), np.zeros(len(idx), int)])
    w = np.concatenate([rate[idx], 1 - rate[idx]])
    keep = w > 0
    return rows[keep], yy[keep], w[keep]


def fit_probe(site, train_mask, C, target):
    model = P.probe_model(C)
    if target == "binary":
        model.fit(X[site][train_mask], y[train_mask])
    else:
        rows, yy, w = binomial_xy(train_mask)
        model.fit(X[site][rows], yy, logisticregression__sample_weight=w)
    return lambda m: model.predict_proba(X[site][m])[:, 1]


def fit_text(kind, train_mask, C, target):
    txt = TXT[kind]
    model = P.text_model(C)
    if target == "binary":
        tri = np.where(train_mask)[0]
        model.fit([txt[i] for i in tri], y[train_mask])
    else:
        rows, yy, w = binomial_xy(train_mask)
        model.fit([txt[i] for i in rows], yy, logisticregression__sample_weight=w)
    return lambda m: model.predict_proba([txt[i] for i in np.where(m)[0]])[:, 1]


def fit_template(train_mask, target):
    tgt = y if target == "binary" else rate
    means = {t: tgt[train_mask & (tmpl == t)].mean() for t in np.unique(tmpl[train_mask])}
    glob = tgt[train_mask].mean()
    return lambda m: np.array([means.get(t, glob) for t in tmpl[m]])


def nested_C(site, train_mask, target, n_inner=4):
    """Pick the probe's C by GroupKFold over cases inside the training notes only."""
    tri = np.where(train_mask)[0]
    gkf = GroupKFold(n_splits=min(n_inner, len(np.unique(cases[tri]))))
    best, best_score = None, -np.inf
    for C in C_GRID:
        oof = np.full(len(ids), np.nan)
        for a, b in gkf.split(tri, groups=cases[tri]):
            mtr = np.zeros(len(ids), bool); mtr[tri[a]] = True
            mva = np.zeros(len(ids), bool); mva[tri[b]] = True
            oof[mva] = fit_probe(site, mtr, C, target)(mva)
        sc = (within_spearman(rate[train_mask], oof[train_mask], tmpl[train_mask]) if target == "rate"
              else strat_auc(y[train_mask], oof[train_mask], tmpl[train_mask]))
        if sc > best_score:
            best, best_score = C, sc
    return best


METHODS = ["probe_mre", "probe_note", "text_full", "text_note", "template"]
SITE = {"probe_mre": "memory_result_end", "probe_note": "entry0.occ0.last"}


def out_of_fold(population, groups, n_splits, target, label):
    """Out-of-fold scores for every method; folds hold out whole groups (cases or domains)."""
    t0 = time.time()
    pop = np.where(population)[0]
    oof = {k: np.full(len(ids), np.nan) for k in METHODS}
    chosen = []
    fold = np.full(len(ids), -1)
    splitter = GroupKFold(n_splits=n_splits)
    for f, (a, b) in enumerate(splitter.split(pop, groups=groups[pop])):
        mtr = np.zeros(len(ids), bool); mtr[pop[a]] = True
        mva = np.zeros(len(ids), bool); mva[pop[b]] = True
        fold[mva] = f
        for k in METHODS:
            if k.startswith("probe"):
                C = nested_C(SITE[k], mtr, target)
                chosen.append((k, C))
                oof[k][mva] = fit_probe(SITE[k], mtr, C, target)(mva)
            elif k.startswith("text"):
                oof[k][mva] = fit_text(k, mtr, TEXT_C, target)(mva)
            else:
                oof[k][mva] = fit_template(mtr, target)(mva)
    print(f"  {label}: {n_splits} folds in {time.time() - t0:.0f} s", flush=True)
    oof["fold"] = fold
    np.savez_compressed(RES / f"phase2_oof_{label.replace(' ', '_')}.npz", ids=ids, **oof)
    return oof, chosen


def fold_of(oof):
    return oof["fold"]


def summarize(oof, population, label):
    m = population
    res = {"n_notes": int(m.sum()), "n_fired": int(y[m].sum()), "n_cases": int(len(np.unique(cases[m]))),
           "n_domains": int(len(np.unique(dom[m]))), "methods": {}}
    for k in METHODS:
        s = oof[k]
        r = {
            "auroc_overall": round(float(roc_auc_score(y[m], s[m])), 3),
            "auroc_within_wording": round(float(strat_auc(y[m], s[m], tmpl[m])), 3),
            "auroc_within_wording_ci": boot(lambda i, s=s: strat_auc(y[i], s[i], tmpl[i]), m),
            # Same wording AND same fold: pairs scored by one fitted model, so fold-to-fold shifts in the
            # intercept cannot help or hurt (pooled out-of-fold scores penalize the template baseline otherwise).
            "auroc_within_wording_and_fold": round(float(strat_auc(y[m], s[m], tmpl[m] * 1000 + fold_of(oof)[m])), 3),
            "auroc_within_wording_and_fold_ci": boot(lambda i, s=s: strat_auc(y[i], s[i], tmpl[i] * 1000 + fold_of(oof)[i]), m),
            "spearman_rate_overall": round(float(spearmanr(s[m], rate[m]).statistic), 3),
            "spearman_rate_within_wording": round(within_spearman(rate[m], s[m], tmpl[m]), 3),
            "spearman_rate_within_wording_ci": boot(lambda i, s=s: within_spearman(rate[i], s[i], tmpl[i]), m),
        }
        res["methods"][k] = r
    diffs = {}
    for a in ["probe_mre", "probe_note"]:
        for b in ["text_full", "text_note"]:
            sa, sb = oof[a], oof[b]
            d_auc = strat_auc(y[m], sa[m], tmpl[m]) - strat_auc(y[m], sb[m], tmpl[m])
            d_rho = within_spearman(rate[m], sa[m], tmpl[m]) - within_spearman(rate[m], sb[m], tmpl[m])
            diffs[f"{a}_minus_{b}"] = {
                "auroc_within_wording": round(float(d_auc), 3),
                "auroc_within_wording_ci": boot(lambda i: strat_auc(y[i], sa[i], tmpl[i]) - strat_auc(y[i], sb[i], tmpl[i]), m),
                "spearman_within_wording": round(float(d_rho), 3),
                "spearman_within_wording_ci": boot(lambda i: within_spearman(rate[i], sa[i], tmpl[i]) - within_spearman(rate[i], sb[i], tmpl[i]), m),
            }
    res["paired_differences"] = diffs
    print(f"  {label}: " + ", ".join(f"{k} {v['auroc_within_wording']:.3f}" for k, v in res["methods"].items()), flush=True)
    return res


# -------------------------------------------------------------------------------------------- analyses
out = {"notes": {
    "fire_rate": "rate = fired_any / 8; binary label fired = rate >= 0.5",
    "text_C": TEXT_C, "probe_C": "nested GroupKFold(4) by case inside each training fold, grid " + str(C_GRID),
    "bootstrap": f"{N_BOOT} case resamples, 95% percentile intervals",
}}
fire_rate_by_template = {int(t): round(float(rate[live & (tmpl == t)].mean()), 3) for t in range(6)}
out["fire_rate_by_template_mean"] = fire_rate_by_template
out["binary_vs_rate"] = {
    "notes_not_fired_but_fired_at_least_once": int(((rate < 0.5) & (rate > 0) & live).sum()),
    "notes_not_fired": int(((rate < 0.5) & live).sum()),
}

print("A. leave-one-domain-out, all live notes, binary target (models trained on the binary label)")
oofA, cA = out_of_fold(live, dom, len(np.unique(dom[live])), "binary", "LODO binary")
out["A_lodo_binary"] = summarize(oofA, live, "LODO binary")
out["A_lodo_binary"]["probe_C_chosen"] = {k: sorted(set(c for kk, c in cA if kk == k)) for k in SITE}
per_dom = {}
for d in np.unique(dom[live]):
    m = live & (dom == d)
    if len(np.unique(y[m])) == 2:
        per_dom[d] = {k: round(float(roc_auc_score(y[m], oofA[k][m])), 3) for k in METHODS}
        per_dom[d]["n"] = int(m.sum())
out["A_lodo_binary"]["per_domain_auroc"] = per_dom
macro = {k: round(float(np.mean([v[k] for v in per_dom.values()])), 3) for k in METHODS}
out["A_lodo_binary"]["macro_auroc_over_domains"] = macro

print("B. fire rate as the primary label: held-out cases, all live notes, models trained on the fire rate")
oofB, cB = out_of_fold(live, cases, 5, "rate", "cases rate")
out["B_heldout_cases_rate"] = summarize(oofB, live, "cases rate")
out["B_heldout_cases_rate"]["probe_C_chosen"] = {k: sorted(set(c for kk, c in cB if kk == k)) for k in SITE}

print("B2. same held-out-case folds and all live notes, but models trained on the binary label (separates the effect of the target from the effect of the larger sample)")
oofB2, cB2 = out_of_fold(live, cases, 5, "binary", "cases binary")
out["B2_heldout_cases_binary"] = summarize(oofB2, live, "cases binary")

print("C. replication of the phase 1 control with nested C: training split only, held-out cases, binary")
trl = sets["train"] & live
oofC, cC = out_of_fold(trl, cases, 5, "binary", "train split nested")
out["C_train_split_nested"] = summarize(oofC, trl, "train split nested")
out["C_train_split_nested"]["probe_C_chosen"] = {k: sorted(set(c for kk, c in cC if kk == k)) for k in SITE}

print("D. pre-registered splits with the fire-rate target (train on the training split, score E1 and E2)")
D = {}
for k in METHODS:
    if k.startswith("probe"):
        C = nested_C(SITE[k], trl, "rate")
        f = fit_probe(SITE[k], trl, C, "rate")
    elif k.startswith("text"):
        f = fit_text(k, trl, TEXT_C, "rate")
    else:
        f = fit_template(trl, "rate")
    for e in ["E1_new_templates", "E2_new_domains"]:
        m = sets[e] & live
        s = np.full(len(ids), np.nan); s[m] = f(m)
        D.setdefault(e, {})[k] = {
            "spearman_rate_overall": round(float(spearmanr(s[m], rate[m]).statistic), 3),
            "spearman_rate_within_wording": round(within_spearman(rate[m], s[m], tmpl[m]), 3),
            "auroc_binary_overall": round(float(roc_auc_score(y[m], s[m])), 3),
        }
out["D_preregistered_rate"] = D

RES.mkdir(parents=True, exist_ok=True)
(RES / "phase2_analysis.json").write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------------------------- report
def table(sec):
    lines = ["| Method | AUROC overall | AUROC within wording [95% CI] | within wording and fold [95% CI] | Spearman with fire rate, within wording [95% CI] |",
             "| --- | --- | --- | --- | --- |"]
    for k, r in sec["methods"].items():
        lines.append(f"| {k} | {r['auroc_overall']:.3f} | {r['auroc_within_wording']:.3f} {r['auroc_within_wording_ci']} | "
                     f"{r['auroc_within_wording_and_fold']:.3f} {r['auroc_within_wording_and_fold_ci']} | "
                     f"{r['spearman_rate_within_wording']:.3f} {r['spearman_rate_within_wording_ci']} |")
    lines += ["", "| Paired difference | AUROC within wording [95% CI] | Spearman within wording [95% CI] |", "| --- | --- | --- |"]
    for k, r in sec["paired_differences"].items():
        lines.append(f"| {k} | {r['auroc_within_wording']:+.3f} {r['auroc_within_wording_ci']} | "
                     f"{r['spearman_within_wording']:+.3f} {r['spearman_within_wording_ci']} |")
    return "\n".join(lines)


A, B, B2, C_ = out["A_lodo_binary"], out["B_heldout_cases_rate"], out["B2_heldout_cases_binary"], out["C_train_split_nested"]
md = [
    "# Phase 2 analyses on cached features (exploratory)", "",
    "Script: `interp/probe_live_inert/phase2_analysis.py`. Raw numbers: `phase2_analysis.json`.",
    "Fire rate = fired_any / 8. Binary label = fire rate at least 0.5. 95% CIs resample whole cases.", "",
    f"Of {out['binary_vs_rate']['notes_not_fired']} notes labeled not fired, "
    f"{out['binary_vs_rate']['notes_not_fired_but_fired_at_least_once']} fired in at least one rollout.", "",
    f"## A. Leave one domain out, all live notes ({A['n_notes']} notes, {A['n_domains']} domains, {A['n_fired']} fired)", "",
    table(A), "",
    "Macro AUROC over domains with both classes: " + ", ".join(f"{k} {v:.3f}" for k, v in A["macro_auroc_over_domains"].items()), "",
    f"## B. Fire rate as the primary label, 5 fold held-out cases, all live notes ({B['n_notes']} notes)", "",
    table(B), "",
    f"## B2. Same folds as B, models trained on the binary label ({B2['n_notes']} notes)", "",
    table(B2), "",
    f"## C. Phase 1 control with nested C, training split only ({C_['n_notes']} notes)", "",
    table(C_), "",
    "## D. Pre-registered splits, models trained on the fire rate", "",
    "| Split | Method | Spearman overall | Spearman within wording | AUROC (binary) |", "| --- | --- | --- | --- | --- |",
]
for e, v in D.items():
    for k, r in v.items():
        md.append(f"| {e} | {k} | {r['spearman_rate_overall']:.3f} | {r['spearman_rate_within_wording']:.3f} | {r['auroc_binary_overall']:.3f} |")
(RES / "PHASE2_ANALYSIS.md").write_text("\n".join(md) + "\n")
print("wrote", RES / "phase2_analysis.json", "and PHASE2_ANALYSIS.md")
