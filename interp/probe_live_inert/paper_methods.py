"""CPU analyses adapting four papers to the execution task on the cached gpt-oss-20b features (exploratory).

  1. Conditional probing (Hewitt, Ethayarajh, Liang and Manning 2021, arXiv 2109.09234). How much information do the
     activations add BEYOND a baseline B? Fit one probe on [B; activations] and one on [B; 0] with the same model
     class and report the drop in held out cross entropy, in bits per note (an estimate of conditional V-information
     I_V(activations -> execution | B)), plus within wording AUROC. Baselines B: the wording (6 way one hot), TF-IDF
     on the full prompt, and both together.
  2. Directions never trained on execution (MacDiarmid et al. 2024, "Simple probes can catch sleeper agents").
     (a) 1,000 random directions as a control (best of 1,000, as in the post); (b) the poisoned vs benign twin
     direction, trained without execution labels; (c) poisoned vs clean difference of means; (d) the difference of
     means estimator for execution itself, against the trained probe. Signs are fixed in advance (more "poisoned"
     means more execution). The generic yes/no defection probe of the post needs new forward passes (GPU, see
     interp/phase3/generic_probe.py); its direction is scored here when that file exists.
  3. Confidence based deferral (Ashok and May 2025, arXiv 2502.13329). The probe answers only when confident and
     defers otherwise. Thresholds are chosen on half of the cases (split conformal style) to reach a target accuracy,
     then coverage and accuracy are measured on the other half; 200 random case splits.
  4. Layer and position sweeps and trajectory geometry (Ashok and May; Pandey et al. 2026) need every layer and more
     positions: interp/phase3/capture.py (saves the sweep positions) then interp/phase3/sweeps.py on the pod.

Population: the 888 poisoned notes. Folds: 5 fold GroupKFold by case, the same folds as PHASE2_ANALYSIS.md section B.
Uncertainty: 1,000 case resamples, 95% percentile intervals.

    python interp/probe_live_inert/paper_methods.py          # about 20 to 30 minutes on 2 cores
Writes results/probe_live_inert/PAPER_METHODS.md and paper_methods.json.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_union
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import probe as P  # noqa: E402

DATA = REPO / "interp/data/behavior_v1"
RES = REPO / "results/probe_live_inert"
B = 1000
rng = np.random.default_rng(0)
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:5.0f} s]", *a, flush=True)


# ---------------------------------------------------------------------------------------------- data
labels, manifest = P.load_labels(DATA)
sets = P.eval_sets(labels, manifest)
ids = np.array([l["id"] for l in labels])
beh = {b["id"]: b for b in map(json.loads, gzip.open(DATA / "behavior.jsonl.gz", "rt"))}
framing = np.array([l["framing"] for l in labels])  # dataset values: live = poisoned, inert = benign twin, none = clean
poisoned = (framing == "live") & np.array([i in beh for i in ids])
rate = np.array([beh[i]["fired_any"] / beh[i]["k"] if i in beh else np.nan for i in ids])
y = (np.nan_to_num(rate) >= 0.5).astype(int)
cases = np.array([l["case_id"] for l in labels])
tmpl = np.array([-1 if l["template_id"] is None else int(l["template_id"]) for l in labels])
dom = np.array([l["domain"] for l in labels])
feat = np.load(RES / "features_window14_19.npz")
pos = {i: k for k, i in enumerate(feat["ids"])}
order = [pos[i] for i in ids]
X = {s: feat["X"][j][order].astype(np.float64) for j, s in enumerate(feat["sites"])}
reqs = {r["id"]: r for r in map(json.loads, open(DATA / "requests.jsonl"))}
TEXT = ["\n".join((m.get("content") or "") for m in reqs[i]["request"]["messages"]) for i in ids]
POP = np.where(poisoned)[0]
FOLDS = list(GroupKFold(5).split(POP, groups=cases[POP]))
WORD = np.eye(6)[np.clip(tmpl, 0, 5)]  # wording one hot (only meaningful for notes)


def strat_auc(yy, ss, strata):
    num = den = 0.0
    for g in np.unique(strata):
        m = strata == g
        p, n = ss[m][yy[m] == 1], ss[m][yy[m] == 0]
        if len(p) and len(n):
            num += (p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()
            den += len(p) * len(n)
    return num / den if den else np.nan


def boot(fn, idx):
    uc = np.unique(cases[idx])
    by = {c: idx[cases[idx] == c] for c in uc}
    v = [fn(np.concatenate([by[c] for c in rng.choice(uc, len(uc))])) for _ in range(B)]
    v = [x for x in v if x == x]
    return [round(float(a), 3) for a in np.percentile(v, [2.5, 97.5])]


def soft_ce_bits(p, r):
    """Cross entropy in bits of predicted probability p against the execution rate r (soft label)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(r * np.log2(p) + (1 - r) * np.log2(1 - p))


# ------------------------------------------------------------------------------- 1. conditional probing
def tfidf():
    return make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                      TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2))


def design(kind, tr, te, act):
    """Feature blocks for one fold. Activations are standardized then scaled by 1/sqrt(d) so a row has norm about 1,
    like an L2 normalized TF-IDF row, and one regularization strength treats both blocks alike."""
    blocks_tr, blocks_te = [], []
    if "word" in kind:
        blocks_tr.append(sp.csr_matrix(WORD[tr])); blocks_te.append(sp.csr_matrix(WORD[te]))
    if "text" in kind:
        v = tfidf().fit([TEXT[i] for i in tr])
        blocks_tr.append(v.transform([TEXT[i] for i in tr])); blocks_te.append(v.transform([TEXT[i] for i in te]))
    if "act" in kind:
        sc = StandardScaler().fit(act[tr])
        d = np.sqrt(act.shape[1])
        blocks_tr.append(sp.csr_matrix(sc.transform(act[tr]) / d)); blocks_te.append(sp.csr_matrix(sc.transform(act[te]) / d))
    return sp.hstack(blocks_tr).tocsr(), sp.hstack(blocks_te).tocsr()


def fit_soft(Xtr, rtr, C):
    """Logistic regression on soft labels: each note is a positive with weight r and a negative with weight 1 - r."""
    n = Xtr.shape[0]
    Xd = sp.vstack([Xtr, Xtr]).tocsr()
    yd = np.r_[np.ones(n), np.zeros(n)]
    w = np.r_[rtr, 1 - rtr]
    k = w > 0
    return LogisticRegression(C=C, max_iter=5000).fit(Xd[k], yd[k], sample_weight=w[k])


C_GRID = [0.1, 1, 10, 100]


def oof_conditional(kind, act=None):
    """Out of fold probabilities for one feature set; C chosen by inner 3 fold CV on soft cross entropy."""
    out = np.full(len(ids), np.nan)
    for a, b in FOLDS:
        tr, te = POP[a], POP[b]
        best, bs = None, np.inf
        for C in C_GRID:
            ce = []
            for ia, ib in GroupKFold(3).split(tr, groups=cases[tr]):
                Xa, Xb = design(kind, tr[ia], tr[ib], act)
                m = fit_soft(Xa, rate[tr[ia]], C)
                ce.append(soft_ce_bits(m.predict_proba(Xb)[:, 1], rate[tr[ib]]).mean())
            if np.mean(ce) < bs:
                best, bs = C, np.mean(ce)
        Xa, Xb = design(kind, tr, te, act)
        out[te] = fit_soft(Xa, rate[tr], best).predict_proba(Xb)[:, 1]
    return out


log("1. conditional probing")
COND = {}
for kind, act in [("word", None), ("text", None), ("word+text", None),
                  ("act", X["memory_result_end"]), ("word+act", X["memory_result_end"]),
                  ("text+act", X["memory_result_end"]), ("word+text+act", X["memory_result_end"]),
                  ("word+text+act_note", X["entry0.occ0.last"])]:
    COND[kind] = oof_conditional(kind.replace("act_note", "act"), act)
    p = COND[kind][POP]
    log(f"   {kind:>20}: CE {soft_ce_bits(p, rate[POP]).mean():.4f} bits, within wording AUROC {strat_auc(y[POP], p, tmpl[POP]):.3f}")
H_rate = float(np.mean(-(rate[POP] * np.log2(np.clip(rate[POP].mean(), 1e-6, 1)) + (1 - rate[POP]) * np.log2(np.clip(1 - rate[POP].mean(), 1e-6, 1)))))
cond = {"entropy_bits_constant_predictor": round(H_rate, 4), "feature_sets": {}, "gains": {}}
for k, p in COND.items():
    cond["feature_sets"][k] = {"ce_bits": round(float(soft_ce_bits(p[POP], rate[POP]).mean()), 4),
                               "auroc_within_wording": round(float(strat_auc(y[POP], p[POP], tmpl[POP])), 3),
                               "auroc_overall": round(float(roc_auc_score(y[POP], p[POP])), 3)}
for base, full in [("word", "word+act"), ("text", "text+act"), ("word+text", "word+text+act"),
                   ("word+text", "word+text+act_note"), ("act", "word+text+act")]:
    pb, pf = COND[base], COND[full]
    g = lambda i, pb=pb, pf=pf: soft_ce_bits(pb[i], rate[i]).mean() - soft_ce_bits(pf[i], rate[i]).mean()  # noqa: E731
    ga = lambda i, pb=pb, pf=pf: strat_auc(y[i], pf[i], tmpl[i]) - strat_auc(y[i], pb[i], tmpl[i])  # noqa: E731
    cond["gains"][f"{full} over {base}"] = {"bits": round(float(g(POP)), 4), "bits_ci": boot(g, POP),
                                           "auroc_within_wording": round(float(ga(POP)), 3), "auroc_ci": boot(ga, POP)}
    log(f"   gain {full} over {base}: {cond['gains'][f'{full} over {base}']}")

# ---------------------------------------------------------------- 2. directions not trained on execution
log("2. directions not trained on execution")
A = X["memory_result_end"]
dirs = {}
# (b) poisoned vs benign twin probe on the training split (no execution labels)
pt = np.where(sets["train"] & np.isin(framing, ["live", "inert"]))[0]
sc = StandardScaler().fit(A[pt])
lr = LogisticRegression(C=1e-2, max_iter=5000).fit(sc.transform(A[pt]), (framing[pt] == "live").astype(int))
dirs["poisoned_vs_twin_probe"] = lr.coef_[0] / sc.scale_
# (c) poisoned minus clean, difference of means on the training split
clean_tr = np.where((framing == "none") & (np.array([l["split"] for l in labels]) == "train"))[0]
dirs["poisoned_minus_clean_means"] = A[np.where(sets["train"] & (framing == "live"))[0]].mean(0) - A[clean_tr].mean(0)
# generic yes/no defection direction from the GPU script, if it has been run
gen = RES / "generic_probe_directions.npz"
if gen.exists():
    g = np.load(gen)
    for k in g.files:
        if k.startswith("dir_"):
            dirs["generic_" + k[4:]] = g[k]


def dir_scores(v):
    return A @ (v / np.linalg.norm(v))


dir_res = {}
for k, v in dirs.items():
    s = dir_scores(v)
    dir_res[k] = {"auroc_within_wording": round(float(strat_auc(y[POP], s[POP], tmpl[POP])), 3),
                  "ci": boot(lambda i, s=s: strat_auc(y[i], s[i], tmpl[i]), POP),
                  "auroc_overall": round(float(roc_auc_score(y[POP], s[POP])), 3)}
    log(f"   {k}: {dir_res[k]}")
# (a) random directions in standardized space (sign free: best of |AUROC - 0.5| + 0.5, as a worst case control)
Z = StandardScaler().fit_transform(A[POP])
R = rng.normal(size=(1000, Z.shape[1]))
R /= np.linalg.norm(R, axis=1, keepdims=True)
S = Z @ R.T
ra = np.array([strat_auc(y[POP], S[:, j], tmpl[POP]) for j in range(S.shape[1])])
rs = np.maximum(ra, 1 - ra)
dir_res["random_1000"] = {"mean_auroc_sign_free": round(float(rs.mean()), 3), "p95": round(float(np.percentile(rs, 95)), 3),
                          "best_of_1000": round(float(rs.max()), 3)}
log(f"   random: {dir_res['random_1000']}")
# (d) execution direction: difference of means (the post's estimator) vs trained probe, same held out folds
dm, lrp = np.full(len(ids), np.nan), np.full(len(ids), np.nan)
for a, b in FOLDS:
    tr, te = POP[a], POP[b]
    mu = A[tr][y[tr] == 1].mean(0) - A[tr][y[tr] == 0].mean(0)
    dm[te] = A[te] @ mu
    sc2 = StandardScaler().fit(A[tr])
    lrp[te] = LogisticRegression(C=1e-2, max_iter=5000).fit(sc2.transform(A[tr]), y[tr]).predict_proba(sc2.transform(A[te]))[:, 1]
for k, s in [("execution_diff_of_means", dm), ("execution_trained_probe_C0.01", lrp)]:
    dir_res[k] = {"auroc_within_wording": round(float(strat_auc(y[POP], s[POP], tmpl[POP])), 3),
                  "ci": boot(lambda i, s=s: strat_auc(y[i], s[i], tmpl[i]), POP)}
    log(f"   {k}: {dir_res[k]}")

# ------------------------------------------------------------------------- 3. confidence based deferral
log("3. confidence based deferral")
oof = np.load(RES / "phase2_oof_cases_rate.npz")
opos = {i: k for k, i in enumerate(oof["ids"])}
SC = {k: oof[k][[opos[i] for i in ids]] for k in ["probe_mre", "text_full"]}
SC["probe_plus_text (conditional)"] = COND["word+text+act"]
ucases = np.unique(cases[POP])
defer = {}
for name, s in SC.items():
    pred = (s >= 0.5).astype(int)
    conf = np.maximum(s, 1 - s)
    curves = {}
    for target in [0.90, 0.95, 0.98]:
        cov, acc = [], []
        for rep in range(200):
            rr = np.random.default_rng(rep)
            cal_c = set(rr.choice(ucases, len(ucases) // 2, replace=False))
            cal = POP[np.isin(cases[POP], list(cal_c))]
            tst = POP[~np.isin(cases[POP], list(cal_c))]
            # smallest confidence threshold whose calibration accuracy reaches the target
            thr = 1.01
            for t in np.unique(conf[cal]):
                m = conf[cal] >= t
                if m.sum() >= 10 and (pred[cal][m] == y[cal][m]).mean() >= target:
                    thr = t
                    break
            m = conf[tst] >= thr
            cov.append(m.mean())
            acc.append((pred[tst][m] == y[tst][m]).mean() if m.any() else np.nan)
        curves[f"target_{target}"] = {"coverage": round(float(np.mean(cov)), 3), "accuracy": round(float(np.nanmean(acc)), 3),
                                      "coverage_ci": [round(float(x), 3) for x in np.percentile(cov, [2.5, 97.5])]}
    defer[name] = curves
    log(f"   {name}: {curves}")
# within one wording, where wording cannot help: wording 3 (62% mean execution rate, both outcomes common)
w3 = {}
for name, s in SC.items():
    m = POP[tmpl[POP] == 3]
    conf = np.maximum(s[m], 1 - s[m])
    pred = (s[m] >= 0.5).astype(int)
    order_ = np.argsort(-conf)
    for frac in [0.25, 0.5, 1.0]:
        k = max(1, int(frac * len(m)))
        sel = order_[:k]
        w3.setdefault(name, {})[f"most_confident_{int(frac * 100)}pct_accuracy"] = round(float((pred[sel] == y[m][sel]).mean()), 3)
defer["wording_3_only"] = w3
log(f"   wording 3: {w3}")

OUT = {"conditional_probing": cond, "directions": dir_res, "deferral": defer}
(RES / "paper_methods.json").write_text(json.dumps(OUT, indent=1))


# ---------------------------------------------------------------------------------------------- report
def ci(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


L = ["# Methods adapted from four papers (exploratory, CPU, cached gpt-oss-20b features)", "",
     "Script `interp/probe_live_inert/paper_methods.py`. 888 poisoned notes, 5 fold held out cases (same folds as "
     "PHASE2_ANALYSIS.md B), execution rate as the soft target, 1,000 case resamples for 95% intervals.", "",
     "## 1. Conditional probing (Hewitt et al. 2021)", "",
     "Each row is a logistic probe on the listed features. Lower cross entropy is better; "
     f"a constant predictor scores {cond['entropy_bits_constant_predictor']} bits.", "",
     "| Features | Cross entropy (bits per note) | AUROC within wording | AUROC overall |", "| --- | --- | --- | --- |"]
for k, r in cond["feature_sets"].items():
    L.append(f"| {k} | {r['ce_bits']:.4f} | {r['auroc_within_wording']:.3f} | {r['auroc_overall']:.3f} |")
L += ["", "| Information added | Bits per note [95% CI] | AUROC within wording [95% CI] |", "| --- | --- | --- |"]
for k, r in cond["gains"].items():
    L.append(f"| {k} | {r['bits']:+.4f} {ci(r['bits_ci'])} | {r['auroc_within_wording']:+.3f} {ci(r['auroc_ci'])} |")
L += ["", "`act` = activations at the end of the memory result (mean of layers 14 to 19); `act_note` = at the last "
      "token of the note; `word` = wording one hot; `text` = TF-IDF on the full prompt. A positive gain with an "
      "interval above zero means the activations carry information about execution that the baseline does not."
      " The wording only model gives every note of a wording the same score within a fold, so its within wording AUROC (and the +0.49 gain over it) only reflects fold to fold shifts; read its bits, not its AUROC.", "",
      "## 2. Directions never trained on execution (MacDiarmid et al. 2024)", "",
      "| Direction | AUROC within wording [95% CI] | AUROC overall |", "| --- | --- | --- |"]
for k, r in dir_res.items():
    if "ci" in r:
        L.append(f"| {k} | {r['auroc_within_wording']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | {r.get('auroc_overall', '')} |")
rr_ = dir_res["random_1000"]
L += ["", f"1,000 random directions (sign chosen after the fact, so this is a generous control): mean "
      f"{rr_['mean_auroc_sign_free']}, 95th percentile {rr_['p95']}, best {rr_['best_of_1000']}.", "",
      "## 3. Confidence based deferral (Ashok and May 2025)", "",
      "The detector answers only when its confidence clears a threshold chosen on half the cases to reach the target "
      "accuracy; coverage = share of notes it answers on the other half (200 random splits).", "",
      "| Detector | Target 90%: coverage / accuracy | Target 95% | Target 98% |", "| --- | --- | --- | --- |"]
for k, c in defer.items():
    if k == "wording_3_only":
        continue
    L.append(f"| {k} | " + " | ".join(f"{c[t]['coverage']:.2f} / {c[t]['accuracy']:.3f}" for t in ["target_0.9", "target_0.95", "target_0.98"]) + " |")
L += ["", "Within wording 3 only (where wording cannot help), accuracy of the most confident predictions:", "",
      "| Detector | Top 25% | Top 50% | All |", "| --- | --- | --- | --- |"]
for k, c in defer["wording_3_only"].items():
    L.append(f"| {k} | {c['most_confident_25pct_accuracy']} | {c['most_confident_50pct_accuracy']} | {c['most_confident_100pct_accuracy']} |")
L += ["", "## 4. Needs the GPU", "",
      "* Generic yes/no defection probe (MacDiarmid et al.): `interp/phase3/generic_probe.py`, then rerun this script.",
      "* Position and layer sweeps (Ashok and May) and trajectory geometry (Pandey et al.): "
      "`interp/phase3/capture.py` (always saves the sweep positions), then `interp/phase3/sweeps.py`.",
      "* Function vectors (Todd et al.): head level activation patching, a later session."]
(RES / "PAPER_METHODS.md").write_text("\n".join(L) + "\n")
log("wrote", RES / "PAPER_METHODS.md")
