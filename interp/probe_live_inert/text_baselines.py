"""Stronger text baselines for execution prediction (CPU, about 20 to 40 minutes).

Reviewers will ask whether the probe only beats a weak text model. This script compares the execution probe with a set
of text-only and structure-only baselines on exactly the folds of PHASE2_ANALYSIS.md B (5 fold held out cases, all 888
poisoned notes, execution rate target with binomial weights), scored by AUROC within wording and paired bootstrap
differences over cases.

Baselines (all see only the prompt, never the model's activations)
  tfidf_fixed        word and character TF-IDF on the full prompt, C = 1 (the pre-registered baseline)
  tfidf_tuned        the same with C chosen by nested cross validation, as the probe's C is
  tfidf_svd_gbm      TF-IDF reduced to 256 dimensions, then gradient boosted trees (a nonlinear text model)
  conv_stats         prompt structure only, in the spirit of Kale and Vishwanath (2026): total length, memory result
                     length, note length, note occurrences, number of memory entries, the note's relative position,
                     user request length, number of messages
  tfidf_plus_stats   TF-IDF and the structure features together
A neural text baseline (a small language model reading the same prompt) needs Hugging Face downloads, which this
environment blocks; runpod/text_baseline_lm.py is the GPU version.

    python interp/probe_live_inert/text_baselines.py [--labels namefix|original]
Writes results/probe_live_inert/TEXT_BASELINES.md and text_baselines.json.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_union
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import probe as P  # noqa: E402

DATA = REPO / "interp/data/behavior_v1"
RES = REPO / "results/probe_live_inert"
C_GRID = P.C_GRID
TEXT_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]  # TF-IDF picked the top of C_GRID every fold, so extend upward
N_BOOT = 1000
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:5.0f} s]", *a, flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--labels", choices=["namefix", "original"], default="namefix")
ap.add_argument("--lm", nargs="*", default=None, help="lm_text_*.npz files from runpod/text_baseline_lm.py "
                "(default: every one in results/probe_live_inert)")
args = ap.parse_args()

labels, manifest = P.load_labels(DATA)
ids = np.array([l["id"] for l in labels])
bfile = DATA / ("behavior_namefix.jsonl.gz" if args.labels == "namefix" else "behavior.jsonl.gz")
beh = {b["id"]: b for b in map(json.loads, gzip.open(bfile, "rt"))}
POP = np.array([k for k, l in enumerate(labels) if l["framing"] == "live" and l["id"] in beh])
rate = np.array([beh[ids[k]]["fired_any"] / beh[ids[k]]["k"] for k in POP])
y = (rate >= 0.5).astype(int)
cases = np.array([labels[k]["case_id"] for k in POP])
tmpl = np.array([int(labels[k]["template_id"]) for k in POP])
reqs = {r["id"]: r for r in map(json.loads, open(DATA / "requests.jsonl"))}
text = ["\n".join((m.get("content") or "") for m in reqs[ids[k]]["request"]["messages"]) for k in POP]
feat = np.load(RES / "features_window14_19.npz")
fpos = {i: k for k, i in enumerate(feat["ids"])}
X_act = feat["X"][list(feat["sites"]).index("memory_result_end")][[fpos[ids[k]] for k in POP]].astype(np.float32)


def stats_row(k):
    r = reqs[ids[k]]
    p, n = r["positions"], len(r["input_ids"])
    ei = labels[k]["extra_note_entry_index"]
    note = next((s for s in r["entry_spans"] if s["entry_index"] == ei), None)
    occ = note["occurrences"] if note else [[p["memory_result_end"], p["memory_result_end"]]]
    msgs = r["request"]["messages"]
    user = next((m.get("content") or "" for m in msgs if m.get("role") == "user"), "")
    mem_len = p["memory_result_end"] - p["user_end"]
    return [np.log1p(n), np.log1p(mem_len), np.log1p(occ[0][1] - occ[0][0]), len(occ), len(r["entry_spans"]),
            (occ[0][0] - p["user_end"]) / max(mem_len, 1), np.log1p(len(user)), len(msgs)]


X_stats = np.array([stats_row(k) for k in POP], dtype=np.float64)
log(f"{len(POP)} poisoned notes, labels={args.labels}, reached execution {y.sum()}")


def strat_auc(yy, ss, strata):
    num = den = 0.0
    for g in np.unique(strata):
        m = strata == g
        p, n = ss[m][yy[m] == 1], ss[m][yy[m] == 0]
        if len(p) and len(n):
            num += (p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()
            den += len(p) * len(n)
    return num / den if den else np.nan


def within_spearman(rr, ss, strata):
    v = [spearmanr(ss[strata == g], rr[strata == g]).statistic for g in np.unique(strata)
         if (strata == g).sum() > 2 and np.ptp(rr[strata == g]) > 0 and np.ptp(ss[strata == g]) > 0]
    return float(np.mean(v)) if v else np.nan


def nested(make_X, tr, n_inner=4, grid=None):
    """Pick C by GroupKFold over cases inside the training notes, scored by within wording Spearman.
    Features are built once per inner split and reused for every C."""
    grid = grid or C_GRID
    oof = {C: np.full(len(POP), np.nan) for C in grid}
    for a, b in GroupKFold(n_splits=n_inner).split(tr, groups=cases[tr]):
        Xa, Xb = make_X(tr[a], tr[b])
        rr = rate[tr[a]]
        rows = np.concatenate([np.arange(len(tr[a]))] * 2)
        yy = np.concatenate([np.ones(len(tr[a]), int), np.zeros(len(tr[a]), int)])
        w = np.concatenate([rr, 1 - rr]); k = w > 0
        for C in grid:
            m = LogisticRegression(C=C, max_iter=5000).fit(Xa[rows[k]], yy[k], sample_weight=w[k])
            oof[C][tr[b]] = m.predict_proba(Xb)[:, 1]
    sc = {C: within_spearman(rate[tr], oof[C][tr], tmpl[tr]) for C in grid}
    return max(grid, key=lambda C: (np.nan_to_num(sc[C], nan=-9), -C))


def fit_predict(make_X, tr, te, C):
    Xa, Xb = make_X(tr, te)
    rr = rate[tr]
    rows = np.concatenate([np.arange(len(tr))] * 2)
    yy = np.concatenate([np.ones(len(tr), int), np.zeros(len(tr), int)])
    w = np.concatenate([rr, 1 - rr]); k = w > 0
    m = LogisticRegression(C=C, max_iter=5000).fit(Xa[rows[k]], yy[k], sample_weight=w[k])
    return m.predict_proba(Xb)[:, 1]


def tfidf_X(tr, te):
    v = make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                   TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2))
    A = v.fit_transform([text[i] for i in tr])
    return A, v.transform([text[i] for i in te])


def dense_X(M):
    def make(tr, te):
        sc = StandardScaler().fit(M[tr])
        return sc.transform(M[tr]), sc.transform(M[te])
    return make


def tfidf_stats_X(tr, te):
    A, B = tfidf_X(tr, te)
    sc = StandardScaler().fit(X_stats[tr])
    return (sparse.hstack([A, sparse.csr_matrix(sc.transform(X_stats[tr]) * 0.1)]).tocsr(),
            sparse.hstack([B, sparse.csr_matrix(sc.transform(X_stats[te]) * 0.1)]).tocsr())


def svd_gbm(tr, te):
    A, B = tfidf_X(tr, te)
    svd = TruncatedSVD(256, random_state=0).fit(A)
    Za, Zb = svd.transform(A), svd.transform(B)
    rr = rate[tr]
    Z = np.vstack([Za, Za]); yy = np.concatenate([np.ones(len(tr), int), np.zeros(len(tr), int)])
    w = np.concatenate([rr, 1 - rr]); k = w > 0
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                       random_state=0).fit(Z[k], yy[k], sample_weight=w[k])
    return m.predict_proba(Zb)[:, 1]


METHODS = ["probe_mre", "tfidf_fixed", "tfidf_tuned", "tfidf_svd_gbm", "conv_stats", "tfidf_plus_stats"]
LM = {}
for f in (args.lm if args.lm is not None else sorted(RES.glob("lm_text_*.npz"))):
    z = np.load(f)
    pos = {i: k for k, i in enumerate(z["ids"])}
    if all(ids[k] in pos for k in POP):
        for kind in ("last", "mean"):
            LM[f"lm_{Path(f).stem[8:]}_{kind}"] = z[kind][[pos[ids[k]] for k in POP]].astype(np.float32)
METHODS += sorted(LM)
oof = {k: np.full(len(POP), np.nan) for k in METHODS}
chosen = {k: [] for k in METHODS}
for f, (a, b) in enumerate(GroupKFold(n_splits=5).split(POP, groups=cases)):
    tr, te = a, b
    for k in METHODS:
        if k == "probe_mre":
            mk = dense_X(X_act)
            C = nested(mk, tr)
        elif k == "tfidf_fixed":
            mk, C = tfidf_X, 1.0
        elif k == "tfidf_tuned":
            mk = tfidf_X
            C = nested(mk, tr, grid=TEXT_GRID)
        elif k == "conv_stats":
            mk = dense_X(X_stats)
            C = nested(mk, tr)
        elif k == "tfidf_plus_stats":
            mk = tfidf_stats_X
            C = nested(mk, tr, grid=TEXT_GRID)
        elif k.startswith("lm_"):
            mk = dense_X(LM[k])
            C = nested(mk, tr)
        if k == "tfidf_svd_gbm":
            oof[k][te] = svd_gbm(tr, te)
            C = None
        else:
            oof[k][te] = fit_predict(mk, tr, te, C)
        chosen[k].append(C)
    log(f"fold {f + 1}/5 done")

rng = np.random.default_rng(0)
uc = np.unique(cases)
by = {c: np.where(cases == c)[0] for c in uc}
BS = [np.concatenate([by[c] for c in rng.choice(uc, len(uc))]) for _ in range(N_BOOT)]


def ci(fn):
    v = [fn(i) for i in BS]
    v = [x for x in v if x == x]
    return [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]


res = {"labels": args.labels, "n": int(len(POP)), "n_reached": int(y.sum()), "chosen_C": chosen, "methods": {}, "probe_minus": {}}
for k in METHODS:
    s = oof[k]
    res["methods"][k] = {"auroc_within_wording": round(float(strat_auc(y, s, tmpl)), 3),
                         "auroc_within_wording_ci": ci(lambda i, s=s: strat_auc(y[i], s[i], tmpl[i])),
                         "spearman_within_wording": round(within_spearman(rate, s, tmpl), 3)}
    log(k, res["methods"][k])
for k in METHODS[1:]:
    d = lambda i, k=k: strat_auc(y[i], oof["probe_mre"][i], tmpl[i]) - strat_auc(y[i], oof[k][i], tmpl[i])  # noqa: E731
    res["probe_minus"][k] = {"auroc_within_wording": round(float(d(np.arange(len(POP)))), 3), "ci": ci(d)}
    log("probe minus", k, res["probe_minus"][k])
tag = "" if args.labels == "namefix" else "_original_labels"
(RES / f"text_baselines{tag}.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(RES / f"text_baselines_oof{tag}.npz", ids=ids[POP], **oof)

names = {"probe_mre": "Probe (activations, end of memory result)", "tfidf_fixed": "TF-IDF, full prompt, C = 1 (pre-registered)",
         "tfidf_tuned": "TF-IDF, full prompt, nested C (0.1 to 1000)", "tfidf_svd_gbm": "TF-IDF to 256 dims, gradient boosted trees",
         "conv_stats": "Prompt structure only (8 features)", "tfidf_plus_stats": "TF-IDF and prompt structure"}
md = [f"# Stronger text baselines ({args.labels} labels)", "",
      "Script `interp/probe_live_inert/text_baselines.py`. Same folds as PHASE2_ANALYSIS.md B: 5 fold held out cases, "
      f"{len(POP)} poisoned notes ({int(y.sum())} reached execution), execution rate target, AUROC within wording, "
      "1,000 case resamples.", "",
      "| Method | AUROC within wording [95% CI] | Spearman within wording | Probe minus method [95% CI] |", "| --- | --- | --- | --- |"]
for k in METHODS:
    names.setdefault(k, f"Other model reading the text: {k[3:].rsplit('_', 1)[0]}, {k.rsplit('_', 1)[1]} token(s)")
    m = res["methods"][k]
    pm = res["probe_minus"].get(k)
    md.append(f"| {names[k]} | {m['auroc_within_wording']:.3f} [{m['auroc_within_wording_ci'][0]:.3f}, {m['auroc_within_wording_ci'][1]:.3f}] | "
              f"{m['spearman_within_wording']:.3f} | " + (f"{pm['auroc_within_wording']:+.3f} [{pm['ci'][0]:+.3f}, {pm['ci'][1]:+.3f}]" if pm else "") + " |")
md += ["", ("Rows starting \"Other model reading the text\" use features from runpod/text_baseline_lm.py: another language model "
        "reads exactly the text TF-IDF sees, averaged over the same relative depth as gpt-oss's window, at the last token or "
        "averaged over all tokens; the same probe is then fitted on those features. See beyond_other_lm.json for whether "
        "gpt-oss's own state adds information beyond another model's reading.") if LM else
       "A neural text baseline (another language model reading the same prompt) needs runpod/text_baseline_lm.py on a GPU."]
(RES / f"TEXT_BASELINES{tag}.md").write_text("\n".join(md) + "\n")
log("wrote", RES / f"TEXT_BASELINES{tag}.md")
