"""Cross-model matrix: whose internal state predicts whose behaviour? (CPU, about 15 minutes)

For each pair (features, behaviour labels), the same probe as everywhere else (5 fold held out cases, execution rate
target with binomial weights, C by nested cross validation) scored by AUROC within wording. Then conditional probing
asks, for each model's behaviour, whether that model's own state adds information beyond the other model's state, and
the reverse. If each model's own state carries information about its own behaviour that the other model's state does
not, that is evidence of privileged self-information about susceptibility.

Feature sets
  gptoss_state   gpt-oss-20b, window mean (layers 14 to 19 of 24) at the end of the memory result, its own prompt
  qwen_state     Qwen3-8B, window mean (layers 21 to 28 of 36) at the end of the memory result, its own prompt
  qwen_reads     Qwen3-8B reading the plain prompt text, mean over tokens (runpod/text_baseline_lm.py), if available
  tfidf          word and character TF-IDF of the full prompt
Labels: gpt-oss-20b (corrected phase 1 labels) and Qwen3-8B (run_model.sh, same agent loop and judge, k = 8).

    python interp/phase3/cross_model.py [--qwen-reads path/to/lm_text_qwen3-8b.npz]
Writes results/phase3/CROSS_MODEL.md and cross_model.json.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_union
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "interp/probe_live_inert"))
import probe as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--qwen-reads", default=None)
ap.add_argument("--site", default="memory_result_end", help="read every model's state here (last_prompt: PLAN.md section 9)")
args = ap.parse_args()
D = REPO / "interp/data/behavior_v1"
labels, _ = P.load_labels(D)
ids = np.array([l["id"] for l in labels])
def load_beh(path):
    path = Path(path)
    return {b["id"]: b for b in map(json.loads, open(path) if path.exists() else gzip.open(str(path) + ".gz", "rt"))}


BEH = {"gpt-oss-20b": {b["id"]: b for b in map(json.loads, gzip.open(D / "behavior_namefix.jsonl.gz", "rt"))}}
TAG = "" if args.site == "memory_result_end" else f"_{args.site}"
OTHER = [m for m in ("qwen3-8b", "llama3.1-8b") if (REPO / f"interp/data/phase3/{m}/behavior.jsonl.gz").exists()
         and (REPO / f"results/phase3/{m}/window_features{TAG}.npz").exists()]
for m in OTHER:
    BEH[m] = load_beh(REPO / f"interp/data/phase3/{m}/behavior.jsonl")
POP = np.array([k for k, l in enumerate(labels) if l["framing"] == "live" and all(l["id"] in b for b in BEH.values())])
RATE = {m: np.array([b[ids[k]]["fired_any"] / b[ids[k]]["k"] for k in POP]) for m, b in BEH.items()}
cases = np.array([labels[k]["case_id"] for k in POP]); tmpl = np.array([int(labels[k]["template_id"]) for k in POP])


def rows(path, key, slot=None):
    z = np.load(path); pos = {i: k for k, i in enumerate(z["ids"])}
    a = z[key] if slot is None else z[key][:, slot] if z[key].ndim == 3 else z[key][slot]
    return np.asarray(a)[[pos[ids[k]] for k in POP]].astype(np.float64)


f = np.load(REPO / "results/probe_live_inert/features_window14_19.npz"); fp = {i: k for k, i in enumerate(f["ids"])}
STATE = {"gpt-oss-20b": "gptoss_state", "qwen3-8b": "qwen_state", "llama3.1-8b": "llama_state"}
if TAG:
    FEAT = {"gptoss_state": rows(REPO / f"results/phase3/gpt-oss-20b/window_features{TAG}.npz", "window_mean", 0)}
else:
    FEAT = {"gptoss_state": f["X"][list(f["sites"]).index("memory_result_end")][[fp[ids[k]] for k in POP]].astype(np.float64)}
for m in OTHER:
    FEAT[STATE[m]] = rows(REPO / f"results/phase3/{m}/window_features{TAG}.npz", "window_mean", 0)
if args.qwen_reads:
    FEAT["qwen_reads"] = rows(args.qwen_reads, "mean")
reqs = {r["id"]: r for r in map(json.loads, open(D / "requests.jsonl"))}
TEXT = ["\n".join((m.get("content") or "") for m in reqs[ids[k]]["request"]["messages"]) for k in POP]


def sauc(yy, ss, st):
    num = den = 0.0
    for g in np.unique(st):
        m = st == g; a, b = ss[m][yy[m] == 1], ss[m][yy[m] == 0]
        if len(a) and len(b):
            num += (a[:, None] > b[None]).sum() + .5 * (a[:, None] == b[None]).sum(); den += len(a) * len(b)
    return num / den if den else np.nan


def ce(p, r):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -(r * np.log2(p) + (1 - r) * np.log2(1 - p))


def fit(Xa, ra, C):
    n = Xa.shape[0]; rw = np.concatenate([np.arange(n)] * 2); yy = np.concatenate([np.ones(n, int), np.zeros(n, int)])
    w = np.concatenate([ra, 1 - ra]); k = w > 0
    return LogisticRegression(C=C, max_iter=5000).fit(Xa[rw[k]], yy[k], sample_weight=w[k])


def make(blocks, tr, te):
    if blocks == ["tfidf"]:
        v = make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                       TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2))
        return v.fit_transform([TEXT[i] for i in tr]), v.transform([TEXT[i] for i in te])
    A, B = [], []
    for b in blocks:
        X = FEAT[b]; sc = StandardScaler().fit(X[tr]); s = 1 / np.sqrt(X.shape[1])
        A.append(sc.transform(X[tr]) * s); B.append(sc.transform(X[te]) * s)
    return np.hstack(A), np.hstack(B)


def oof(blocks, rate):
    out = np.full(len(POP), np.nan)
    grid = [1.0] if blocks == ["tfidf"] else [1e-2, 1e-1, 1, 10, 100]
    for a, b in GroupKFold(5).split(POP, groups=cases):
        best, bs = grid[0], np.inf
        if len(grid) > 1:
            for C in grid:
                v = []
                for a2, b2 in GroupKFold(4).split(a, groups=cases[a]):
                    Xa, Xb = make(blocks, a[a2], a[b2]); v.append(ce(fit(Xa, rate[a[a2]], C).predict_proba(Xb)[:, 1], rate[a[b2]]).mean())
                if np.mean(v) < bs:
                    best, bs = C, np.mean(v)
        Xa, Xb = make(blocks, a, b); out[b] = fit(Xa, rate[a], best).predict_proba(Xb)[:, 1]
    return out


rng = np.random.default_rng(0); uc = np.unique(cases); by = {c: np.where(cases == c)[0] for c in uc}
BS = [np.concatenate([by[c] for c in rng.choice(uc, len(uc))]) for _ in range(1000)]
ci = lambda fn: [round(float(x), 3) for x in np.percentile([fn(i) for i in BS], [2.5, 97.5])]  # noqa: E731
res = {"n": int(len(POP)), "reached": {m: int((r >= 0.5).sum()) for m, r in RATE.items()},
       "rate_correlation": {f"{a} vs {b}": round(float(np.corrcoef(RATE[a], RATE[b])[0, 1]), 3)
                            for i, a in enumerate(RATE) for b in list(RATE)[i + 1:]},
       "matrix": {}, "conditional": {}, "H7": {}}
singles = ["tfidf"] + list(FEAT)
S = {}
for m, rate in RATE.items():
    y = (rate >= 0.5).astype(int)
    for fs in singles:
        S[(fs, m)] = s = oof([fs], rate)
        res["matrix"][f"{fs} -> {m}"] = {"auroc_within_wording": round(float(sauc(y, s, tmpl)), 3),
                                        "ci": ci(lambda i, s=s: sauc(y[i], s[i], tmpl[i])), "ce_bits": round(float(ce(s, rate).mean()), 4)}
        print(fs, "->", m, res["matrix"][f"{fs} -> {m}"], flush=True)
    own = STATE[m]
    for other_m in RATE:
        if other_m == m:
            continue
        other = STATE[other_m]
        both = oof([own, other], rate)
        for a_, b_ in [(own, other), (other, own)]:
            base = S[(b_, m)]
            g = lambda i, base=base, both=both: ce(base[i], rate[i]).mean() - ce(both[i], rate[i]).mean()  # noqa: E731
            ga = lambda i, base=base, both=both: sauc(y[i], both[i], tmpl[i]) - sauc(y[i], base[i], tmpl[i])  # noqa: E731
            res["conditional"][f"{m}: {a_} beyond {b_}"] = {"bits": round(float(g(np.arange(len(POP)))), 4), "bits_ci": ci(g),
                                                           "auroc": round(float(ga(np.arange(len(POP)))), 3), "auroc_ci": ci(ga)}
            print(m, a_, "beyond", b_, res["conditional"][f"{m}: {a_} beyond {b_}"], flush=True)
        fwd, rev = res["conditional"][f"{m}: {own} beyond {other}"], res["conditional"][f"{m}: {other} beyond {own}"]
        res["H7"][f"{m} vs {other_m}"] = {"a_own_beyond_other_ci_above_zero": fwd["bits_ci"][0] > 0,
                                         "b_other_beyond_own_not_above_zero": rev["bits_ci"][0] <= 0,
                                         "holds": bool(fwd["bits_ci"][0] > 0 and rev["bits_ci"][0] <= 0)}
out = REPO / "results/phase3"
res["site"] = args.site
(out / f"cross_model{TAG}.json").write_text(json.dumps(res, indent=1))
nm = {"tfidf": "TF-IDF of the prompt", "gptoss_state": "gpt-oss-20b's state", "qwen_state": "Qwen3-8B's state",
      "llama_state": "Llama-3.1-8B's state", "qwen_reads": "Qwen3-8B reading plain text"}
L = ["# Cross-model matrix: whose state predicts whose behaviour", "",
     f"{res['n']} poisoned notes; reached execution: " + ", ".join(f"{m} {v}" for m, v in res["reached"].items()) +
     "; correlations of execution rates: " + ", ".join(f"{k} {v}" for k, v in res["rate_correlation"].items()) +
     ". AUROC within wording, 5 fold held out cases, 95% CI over cases.", "",
     "| Features | " + " | ".join(f"predicts {m}" for m in RATE) + " |", "| --- |" + " --- |" * len(RATE)]
for fs in singles:
    cells = [res["matrix"][f"{fs} -> {m}"] for m in RATE]
    L.append(f"| {nm[fs]} | " + " | ".join(f"{c['auroc_within_wording']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]" for c in cells) + " |")
L += ["", "| Conditional probing | Bits per note [95% CI] | AUROC within wording [95% CI] |", "| --- | --- | --- |"]
for k, v in res["conditional"].items():
    L.append(f"| {k} | {v['bits']:+.4f} [{v['bits_ci'][0]:+.4f}, {v['bits_ci'][1]:+.4f}] | {v['auroc']:+.3f} [{v['auroc_ci'][0]:+.3f}, {v['auroc_ci'][1]:+.3f}] |")
L += ["", "## H7 (pre-registered in PLAN.md section 7; applies to models added after gpt-oss-20b and Qwen3-8B)", "",
      "| Pair (behaviour of the first) | (a) own beyond other, interval above zero | (b) other beyond own, not above zero | Holds |",
      "| --- | --- | --- | --- |"]
for k, v in res["H7"].items():
    L.append(f"| {k} | {v['a_own_beyond_other_ci_above_zero']} | {v['b_other_beyond_own_not_above_zero']} | {v['holds']} |")
L.insert(1, f"\nStates read at: {args.site}.")
(out / f"CROSS_MODEL{TAG}.md").write_text("\n".join(L) + "\n")
print("wrote", out / f"CROSS_MODEL{TAG}.md")
