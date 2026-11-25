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
args = ap.parse_args()
D = REPO / "interp/data/behavior_v1"
labels, _ = P.load_labels(D)
ids = np.array([l["id"] for l in labels])
gb = {b["id"]: b for b in map(json.loads, gzip.open(D / "behavior_namefix.jsonl.gz", "rt"))}
_q = REPO / "interp/data/phase3/qwen3-8b/behavior.jsonl"
qb = {b["id"]: b for b in map(json.loads, open(_q) if _q.exists() else gzip.open(str(_q) + ".gz", "rt"))}
POP = np.array([k for k, l in enumerate(labels) if l["framing"] == "live" and l["id"] in gb and l["id"] in qb])
RATE = {"gpt-oss-20b": np.array([gb[ids[k]]["fired_any"] / gb[ids[k]]["k"] for k in POP]),
        "qwen3-8b": np.array([qb[ids[k]]["fired_any"] / qb[ids[k]]["k"] for k in POP])}
cases = np.array([labels[k]["case_id"] for k in POP]); tmpl = np.array([int(labels[k]["template_id"]) for k in POP])


def rows(path, key, slot=None):
    z = np.load(path); pos = {i: k for k, i in enumerate(z["ids"])}
    a = z[key] if slot is None else z[key][:, slot] if z[key].ndim == 3 else z[key][slot]
    return np.asarray(a)[[pos[ids[k]] for k in POP]].astype(np.float64)


f = np.load(REPO / "results/probe_live_inert/features_window14_19.npz"); fp = {i: k for k, i in enumerate(f["ids"])}
FEAT = {"gptoss_state": f["X"][list(f["sites"]).index("memory_result_end")][[fp[ids[k]] for k in POP]].astype(np.float64),
        "qwen_state": rows(REPO / "results/phase3/qwen3-8b/window_features.npz", "window_mean", 0)}
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
       "rate_correlation": round(float(np.corrcoef(RATE["gpt-oss-20b"], RATE["qwen3-8b"])[0, 1]), 3),
       "matrix": {}, "conditional": {}}
singles = ["tfidf"] + list(FEAT)
S = {}
for m, rate in RATE.items():
    y = (rate >= 0.5).astype(int)
    for fs in singles:
        S[(fs, m)] = s = oof([fs], rate)
        res["matrix"][f"{fs} -> {m}"] = {"auroc_within_wording": round(float(sauc(y, s, tmpl)), 3),
                                        "ci": ci(lambda i, s=s: sauc(y[i], s[i], tmpl[i])), "ce_bits": round(float(ce(s, rate).mean()), 4)}
        print(fs, "->", m, res["matrix"][f"{fs} -> {m}"], flush=True)
    both = oof(["gptoss_state", "qwen_state"], rate)
    for own, other in [("gptoss_state", "qwen_state"), ("qwen_state", "gptoss_state")]:
        base = S[(other, m)]
        g = lambda i, base=base: ce(base[i], rate[i]).mean() - ce(both[i], rate[i]).mean()  # noqa: E731
        ga = lambda i, base=base: sauc(y[i], both[i], tmpl[i]) - sauc(y[i], base[i], tmpl[i])  # noqa: E731
        res["conditional"][f"{m}: {own} beyond {other}"] = {"bits": round(float(g(np.arange(len(POP)))), 4), "bits_ci": ci(g),
                                                           "auroc": round(float(ga(np.arange(len(POP)))), 3), "auroc_ci": ci(ga)}
        print(m, own, "beyond", other, res["conditional"][f"{m}: {own} beyond {other}"], flush=True)
out = REPO / "results/phase3"
(out / "cross_model.json").write_text(json.dumps(res, indent=1))
nm = {"tfidf": "TF-IDF of the prompt", "gptoss_state": "gpt-oss-20b's own state", "qwen_state": "Qwen3-8B's own state",
      "qwen_reads": "Qwen3-8B reading plain text"}
L = ["# Cross-model matrix: whose state predicts whose behaviour", "",
     f"{res['n']} poisoned notes; reached execution: gpt-oss-20b {res['reached']['gpt-oss-20b']}, Qwen3-8B {res['reached']['qwen3-8b']}; "
     f"correlation of the two models' execution rates {res['rate_correlation']}. AUROC within wording, 5 fold held out cases, 95% CI over cases.", "",
     "| Features | predicts gpt-oss-20b | predicts Qwen3-8B |", "| --- | --- | --- |"]
for fs in singles:
    a, b = res["matrix"][f"{fs} -> gpt-oss-20b"], res["matrix"][f"{fs} -> qwen3-8b"]
    L.append(f"| {nm[fs]} | {a['auroc_within_wording']:.3f} [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}] | {b['auroc_within_wording']:.3f} [{b['ci'][0]:.3f}, {b['ci'][1]:.3f}] |")
L += ["", "| Conditional probing | Bits per note [95% CI] | AUROC within wording [95% CI] |", "| --- | --- | --- |"]
for k, v in res["conditional"].items():
    L.append(f"| {k} | {v['bits']:+.4f} [{v['bits_ci'][0]:+.4f}, {v['bits_ci'][1]:+.4f}] | {v['auroc']:+.3f} [{v['auroc_ci'][0]:+.3f}, {v['auroc_ci'][1]:+.3f}] |")
(out / "CROSS_MODEL.md").write_text("\n".join(L) + "\n")
print("wrote", out / "CROSS_MODEL.md")
