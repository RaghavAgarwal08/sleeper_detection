"""Does gpt-oss's own state add information beyond another model's reading of the same text? (CPU, minutes)

Conditional probing (Hewitt et al. 2021) with another language model as the baseline: a probe on Qwen3-8B's mean
pooled reading of the prompt, versus a probe on that plus gpt-oss-20b's window activations at the end of the memory
result. Same folds, labels (corrected), target and metrics as text_baselines.py.

    python interp/probe_live_inert/beyond_other_lm.py results/probe_live_inert/lm_text_qwen3-8b.npz
Writes results/probe_live_inert/beyond_other_lm.json.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import probe as P  # noqa: E402

DATA, RES = REPO / "interp/data/behavior_v1", REPO / "results/probe_live_inert"
labels, _ = P.load_labels(DATA)
ids = np.array([l["id"] for l in labels])
beh = {b["id"]: b for b in map(json.loads, gzip.open(DATA / "behavior_namefix.jsonl.gz", "rt"))}
POP = np.array([k for k, l in enumerate(labels) if l["framing"] == "live" and l["id"] in beh])
rate = np.array([beh[ids[k]]["fired_any"] / beh[ids[k]]["k"] for k in POP]); y = (rate >= 0.5).astype(int)
cases = np.array([labels[k]["case_id"] for k in POP]); tmpl = np.array([int(labels[k]["template_id"]) for k in POP])
f = np.load(RES / "features_window14_19.npz"); fp = {i: k for k, i in enumerate(f["ids"])}
G = f["X"][list(f["sites"]).index("memory_result_end")][[fp[ids[k]] for k in POP]].astype(np.float64)
z = np.load(sys.argv[1]); zp = {i: k for k, i in enumerate(z["ids"])}
Q = z["mean"][[zp[ids[k]] for k in POP]].astype(np.float64)
Ql = z["last"][[zp[ids[k]] for k in POP]].astype(np.float64)


def ce(p, r):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(r * np.log2(p) + (1 - r) * np.log2(1 - p))


def sauc(yy, ss, st):
    num = den = 0.0
    for g in np.unique(st):
        m = st == g; a, b = ss[m][yy[m] == 1], ss[m][yy[m] == 0]
        if len(a) and len(b):
            num += (a[:, None] > b[None]).sum() + .5 * (a[:, None] == b[None]).sum(); den += len(a) * len(b)
    return num / den


def fit(Xa, ra, C):
    rows = np.concatenate([np.arange(len(Xa))] * 2); yy = np.concatenate([np.ones(len(Xa), int), np.zeros(len(Xa), int)])
    w = np.concatenate([ra, 1 - ra]); k = w > 0
    return LogisticRegression(C=C, max_iter=5000).fit(Xa[rows[k]], yy[k], sample_weight=w[k])


def oof(blocks):
    out = np.full(len(POP), np.nan)
    for a, b in GroupKFold(5).split(POP, groups=cases):
        def mk(tr, te):
            parts_tr, parts_te = [], []
            for B in blocks:
                sc = StandardScaler().fit(B[tr]); s = 1 / np.sqrt(B.shape[1])  # equal weight per block
                parts_tr.append(sc.transform(B[tr]) * s); parts_te.append(sc.transform(B[te]) * s)
            return np.hstack(parts_tr), np.hstack(parts_te)
        best, bs = None, np.inf
        for C in [1e-2, 1e-1, 1, 10, 100]:
            v = []
            for a2, b2 in GroupKFold(4).split(a, groups=cases[a]):
                Xa, Xb = mk(a[a2], a[b2]); v.append(ce(fit(Xa, rate[a[a2]], C).predict_proba(Xb)[:, 1], rate[a[b2]]).mean())
            if np.mean(v) < bs:
                best, bs = C, np.mean(v)
        Xa, Xb = mk(a, b); out[b] = fit(Xa, rate[a], best).predict_proba(Xb)[:, 1]
    return out


S = {"qwen_mean": oof([Q]), "gptoss": oof([G]), "qwen_mean+gptoss": oof([Q, G]), "qwen_mean+qwen_last": oof([Q, Ql])}
rng = np.random.default_rng(0); uc = np.unique(cases); by = {c: np.where(cases == c)[0] for c in uc}
BS = [np.concatenate([by[c] for c in rng.choice(uc, len(uc))]) for _ in range(1000)]
res = {"other_model": str(z["model"]), "feature_sets": {}, "gains": {}}
for k, s in S.items():
    res["feature_sets"][k] = {"ce_bits": round(float(ce(s, rate).mean()), 4), "auroc_within_wording": round(float(sauc(y, s, tmpl)), 3)}
for base, full in [("qwen_mean", "qwen_mean+gptoss"), ("gptoss", "qwen_mean+gptoss"), ("qwen_mean", "qwen_mean+qwen_last")]:
    gb = lambda i: ce(S[base][i], rate[i]).mean() - ce(S[full][i], rate[i]).mean()  # noqa: E731
    ga = lambda i: sauc(y[i], S[full][i], tmpl[i]) - sauc(y[i], S[base][i], tmpl[i])  # noqa: E731
    res["gains"][f"{full} over {base}"] = {"bits": round(float(gb(np.arange(len(POP)))), 4),
                                           "bits_ci": [round(float(x), 4) for x in np.percentile([gb(i) for i in BS], [2.5, 97.5])],
                                           "auroc": round(float(ga(np.arange(len(POP)))), 3),
                                           "auroc_ci": [round(float(x), 3) for x in np.percentile([ga(i) for i in BS], [2.5, 97.5])]}
print(json.dumps(res, indent=1))
(RES / "beyond_other_lm.json").write_text(json.dumps(res, indent=1))
