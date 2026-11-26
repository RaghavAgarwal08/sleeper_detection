"""H8a and H8b of PLAN.md section 10: the probe against text classifiers fine tuned on the full prompt (CPU, minutes).

    python interp/probe_live_inert/finetuned_baseline_analysis.py
Reads results/probe_live_inert/ft_text_*.npz (from runpod/finetune_text_baseline.py), text_baselines_oof.npz and
features_window14_19.npz. Writes FINETUNED_BASELINE.md and finetuned_baseline.json in results/probe_live_inert.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = REPO / "results/probe_live_inert"
F = json.load(open(RES / "folds_heldout_cases.json"))
ids, fold, cases = np.array(F["ids"]), np.array(F["fold"]), np.array(F["case"])
tmpl, rate = np.array(F["template"]), np.array(F["rate"])
y = (rate >= 0.5).astype(int)
N = len(ids)
T = np.load(RES / "text_baselines_oof.npz")
assert (T["ids"] == ids).all(), "text_baselines_oof.npz is not in the fold file's order"
f = np.load(RES / "features_window14_19.npz"); fp = {i: k for k, i in enumerate(f["ids"])}
G = f["X"][list(f["sites"]).index("memory_result_end")][[fp[i] for i in ids]].astype(np.float64)

FT = {}
for p in sorted(RES.glob("ft_text_*.npz")):
    z = np.load(p)
    assert (z["ids"] == ids).all(), p
    if np.isfinite(z["oof"]).all():
        FT[p.stem[8:]] = z["oof"].astype(np.float64)
    else:
        print("skipping incomplete", p.name)
if not FT:
    raise SystemExit("no complete ft_text_*.npz files")


def sauc(yy, ss, st):
    num = den = 0.0
    for g in np.unique(st):
        m = st == g; a, b = ss[m][yy[m] == 1], ss[m][yy[m] == 0]
        if len(a) and len(b):
            num += (a[:, None] > b[None]).sum() + .5 * (a[:, None] == b[None]).sum(); den += len(a) * len(b)
    return num / den


def ce(p, r):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(r * np.log2(p) + (1 - r) * np.log2(1 - p))


def fit(Xa, ra, C):
    rows = np.concatenate([np.arange(len(Xa))] * 2); yy = np.concatenate([np.ones(len(Xa), int), np.zeros(len(Xa), int)])
    w = np.concatenate([ra, 1 - ra]); k = w > 0
    return LogisticRegression(C=C, max_iter=5000).fit(Xa[rows[k]], yy[k], sample_weight=w[k])


def oof(blocks):
    """Same procedure as beyond_other_lm.py: equal weight per block, C by inner GroupKFold on cross entropy."""
    out = np.full(N, np.nan)
    for k in range(5):
        a, b = np.where(fold != k)[0], np.where(fold == k)[0]

        def mk(tr, te):
            ptr, pte = [], []
            for B in blocks:
                sc = StandardScaler().fit(B[tr]); s = 1 / np.sqrt(B.shape[1])
                ptr.append(sc.transform(B[tr]) * s); pte.append(sc.transform(B[te]) * s)
            return np.hstack(ptr), np.hstack(pte)
        best, bs = None, np.inf
        for C in [1e-2, 1e-1, 1, 10, 100]:
            v = []
            for a2, b2 in GroupKFold(4).split(a, groups=cases[a]):
                Xa, Xb = mk(a[a2], a[b2]); v.append(ce(fit(Xa, rate[a[a2]], C).predict_proba(Xb)[:, 1], rate[a[b2]]).mean())
            if np.mean(v) < bs:
                best, bs = C, np.mean(v)
        Xa, Xb = mk(a, b); out[b] = fit(Xa, rate[a], best).predict_proba(Xb)[:, 1]
    return out


rng = np.random.default_rng(0); uc = np.unique(cases); by = {c: np.where(cases == c)[0] for c in uc}
BS = [np.concatenate([by[c] for c in rng.choice(uc, len(uc))]) for _ in range(1000)]


def with_ci(fn, nd=3):
    v = [fn(i) for i in BS]
    return round(float(fn(np.arange(N))), nd), [round(float(x), nd) for x in np.percentile(v, [2.5, 97.5])]


S = {"probe_mre": T["probe_mre"], "tfidf_fixed": T["tfidf_fixed"], "lm_qwen3-8b_mean": T["lm_qwen3-8b_mean"]}
S.update({f"ft_{k}": v for k, v in FT.items()})
res = {"n": N, "n_reached": int(y.sum()), "methods": {}, "probe_minus": {}}
for k, s in S.items():
    v, c = with_ci(lambda i, s=s: sauc(y[i], s[i], tmpl[i]))
    res["methods"][k] = {"auroc_within_wording": v, "ci": c}
    if k != "probe_mre":
        v, c = with_ci(lambda i, s=s: sauc(y[i], S["probe_mre"][i], tmpl[i]) - sauc(y[i], s[i], tmpl[i]))
        res["probe_minus"][k] = {"auroc_within_wording": v, "ci": c}
    print(k, res["methods"][k], res["probe_minus"].get(k), flush=True)

best = max(FT, key=lambda k: res["methods"][f"ft_{k}"]["auroc_within_wording"])
res["best_finetuned"] = best
pm = res["probe_minus"][f"ft_{best}"]
res["H8a"] = {"probe_minus_best": pm, "holds": pm["ci"][0] > 0}

base = FT[best][:, None]
P = {"ft": oof([base]), "gptoss": oof([G]), "ft+gptoss": oof([base, G])}
res["conditional"] = {}
for b_, full in [("ft", "ft+gptoss"), ("gptoss", "ft+gptoss")]:
    gb = lambda i: ce(P[b_][i], rate[i]).mean() - ce(P[full][i], rate[i]).mean()  # noqa: E731
    ga = lambda i: sauc(y[i], P[full][i], tmpl[i]) - sauc(y[i], P[b_][i], tmpl[i])  # noqa: E731
    bits, bci = with_ci(gb, 4); au, aci = with_ci(ga)
    res["conditional"][f"{full} over {b_}"] = {"bits": bits, "bits_ci": bci, "auroc": au, "auroc_ci": aci}
    print(full, "over", b_, res["conditional"][f"{full} over {b_}"], flush=True)
h = res["conditional"]["ft+gptoss over ft"]
res["H8b"] = {"gptoss_beyond_best": h, "holds": h["bits_ci"][0] > 0}
(RES / "finetuned_baseline.json").write_text(json.dumps(res, indent=1))

names = {"probe_mre": "Probe on gpt-oss-20b's state (end of memory result)", "tfidf_fixed": "TF-IDF, full prompt (pre-registered)",
         "lm_qwen3-8b_mean": "Frozen Qwen3-8B reading the text, mean pooled, linear probe",
         "ft_modernbert-large": "ModernBERT-large, fine tuned", "ft_qwen3-0.6b": "Qwen3-0.6B, fine tuned",
         "ft_qwen3-8b_lora": "Qwen3-8B, fine tuned with LoRA"}
L = ["# Fine tuned text classifiers as the full prompt baseline (PLAN.md section 10)", "",
     f"{N} poisoned notes ({int(y.sum())} reached execution), corrected labels, the probe's 5 outer folds over cases "
     "(`folds_heldout_cases.json`), execution rate target, AUROC within wording, 95% intervals over 1,000 case resamples.", "",
     "| Method | AUROC within wording [95% CI] | Probe minus method [95% CI] |", "| --- | --- | --- |"]
for k in S:
    m = res["methods"][k]; d = res["probe_minus"].get(k)
    L.append(f"| {names.get(k, k)} | {m['auroc_within_wording']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}] | "
             + (f"{d['auroc_within_wording']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]" if d else "") + " |")
L += ["", f"**H8a** (probe beats the strongest fine tuned classifier, {names.get('ft_' + best, best)}): "
      f"{pm['auroc_within_wording']:+.3f} [{pm['ci'][0]:+.3f}, {pm['ci'][1]:+.3f}], "
      f"**{'holds' if res['H8a']['holds'] else 'does not hold'}**.", "",
      "| Conditional probing | Bits per note [95% CI] | AUROC within wording [95% CI] |", "| --- | --- | --- |"]
for k, v in res["conditional"].items():
    L.append(f"| {k.replace('ft', 'fine tuned score').replace('gptoss', 'gpt-oss state')} | {v['bits']:+.4f} "
             f"[{v['bits_ci'][0]:+.4f}, {v['bits_ci'][1]:+.4f}] | {v['auroc']:+.3f} [{v['auroc_ci'][0]:+.3f}, {v['auroc_ci'][1]:+.3f}] |")
L += ["", f"**H8b** (gpt-oss-20b's state adds information beyond the strongest fine tuned classifier): "
      f"**{'holds' if res['H8b']['holds'] else 'does not hold'}**."]
(RES / "FINETUNED_BASELINE.md").write_text("\n".join(L) + "\n")
print("wrote", RES / "FINETUNED_BASELINE.md")
