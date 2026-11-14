"""CPU step for the attention baselines, on the features written by attention_capture.py.

Methods (each follows its paper as closely as our data allows; deviations are listed in the report)
  AT          Attention Tracker (Hung et al. 2024, arXiv 2411.00348). Heads chosen on the 30-sentence calibration
              set (keep head if (mu_normal - k sigma_normal) - (mu_attack + k sigma_attack) > 0, k = 4, relaxed to
              k = 2, 1, 0 and finally the top 10 heads by margin if none survive). Focus score FS = mean attention
              from the last prompt token to the user request (the instruction) over those heads. Flag when FS is
              low, so the score is minus FS. Training free.
  AT_fit      Same focus score, heads chosen with the same rule on our training split labels (poisoned notes or notes that
              reached execution play the role of attacks).
  AttnDA      Kawasaki and Lee 2026 (arXiv 2610.28988). Feature per head FS(l,h) = A(I) / (A(I) + A(D)), I = user
              request, D = memory tool result, all 24 x 64 heads. Shrinkage diagonal LDA,
              w = ((1 - eps) S + eps I)^-1 (mu_pos - mu_neg), eps picked by grouped CV on the training notes.
  AttnDA_full The same on full-attention layers only (sliding-window layers mostly cannot see the request).
  AttnDA_mre  The same with the query at the end of the memory result (our probe's position) instead of the
              last prompt token.
  AttnDef_RF  AttentionDefense (Siska and Sankaran 2025, arXiv 2504.12321): last-layer attention from the
              generation position to every system-prompt token, per-head z-normalized (fit on training notes),
              concatenated, random forest. AttnDef_LR is a logistic regression on the same features.
  FC          The AttnDA paper's FC baseline: logistic regression, C = 0.1, on the final-norm hidden state at the
              last prompt token.
  probe_mre   Our pre-registered probe (mean of layers 14 to 19 at the end of the memory result), for reference.
  text_full   TF-IDF on the full prompt, C = 1.0, for reference.

Evaluations (both tasks: poisoned vs benign twin over all notes, and reached execution vs triggered resisted over
poisoned notes; terms as in results/probe_live_inert/TERMINOLOGY.md)
  pre-registered: fit on the training split, score E1 (new wordings) and E2 (new domains)
  grouped CV:     5 folds holding out whole cases over all notes of the task (the execution-task analysis that has
                  enough positives), with within-wording AUROC
Metrics: AUROC with case-bootstrap CI, AUROC within wording, TPR at 1% and 5% FPR.

    python interp/probe_live_inert/attention_baselines.py --attn cache/behavior_v1_attn/attn_features.npz
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

DATA = REPO / "interp/data/behavior_v1"
RES = REPO / "results/probe_live_inert"
EPS_GRID = [0.01, 0.05, 0.1, 0.3, 0.6, 0.9]
EPS0 = 1e-6


# ------------------------------------------------------------------------------------------ methods
def at_heads(mu_n, sd_n, mu_a, sd_a, ks=(4, 2, 1, 0), fallback=10):
    """Attention Tracker head rule. Returns (boolean mask [L, H], k used)."""
    for k in ks:
        keep = (mu_n - k * sd_n) - (mu_a + k * sd_a) > 0
        if keep.any():
            return keep, k
    margin = (mu_n - mu_a).ravel()
    keep = np.zeros(margin.shape, bool)
    keep[np.argsort(-margin)[:fallback]] = True
    return keep.reshape(mu_n.shape), "top10"


def at_score(A_user, heads):
    """A_user [n, L, H] attention to the instruction. Low focus means attack, so return minus focus."""
    return -A_user[:, heads].mean(1)


class ShrinkLDA:
    """Diagonal-covariance shrinkage LDA as in AttnDA: w = ((1 - eps) S_F + eps I)^-1 (mu_pos - mu_neg)."""

    def __init__(self, eps):
        self.eps = eps

    def fit(self, X, y):
        Xp, Xn = X[y == 1], X[y == 0]
        self.mu_p, self.mu_n = Xp.mean(0), Xn.mean(0)
        S = 0.5 * (Xp.var(0) + Xn.var(0))
        self.w = (self.mu_p - self.mu_n) / ((1 - self.eps) * S + self.eps + EPS0)
        self.b = -0.5 * self.w @ (self.mu_p + self.mu_n)
        return self

    def score(self, X):
        return X @ self.w + self.b


def attnda_features(seg, q, layers, si, eps0=1e-8):
    """FS(l, h) = A(I) / (A(I) + A(D)) from seg [n, Q, L, H, S] at query q over the chosen layers."""
    sub = seg[:, q][:, layers]  # [n, len(layers), H, S]
    a_i = sub[..., si["user_content"]].astype(np.float64)
    a_d = sub[..., si["tool_content"]].astype(np.float64)
    f = a_i / (a_i + a_d + eps0)
    return f.reshape(len(seg), -1)


def fit_attnda(F, y, groups):
    """Pick eps by 4-fold grouped CV AUROC on the training notes, then refit."""
    best, best_auc = EPS_GRID[0], -1
    for eps in EPS_GRID:
        aucs = []
        for a, b in GroupKFold(4).split(F, y, groups):
            if len(np.unique(y[b])) == 2 and len(np.unique(y[a])) == 2:
                aucs.append(roc_auc_score(y[b], ShrinkLDA(eps).fit(F[a], y[a]).score(F[b])))
        if aucs and np.mean(aucs) > best_auc:
            best, best_auc = eps, float(np.mean(aucs))
    return ShrinkLDA(best).fit(F, y), best


def zscore_heads(S_tr, S_te):
    """Per-head z-normalization of AttentionDefense features [n, H, n_sys], statistics from training notes."""
    mu = S_tr.mean((0, 2), keepdims=True)
    sd = S_tr.std((0, 2), keepdims=True) + 1e-8
    return ((S_tr - mu) / sd).reshape(len(S_tr), -1), ((S_te - mu) / sd).reshape(len(S_te), -1)


def run_methods(tr, te, y, groups, feats, cal, si, full_layers):
    """Fit every method on rows tr and score rows te. Returns {method: scores for te}."""
    seg, sysA, fc = feats["seg"], feats["sys"], feats["fc"]
    L = seg.shape[2]
    all_layers = list(range(L))
    out, info = {}, {}
    A_user = seg[:, 0, :, :, si["user_content"]].astype(np.float64)
    if cal is not None:
        heads, k = at_heads(*cal)
        out["AT"] = at_score(A_user[te], heads)
        info["AT_heads"] = int(heads.sum()); info["AT_k"] = k
    mu_n, sd_n = A_user[tr][y[tr] == 0].mean(0), A_user[tr][y[tr] == 0].std(0)
    mu_a, sd_a = A_user[tr][y[tr] == 1].mean(0), A_user[tr][y[tr] == 1].std(0)
    heads, k = at_heads(mu_n, sd_n, mu_a, sd_a)
    out["AT_fit"] = at_score(A_user[te], heads)
    info["AT_fit_heads"] = int(heads.sum()); info["AT_fit_k"] = k
    for name, q, layers in [("AttnDA", 0, all_layers), ("AttnDA_full", 0, full_layers), ("AttnDA_mre", 1, all_layers)]:
        F = attnda_features(seg, q, layers, si)
        model, eps = fit_attnda(F[tr], y[tr], groups[tr])
        out[name] = model.score(F[te])
        info[f"{name}_eps"] = eps
    Ztr, Zte = zscore_heads(sysA[tr, -1].astype(np.float64), sysA[te, -1].astype(np.float64))
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(Ztr, y[tr])
    out["AttnDef_RF"] = rf.predict_proba(Zte)[:, 1]
    lr = LogisticRegression(C=0.1, max_iter=5000).fit(Ztr, y[tr])
    out["AttnDef_LR"] = lr.predict_proba(Zte)[:, 1]
    fcm = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(fc[tr].astype(np.float32), y[tr])
    out["FC"] = fcm.predict_proba(fc[te].astype(np.float32))[:, 1]
    return out, info


# ------------------------------------------------------------------------------------------ metrics
def strat_auc(yy, ss, strata):
    num = den = 0.0
    for g in np.unique(strata):
        m = strata == g
        p, n = ss[m][yy[m] == 1], ss[m][yy[m] == 0]
        if len(p) and len(n):
            num += (p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()
            den += len(p) * len(n)
    return num / den if den else float("nan")


def tpr_at(yy, ss, f):
    fpr, tpr, _ = roc_curve(yy, ss)
    ok = fpr <= f
    return float(tpr[ok].max()) if ok.any() else 0.0


def boot_ci(fn, cases, rng, B=1000):
    uc = np.unique(cases)
    by = {c: np.where(cases == c)[0] for c in uc}
    v = [fn(np.concatenate([by[c] for c in rng.choice(uc, len(uc))])) for _ in range(B)]
    v = [x for x in v if x == x]
    return [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])] if v else [float("nan")] * 2


def evaluate(yy, s, cases, tmpl, rng):
    if len(np.unique(yy)) < 2:
        return {"auroc": None}
    safe = lambda i: roc_auc_score(yy[i], s[i]) if len(np.unique(yy[i])) == 2 else np.nan  # noqa: E731
    return {"auroc": round(float(roc_auc_score(yy, s)), 3), "auroc_ci": boot_ci(safe, cases, rng),
            "auroc_within_wording": round(float(strat_auc(yy, s, tmpl)), 3),
            "tpr_at_1pct_fpr": round(tpr_at(yy, s, 0.01), 3), "tpr_at_5pct_fpr": round(tpr_at(yy, s, 0.05), 3)}


# --------------------------------------------------------------------------------------------- main
def reference_inputs(labels):
    """Probe features and full-prompt text, for the reference rows."""
    ids = [l["id"] for l in labels]
    feat = np.load(RES / "features_window14_19.npz")
    pos = {i: k for k, i in enumerate(feat["ids"])}
    X = feat["X"][list(feat["sites"]).index("memory_result_end")][[pos[i] for i in ids]].astype(np.float32)
    reqs = {r["id"]: r for r in map(json.loads, open(DATA / "requests.jsonl"))}
    full = ["\n".join((m.get("content") or "") for m in reqs[i]["request"]["messages"]) for i in ids]
    return X, full


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attn", default=str(REPO / "interp/cache/behavior_v1_attn/attn_features.npz"))
    ap.add_argument("--cal", default="", help="at_calibration.json (default: next to --attn)")
    ap.add_argument("--out", default=str(RES))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    import probe as P
    labels, manifest = P.load_labels(DATA)
    sets = P.eval_sets(labels, manifest)
    ids = [l["id"] for l in labels]
    bpath = DATA / "behavior.jsonl"
    beh = {b["id"]: b for b in map(json.loads, open(bpath) if bpath.exists() else gzip.open(str(bpath) + ".gz", "rt"))}
    cases = np.array([l["case_id"] for l in labels])
    tmpl = np.array([-1 if l["template_id"] is None else int(l["template_id"]) for l in labels])

    z = np.load(args.attn)
    pos = {i: k for k, i in enumerate(z["ids"])}
    order = [pos[i] for i in ids]
    feats = {"seg": z["seg_attn"][order], "sys": z["sys_attn"][order], "fc": z["fc_last"][order]}
    si = {s: k for k, s in enumerate(z["spans"])}
    lt = list(z["layer_types"])
    full_layers = [l for l, t in enumerate(lt) if "full" in str(t)] or list(range(1, feats["seg"].shape[2], 2))
    cal_path = Path(args.cal) if args.cal else Path(args.attn).parent / "at_calibration.json"
    cal = None
    if cal_path.exists():
        c = json.load(open(cal_path))
        cal = tuple(np.array(c[k][s]) for k, s in [("normal", "mean"), ("normal", "std"), ("attack", "mean"), ("attack", "std")])

    X, full = reference_inputs(labels)
    report = {"full_attention_layers": full_layers, "tasks": {}}
    # behavior_v1 field values: framing "live" = poisoned note, "inert" = benign twin; fired_any = rollouts executed.
    for task in ["poison_vs_twin", "execution"]:
        if task == "poison_vs_twin":
            y = np.array([int(l["framing"] == "live") for l in labels])
            pop = np.array([l["framing"] in ("live", "inert") for l in labels])
        else:
            rate = np.array([beh[i]["fired_any"] / beh[i]["k"] if i in beh else 0 for i in ids])
            y = (rate >= 0.5).astype(int)
            pop = np.array([l["framing"] == "live" and l["id"] in beh for l in labels])
        T = {}
        # pre-registered splits
        tr = np.where(sets["train"] & pop)[0]
        for sp in ["E1_new_templates", "E2_new_domains"]:
            te = np.where(sets[sp] & pop)[0]
            sc, info = run_methods(tr, te, y, cases, feats, cal, si, full_layers)
            C, _ = P.pick_C(P.probe_model, X[tr], y[tr], tmpl[tr])
            sc["probe_mre"] = P.probe_model(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            sc["text_full"] = P.text_model(1.0).fit([full[i] for i in tr], y[tr]).predict_proba([full[i] for i in te])[:, 1]
            T[sp] = {"n": int(len(te)), "n_pos": int(y[te].sum()), "fit_info": info,
                     "methods": {k: evaluate(y[te], v, cases[te], tmpl[te], rng) for k, v in sc.items()}}
            print(task, sp, {k: v["auroc"] for k, v in T[sp]["methods"].items()}, flush=True)
        # grouped CV over all notes of the task
        allp = np.where(pop)[0]
        oof = {}
        for a, b in GroupKFold(5).split(allp, groups=cases[allp]):
            tr_, te_ = allp[a], allp[b]
            sc, _ = run_methods(tr_, te_, y, cases, feats, cal, si, full_layers)
            C, _ = P.pick_C(P.probe_model, X[tr_], y[tr_], tmpl[tr_])
            sc["probe_mre"] = P.probe_model(C).fit(X[tr_], y[tr_]).predict_proba(X[te_])[:, 1]
            sc["text_full"] = P.text_model(1.0).fit([full[i] for i in tr_], y[tr_]).predict_proba([full[i] for i in te_])[:, 1]
            for k, v in sc.items():
                oof.setdefault(k, np.full(len(ids), np.nan))[te_] = v
        T["grouped_cv_5fold_cases"] = {"n": int(len(allp)), "n_pos": int(y[allp].sum()),
                                       "methods": {k: evaluate(y[allp], v[allp], cases[allp], tmpl[allp], rng) for k, v in oof.items()}}
        print(task, "CV", {k: v["auroc"] for k, v in T["grouped_cv_5fold_cases"]["methods"].items()}, flush=True)
        report["tasks"][task] = T

    out = Path(args.out)
    (out / "attention_baselines.json").write_text(json.dumps(report, indent=1))
    L = ["# Attention baselines on behavior_v1", "",
         "Script: `interp/probe_live_inert/attention_baselines.py` on features from `attention_capture.py`.",
         f"Full-attention layers: {full_layers}. Methods and deviations from the papers are in the script docstring.", ""]
    for task, T in report["tasks"].items():
        for sp, r in T.items():
            L += [f"## {task}, {sp} ({r['n']} notes, {r['n_pos']} positive)", "",
                  "| Method | AUROC [95% CI] | AUROC within wording | TPR at 1% FPR | TPR at 5% FPR |", "| --- | --- | --- | --- | --- |"]
            for k, v in r["methods"].items():
                if v["auroc"] is not None:
                    L.append(f"| {k} | {v['auroc']} {v['auroc_ci']} | {v['auroc_within_wording']} | {v['tpr_at_1pct_fpr']} | {v['tpr_at_5pct_fpr']} |")
            L.append("")
    (out / "ATTENTION_BASELINES.md").write_text("\n".join(L) + "\n")
    print("wrote", out / "ATTENTION_BASELINES.md")


if __name__ == "__main__":
    main()
