"""Step 2 (CPU): train and evaluate a linear probe for LIVE vs INERT memory notes on behavior_v1.

Question. When a memory note contains an action, can a linear probe on gpt-oss-20b's residual stream tell a
pending instruction ("still to do, handle it now") from a closed record of the same action ("finished, kept for
reference"), on note templates and tool domains it never saw in training? And does it beat a text-only baseline?

Everything below was fixed before looking at any activations (the split and window come from the dataset's
manifest.json, written before activations existed):
  * Train: split == "train" (templates 0-3, train domains). Labels: live = 1, inert = 0. "none" requests are not
    used for training; their scores are only reported.
  * Evaluation sets
      E1 new templates   held_out_mixed, templates 4-5, seen domains
      E2 new domains     held_out_mixed, templates 0-3, test domains
      E3 both new        test split, templates 4-5, test domains        <- primary
  * Primary feature: residual stream at memory_result_end, averaged over resid layers 14-19 (manifest window).
    Secondary site: entry0.occ0.last (last token of the newest memory note; in this dataset the planted note is
    always entry 0, which a real defender would not know, so this site is reported as secondary).
  * Probe: standardise + L2 logistic regression. C is chosen by leave-one-template-out CV on the train split only.
  * Baselines: TF-IDF (word 1-2 grams and char 3-5 grams) + logistic regression on the note text, chosen the same
    way; a difference-of-means direction (no fitting beyond two means); a label-shuffled probe (should be ~0.5).
  * Uncertainty: 95% bootstrap CIs resampling cases (case_id), since twins share a case. The probe-minus-text
    AUROC difference gets its own paired bootstrap CI.
  * The per-layer sweep is exploratory and is never used to pick the reported numbers.

Usage (from the repo root or interp/):
    python interp/probe_live_inert/probe.py                       # needs the cache from capture.py
    python interp/probe_live_inert/probe.py --text-only           # text baseline only, no activations needed
Outputs: results/probe_live_inert/report_<task>.json and REPORT_<task>.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
INTERP = HERE.parent
ROOT = INTERP.parent
DEFAULT_DATA = INTERP / "data" / "behavior_v1"
DEFAULT_CACHE = INTERP / "cache" / "behavior_v1_probe"
DEFAULT_OUT = ROOT / "results" / "probe_live_inert"

C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
N_BOOT = 2000
N_SHUFFLE = 20
SEED = 0


# ------------------------------------------------------------------------------------------------ data
def load_labels(data_dir: Path) -> tuple[list[dict], dict]:
    labels = [json.loads(l) for l in open(data_dir / "labels.jsonl")]
    manifest = json.loads((data_dir / "manifest.json").read_text())
    return labels, manifest


def eval_sets(labels: list[dict], manifest: dict) -> dict[str, np.ndarray]:
    pre = manifest["preregistered"]
    test_t = set(pre["test_template_ids"])
    test_d = set(pre["test_domains"])
    framing = np.array([l["framing"] for l in labels])
    split = np.array([l["split"] for l in labels])
    tmpl = np.array([-1 if l["template_id"] is None else int(l["template_id"]) for l in labels])
    dom = np.array([l["domain"] for l in labels])
    note = framing != "none"
    new_t = np.isin(tmpl, list(test_t))
    new_d = np.isin(dom, list(test_d))
    sets = {
        "train": (split == "train") & note,
        "E1_new_templates": (split == "held_out_mixed") & note & new_t & ~new_d,
        "E2_new_domains": (split == "held_out_mixed") & note & ~new_t & new_d,
        "E3_new_templates_and_domains": (split == "test") & note,
        "none_train": (split == "train") & ~note,
        "none_test": (split == "test") & ~note,
    }
    # Sanity: the split must match its documented meaning.
    assert not (sets["train"] & (new_t | new_d)).any(), "train contains held-out templates or domains"
    assert (new_t[sets["E3_new_templates_and_domains"]]).all() and (new_d[sets["E3_new_templates_and_domains"]]).all()
    covered = sets["E1_new_templates"] | sets["E2_new_domains"]
    assert covered.sum() == ((split == "held_out_mixed") & note).sum(), "held_out_mixed has an unexpected subgroup"
    return sets


def load_features(cache_dir: Path, ids: list[str], position: str) -> np.ndarray:
    """resid [n, 25, d] float32 at one named position, in the order of ids. Fails loudly on gaps."""
    index = json.loads((cache_dir / "index.json").read_text())
    missing = [i for i in ids if i not in index]
    if missing:
        sys.exit(f"{len(missing)} of {len(ids)} requests are not in the cache ({cache_dir}). Run capture.py first.")
    out = None
    for k, i in enumerate(ids):
        names = index[i]["position_names"]
        if position not in names:
            sys.exit(f"position {position} missing for request {i}")
        z = np.load(cache_dir / f"{i}.npz")["resid"][names.index(position)]
        if out is None:
            out = np.empty((len(ids),) + z.shape, dtype=np.float32)
        out[k] = z
    return out


# ---------------------------------------------------------------------------------------------- models
def probe_model(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def text_model(C: float):
    feats = make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                       TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2))
    return make_pipeline(feats, LogisticRegression(C=C, max_iter=5000))


def pick_C(make, X, y, groups) -> tuple[float, dict]:
    """Leave-one-group-out CV on the training split only."""
    cv = {}
    for C in C_GRID:
        scores = []
        for g in np.unique(groups):
            tr, va = groups != g, groups == g
            if len(np.unique(y[va])) < 2:
                continue
            m = make(C).fit(X[tr] if not isinstance(X, list) else [X[i] for i in np.where(tr)[0]], y[tr])
            Xv = X[va] if not isinstance(X, list) else [X[i] for i in np.where(va)[0]]
            scores.append(roc_auc_score(y[va], m.predict_proba(Xv)[:, 1]))
        cv[C] = float(np.mean(scores))
    best = max(cv, key=cv.get)
    return best, cv


# --------------------------------------------------------------------------------------------- metrics
def tpr_at_fpr(y, s, fpr_target):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def boot_auc(y, s, cases, rng, s2=None):
    """Case-resampled bootstrap of AUROC (or of AUROC(s) - AUROC(s2) when s2 is given)."""
    uniq = np.unique(cases)
    by_case = {c: np.where(cases == c)[0] for c in uniq}
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([by_case[c] for c in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        a = roc_auc_score(y[idx], s[idx])
        vals.append(a - roc_auc_score(y[idx], s2[idx]) if s2 is not None else a)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(y, s, cases, rng) -> dict:
    lo, hi = boot_auc(y, s, cases, rng)
    return {"n": int(len(y)), "n_live": int(y.sum()), "n_cases": int(len(np.unique(cases))),
            "auroc": float(roc_auc_score(y, s)), "auroc_ci95": [lo, hi],
            "tpr_at_5pct_fpr": tpr_at_fpr(y, s, 0.05), "tpr_at_1pct_fpr": tpr_at_fpr(y, s, 0.01)}


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(DEFAULT_DATA))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--text-only", action="store_true", help="run only the text baseline (no activations)")
    ap.add_argument("--task", choices=["live_inert", "fired"], default="live_inert",
                    help="live_inert: live vs inert note. fired: among LIVE notes, will the agent execute the action at "
                         "the trigger turn (needs behavior.jsonl from demo/sample_behavior.py; label = fired_any/k >= 0.5)")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    labels, manifest = load_labels(Path(args.dataset))
    sets = eval_sets(labels, manifest)
    lo_w, hi_w = manifest["preregistered"]["probe_window"]["resid_layers"]
    ids = np.array([l["id"] for l in labels])
    y = np.array([1 if l["framing"] == "live" else 0 for l in labels])
    task_desc = "live (1) vs inert (0) memory note"
    if args.task == "fired":
        bpath = Path(args.dataset) / "behavior.jsonl"
        if not bpath.exists():
            sys.exit(f"{bpath} not found. Sample behaviour first: .venv-tools/bin/python demo/sample_behavior.py --k 8")
        beh = {b["id"]: b for b in (json.loads(l) for l in open(bpath))}
        has = np.array([l["id"] in beh for l in labels])
        y = np.array([1 if (l["id"] in beh and beh[l["id"]]["fired_any"] / beh[l["id"]]["k"] >= 0.5) else 0
                      for l in labels])
        live = np.array([l["framing"] == "live" for l in labels])
        for k in ["train", "E1_new_templates", "E2_new_domains", "E3_new_templates_and_domains"]:
            sets[k] = sets[k] & live & has
        task_desc = "among live notes: fired at the trigger turn (fired_any/k >= 0.5) vs not"
    cases = np.array([l["case_id"] for l in labels])
    tmpl = np.array([-1 if l["template_id"] is None else int(l["template_id"]) for l in labels])
    texts = [l["note_text"] or "" for l in labels]
    tr = sets["train"]
    eval_names = [e for e in ["E1_new_templates", "E2_new_domains", "E3_new_templates_and_domains"]
                  if len(np.unique(y[sets[e]])) == 2]
    if len(np.unique(y[sets["train"]])) < 2 or not eval_names:
        sys.exit("Not enough of both classes to train and evaluate: " +
                 ", ".join(f"{k} n={int(v.sum())} pos={int(y[v].sum())}" for k, v in sets.items()))

    report = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "dataset": str(Path(args.dataset).name), "task": task_desc,
              "window_resid_layers": [lo_w, hi_w], "set_sizes": {k: int(v.sum()) for k, v in sets.items()},
              "positives": {k: int(y[v].sum()) for k, v in sets.items()}, "eval_sets": eval_names,
              "results": {}}

    # ---- text baseline (needs no activations)
    tr_idx = np.where(tr)[0]
    C_t, cv_t = pick_C(text_model, [texts[i] for i in tr_idx], y[tr], tmpl[tr])
    text = text_model(C_t).fit([texts[i] for i in tr_idx], y[tr])
    s_text = text.predict_proba(texts)[:, 1]
    report["results"]["text_tfidf"] = {"C": C_t, "train_cv_auroc": cv_t,
                                       **{e: summarize(y[sets[e]], s_text[sets[e]], cases[sets[e]], rng)
                                          for e in eval_names}}

    if not args.text_only:
        cache = Path(args.cache)
        for site in ["memory_result_end", "entry0.occ0.last"]:
            used = tr | sets["E1_new_templates"] | sets["E2_new_domains"] | sets["E3_new_templates_and_domains"]
            if site == "memory_result_end":
                used = used | (sets["none_train"] | sets["none_test"] if args.task == "live_inert" else False)  # "none" requests have no entry0 note
            sel = np.where(used)[0]
            R = np.full((len(ids),) + load_features(cache, [ids[sel[0]]], site).shape[1:], np.nan, dtype=np.float32)
            R[sel] = load_features(cache, list(ids[sel]), site)
            X = R[:, lo_w:hi_w + 1].mean(axis=1)  # pre-registered window mean

            C_p, cv_p = pick_C(probe_model, X[tr], y[tr], tmpl[tr])
            probe = probe_model(C_p).fit(X[tr], y[tr])
            s_probe = np.full(len(ids), np.nan)
            s_probe[sel] = probe.predict_proba(X[sel])[:, 1]

            # difference-of-means direction (train only)
            d = X[tr & (y == 1)].mean(0) - X[tr & (y == 0)].mean(0)
            s_dom = np.full(len(ids), np.nan)
            s_dom[sel] = X[sel] @ (d / np.linalg.norm(d))

            res = {"C": C_p, "train_cv_auroc": cv_p}
            for e in eval_names:
                m = sets[e]
                res[e] = summarize(y[m], s_probe[m], cases[m], rng)
                res[e]["diff_of_means_auroc"] = float(roc_auc_score(y[m], s_dom[m]))
                lo, hi = boot_auc(y[m], s_probe[m], cases[m], rng, s2=s_text[m])
                res[e]["minus_text_auroc"] = float(roc_auc_score(y[m], s_probe[m]) - roc_auc_score(y[m], s_text[m]))
                res[e]["minus_text_ci95"] = [lo, hi]

            # control: label-shuffled probe (should be ~0.5 on E3)
            shuf = []
            for _ in range(N_SHUFFLE):
                yp = rng.permutation(y[tr])
                shuf.append(roc_auc_score(y[sets[eval_names[-1]]],
                                          probe_model(C_p).fit(X[tr], yp).predict_proba(X[sets[eval_names[-1]]])[:, 1]))
            res["shuffled_label_control_E3_auroc_mean"] = float(np.mean(shuf))
            res["shuffled_label_control_E3_auroc_max"] = float(np.max(shuf))

            if site == "memory_result_end" and args.task == "live_inert":
                res["none_requests_mean_live_prob"] = {
                    "train": float(np.nanmean(s_probe[sets["none_train"]])),
                    "test": float(np.nanmean(s_probe[sets["none_test"]]))}

            # exploratory per-layer sweep on E3 (not used to choose anything)
            sweep = {}
            for L in range(R.shape[1]):
                XL = R[:, L]
                mL = probe_model(C_p).fit(XL[tr], y[tr])
                m3 = sets[eval_names[-1]]
                sweep[L] = float(roc_auc_score(y[m3], mL.predict_proba(XL[m3])[:, 1]))
            res["exploratory_per_layer_E3_auroc"] = sweep
            report["results"][f"probe_{site}"] = res
            del R

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task}{'_text_only' if args.text_only else ''}"
    (out / f"report_{stem}.json").write_text(json.dumps(report, indent=1))
    md = render(report)
    (out / f"REPORT_{stem}.md").write_text(md)
    print(md)


def render(rep: dict) -> str:
    def f(x):
        return f"{x:.3f}"
    lines = [f"# {'Fired' if rep['task'].startswith('among live') else 'Live vs inert'} probe on {rep['dataset']}", "",
             f"Created {rep['created']}. Task: {rep['task']}. Window: resid layers "
             f"{rep['window_resid_layers'][0]} to {rep['window_resid_layers'][1]} (pre-registered).", "",
             "Set sizes (positives): " + ", ".join(f"{k} {v} ({rep['positives'][k]})" for k, v in rep["set_sizes"].items()), "",
             "| Method | Eval set | AUROC [95% CI] | TPR at 5% FPR | Probe minus text [95% CI] |",
             "| --- | --- | --- | --- | --- |"]
    for name, r in rep["results"].items():
        for e in rep["eval_sets"]:
            x = r[e]
            diff = (f"{x['minus_text_auroc']:+.3f} [{x['minus_text_ci95'][0]:+.3f}, {x['minus_text_ci95'][1]:+.3f}]"
                    if "minus_text_auroc" in x else "")
            lines.append(f"| {name} | {e} | {f(x['auroc'])} [{f(x['auroc_ci95'][0])}, {f(x['auroc_ci95'][1])}] "
                         f"| {f(x['tpr_at_5pct_fpr'])} | {diff} |")
    for name, r in rep["results"].items():
        if "shuffled_label_control_E3_auroc_mean" in r:
            lines += ["", f"**{name} controls.** Shuffled label probe on E3, mean AUROC "
                      f"{f(r['shuffled_label_control_E3_auroc_mean'])} (max {f(r['shuffled_label_control_E3_auroc_max'])}). "
                      f"Difference of means on {rep['eval_sets'][-1]} {f(r[rep['eval_sets'][-1]]['diff_of_means_auroc'])}. "
                      f"Chosen C {r['C']}."]
            if "none_requests_mean_live_prob" in r:
                n = r["none_requests_mean_live_prob"]
                lines.append(f"No note requests, mean live probability: train {f(n['train'])}, test {f(n['test'])}.")
            sw = r["exploratory_per_layer_E3_auroc"]
            lines.append("Exploratory per layer AUROC on E3: " + ", ".join(f"L{k} {v:.2f}" for k, v in sw.items()))
    lines += ["", f"E3 has only {rep['set_sizes']['E3_new_templates_and_domains']} notes ({rep['positives']['E3_new_templates_and_domains']} positive) from 9 test domains, so its CI is wide. TPR at 1% FPR is in report.json but is"
              " not meaningful below about 100 negatives."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
