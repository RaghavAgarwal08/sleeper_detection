"""CPU: the pre-registered phase 3 analysis for one model (PLAN.md section 3). Runs on the pod after capture.py.

Inputs  interp/data/phase3/<model>/{activations.npz, behavior.jsonl (or behavior_pilot.jsonl), labels.jsonl}
        and the behavior_v1 requests for the text baselines (the same text for every model).
Output  results/phase3/<model>/analysis.json and window_features.npz (small; copy these back, not activations.npz)

Analyses, all comparing only notes that share a wording, 95% CIs from 1,000 case resamples:
  primary   H1  held out cases (5 folds), all poisoned notes, execution rate target, probe C nested:
                within wording AUROC, probe (end of memory result, window mean) minus TF-IDF on the full prompt
  secondary H2  leave one domain out, same comparison
            H3  Spearman with the execution rate within wording
            H4  poisoned vs benign twin on the pre-registered splits E1 and E2 (sanity check, expected near 1.0)
  descriptive   layer sweep: the same held out case probe at every residual layer (binary target, C = 1e-2)

    python interp/phase3/analyze.py --model qwen2.5-7b [--pilot]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "interp/probe_live_inert"))
import models as M  # noqa: E402
import probe as P  # noqa: E402

DATA = Path(os.environ.get("PHASE3_DATA", REPO / "interp/data/phase3"))
OUT = Path(os.environ.get("PHASE3_OUT", REPO / "results/phase3"))
C_GRID = P.C_GRID
TEXT_C = 1.0
B = 1000


def strat_auc(y, s, strata):
    num = den = 0.0
    for g in np.unique(strata):
        m = strata == g
        p, n = s[m][y[m] == 1], s[m][y[m] == 0]
        if len(p) and len(n):
            num += (p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()
            den += len(p) * len(n)
    return num / den if den else np.nan


def within_spearman(r, s, strata):
    v = [spearmanr(s[strata == g], r[strata == g]).statistic for g in np.unique(strata)
         if (strata == g).sum() > 2 and np.ptp(r[strata == g]) > 0 and np.ptp(s[strata == g]) > 0]
    return float(np.mean(v)) if v else np.nan


class Analysis:
    def __init__(self, key: str, pilot: bool, seed: int = 0, site: str = "memory_result_end"):
        self.key = key
        d = DATA / key
        self.labels = [json.loads(l) for l in open(d / "labels.jsonl")]
        ids = [l["id"] for l in self.labels]
        bpath = d / ("behavior_pilot.jsonl" if pilot else "behavior.jsonl")
        if key == "gpt-oss-20b" and not bpath.exists():  # phase 1 labels
            src = REPO / "interp/data/behavior_v1/behavior.jsonl"
            bpath = src if src.exists() else Path(str(src) + ".gz")
        opener = gzip.open if str(bpath).endswith(".gz") else open
        beh = {b["id"]: b for b in map(json.loads, opener(bpath, "rt"))}
        z = np.load(d / "activations.npz")
        pos = {i: k for k, i in enumerate(z["ids"])}
        keep = np.array([i in pos for i in ids])
        self.labels = [l for l, k in zip(self.labels, keep) if k]
        ids = [l["id"] for l in self.labels]
        self.ids = np.array(ids)
        order = [pos[i] for i in ids]
        self.W = z["window_mean"][order]                  # [n, 2, d]
        if site != "memory_result_end":  # read the main probe at another label free position (PLAN.md section 9)
            j = [str(x) for x in z["sweep_positions"]].index(site)
            self.W = np.stack([z["sweep_window_mean"][order][:, j].astype(np.float32), self.W[:, 1]], axis=1)
        self.layers = z["layers"][order] if "layers" in z.files else None
        self.window = z["window"].tolist()
        self.poisoned = np.array([l["framing"] == "live" and l["id"] in beh for l in self.labels])
        self.rate = np.array([beh[i]["fired_any"] / beh[i]["k"] if i in beh else np.nan for i in ids])
        self.y = (np.nan_to_num(self.rate) >= 0.5).astype(int)
        self.cases = np.array([l["case_id"] for l in self.labels])
        self.tmpl = np.array([-1 if l["template_id"] is None else int(l["template_id"]) for l in self.labels])
        self.dom = np.array([l["domain"] for l in self.labels])
        reqs = {r["id"]: r for r in map(json.loads, open(REPO / "interp/data/behavior_v1/requests.jsonl"))}
        self.text = ["\n".join((m.get("content") or "") for m in reqs[i]["request"]["messages"]) for i in ids]
        self.rng = np.random.default_rng(seed)
        self.sets = P.eval_sets(self.labels, json.loads((d / "manifest.json").read_text()))

    # --------------------------------------------------------------------------------------- models
    def bin_xy(self, idx):
        rows = np.concatenate([idx, idx])
        yy = np.concatenate([np.ones(len(idx), int), np.zeros(len(idx), int)])
        w = np.concatenate([self.rate[idx], 1 - self.rate[idx]])
        k = w > 0
        return rows[k], yy[k], w[k]

    def fit_probe(self, X, tr, C, target):
        m = P.probe_model(C)
        if target == "binary":
            m.fit(X[tr], self.y[tr])
        else:
            rows, yy, w = self.bin_xy(tr)
            m.fit(X[rows], yy, logisticregression__sample_weight=w)
        return lambda te: m.predict_proba(X[te])[:, 1]

    def fit_text(self, tr, target):
        m = P.text_model(TEXT_C)
        if target == "binary":
            m.fit([self.text[i] for i in tr], self.y[tr])
        else:
            rows, yy, w = self.bin_xy(tr)
            m.fit([self.text[i] for i in rows], yy, logisticregression__sample_weight=w)
        return lambda te: m.predict_proba([self.text[i] for i in te])[:, 1]

    def nested_C(self, X, tr, target):
        best, bs = C_GRID[0], -np.inf
        for C in C_GRID:
            oof = np.full(len(self.ids), np.nan)
            for a, b in GroupKFold(4).split(tr, groups=self.cases[tr]):
                oof[tr[b]] = self.fit_probe(X, tr[a], C, target)(tr[b])
            sc = (within_spearman(self.rate[tr], oof[tr], self.tmpl[tr]) if target == "rate"
                  else strat_auc(self.y[tr], oof[tr], self.tmpl[tr]))
            if sc > bs:
                best, bs = C, sc
        return best

    def oof(self, groups, n_splits, target, methods=("probe_mre", "probe_note", "text_full")):
        pop = np.where(self.poisoned)[0]
        out = {k: np.full(len(self.ids), np.nan) for k in methods}
        for a, b in GroupKFold(n_splits).split(pop, groups=groups[pop]):
            tr, te = pop[a], pop[b]
            for k in methods:
                if k.startswith("probe"):
                    X = self.W[:, 0 if k == "probe_mre" else 1]
                    out[k][te] = self.fit_probe(X, tr, self.nested_C(X, tr, target), target)(te)
                else:
                    out[k][te] = self.fit_text(tr, target)(te)
        return out

    # -------------------------------------------------------------------------------------- metrics
    def boot(self, fn, mask):
        uc = np.unique(self.cases[mask])
        by = {c: np.where(mask & (self.cases == c))[0] for c in uc}
        v = [fn(np.concatenate([by[c] for c in self.rng.choice(uc, len(uc))])) for _ in range(B)]
        v = [x for x in v if x == x]
        return [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])] if v else [None, None]

    def summarize(self, oof):
        m = self.poisoned
        y, r, t = self.y, self.rate, self.tmpl
        res = {}
        for k, s in oof.items():
            res[k] = {"auroc_within_wording": round(float(strat_auc(y[m], s[m], t[m])), 3),
                      "auroc_within_wording_ci": self.boot(lambda i, s=s: strat_auc(y[i], s[i], t[i]), m),
                      "auroc_overall": round(float(roc_auc_score(y[m], s[m])), 3),
                      "spearman_within_wording": round(within_spearman(r[m], s[m], t[m]), 3)}
        a, b = oof["probe_mre"], oof["text_full"]
        res["probe_mre_minus_text_full"] = {
            "auroc_within_wording": round(float(strat_auc(y[m], a[m], t[m]) - strat_auc(y[m], b[m], t[m])), 3),
            "ci": self.boot(lambda i: strat_auc(y[i], a[i], t[i]) - strat_auc(y[i], b[i], t[i]), m),
            "spearman_diff": round(within_spearman(r[m], a[m], t[m]) - within_spearman(r[m], b[m], t[m]), 3),
            "spearman_diff_ci": self.boot(lambda i: within_spearman(r[i], a[i], t[i]) - within_spearman(r[i], b[i], t[i]), m)}
        return res

    def layer_sweep(self):
        if self.layers is None:
            return None
        pop = np.where(self.poisoned)[0]
        out = []
        for l in range(self.layers.shape[2]):
            X = self.layers[:, 0, l].astype(np.float32)
            s = np.full(len(self.ids), np.nan)
            for a, b in GroupKFold(5).split(pop, groups=self.cases[pop]):
                s[pop[b]] = self.fit_probe(X, pop[a], 1e-2, "binary")(pop[b])
            out.append(round(float(strat_auc(self.y[pop], s[pop], self.tmpl[pop])), 4))
        return out

    def poisoned_vs_twin(self):
        lab = np.array([l["framing"] for l in self.labels])
        y = (lab == "live").astype(int)
        pop = np.isin(lab, ["live", "inert"])
        tr = np.where(self.sets["train"] & pop)[0]
        res = {}
        Xm = self.W[:, 0]
        m = P.probe_model(1e-2).fit(Xm[tr], y[tr])
        for e in ["E1_new_templates", "E2_new_domains"]:
            te = np.where(self.sets[e] & pop)[0]
            res[e] = round(float(roc_auc_score(y[te], m.predict_proba(Xm[te])[:, 1])), 4)
        return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pilot", action="store_true", help="use the pilot behaviour labels (fewer notes, k = 4)")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--site", default="memory_result_end", help="position for the main probe (default: the pre-registered "
                    "end of the memory result; last_prompt is the PLAN.md section 9 re-test)")
    args = ap.parse_args()
    spec = M.get(args.model)
    A = Analysis(spec["key"], args.pilot, site=args.site)
    m = A.poisoned
    out = {"model": spec["hf"], "key": spec["key"], "pilot": args.pilot, "window": A.window,
           "n_poisoned": int(m.sum()), "n_reached_execution": int(A.y[m].sum()),
           "execution_rate_by_wording": {int(t): round(float(np.nanmean(A.rate[m & (A.tmpl == t)])), 3)
                                         for t in np.unique(A.tmpl[m])},
           "mean_execution_rate": round(float(np.nanmean(A.rate[m])), 3)}
    both = sum(1 for t in np.unique(A.tmpl[m]) if 0 < A.y[m & (A.tmpl == t)].sum() < (m & (A.tmpl == t)).sum())
    out["included"] = bool(A.y[m].sum() >= 50 and (m.sum() - A.y[m].sum()) >= 50 and both >= 2)
    print(json.dumps({k: out[k] for k in ["key", "n_poisoned", "n_reached_execution", "included"]}), flush=True)
    if out["included"] or args.pilot:
        out["H1_heldout_cases_rate"] = A.summarize(A.oof(A.cases, 5, "rate"))
        print("H1", out["H1_heldout_cases_rate"]["probe_mre_minus_text_full"], flush=True)
        if not args.pilot:
            n_dom = len(np.unique(A.dom[m]))
            out["H2_lodo_binary"] = A.summarize(A.oof(A.dom, n_dom, "binary"))
            print("H2", out["H2_lodo_binary"]["probe_mre_minus_text_full"], flush=True)
    out["H4_poisoned_vs_twin"] = A.poisoned_vs_twin()
    if not args.no_sweep and not args.pilot:
        out["layer_sweep_within_wording_auroc"] = A.layer_sweep()
    o = OUT / spec["key"]
    o.mkdir(parents=True, exist_ok=True)
    out["site"] = args.site
    tag = "" if args.site == "memory_result_end" else f"_{args.site}"
    (o / ("analysis_pilot.json" if args.pilot else f"analysis{tag}.json")).write_text(json.dumps(out, indent=1))
    np.savez_compressed(o / f"window_features{tag}.npz", ids=A.ids, window_mean=A.W.astype(np.float16), window=np.array(A.window))
    print("wrote", o)


if __name__ == "__main__":
    main()
