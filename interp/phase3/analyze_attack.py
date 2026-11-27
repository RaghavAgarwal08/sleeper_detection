"""A1, A2, transfer and A3 of PLAN.md section 12, from the outputs of attack_probe.py (CPU, minutes).

    python interp/phase3/analyze_attack.py [--dir results/phase3/attack] [--dry-run]
Writes ATTACK_REPORT.md and attack_analysis.json in that directory.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from attack_probe import fit_probe, probe_score, replace_note  # noqa: E402

SRC = REPO / "interp/data/behavior_v1"
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=str(REPO / "results/phase3/attack"))
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
D = Path(args.dir); sfx = "_dryrun" if args.dry_run else ""
F = json.load(open(D / f"final{sfx}.json")); mon = F["monitor"]; fin = F["finals"]; tau = mon["tau"]
cands = [json.loads(l) for l in open(D / f"candidates{sfx}.jsonl")]
labels = {l["id"]: l for l in map(json.loads, open(SRC / "labels.jsonl"))}
targets = [i for i in mon["targets"] if i in fin]
case = {i: labels[i]["case_id"] for i in targets}
uc = sorted(set(case.values())); byc = {c: [i for i in targets if case[i] == c] for c in uc}
rng = np.random.default_rng(0)
BS = [[i for c in rng.choice(uc, len(uc)) for i in byc[c]] for _ in range(1000)]


def ci(fn):
    v = [fn(b) for b in BS]; v = [x for x in v if x == x]
    return [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])] if v else [None, None]


ev = lambda ids, arm: float(np.mean([fin[i][arm]["evaded"] for i in ids if arm in fin[i]]))  # noqa: E731
res = {"n_targets": len(targets), "n_cases": len(uc), "monitor": mon, "arms": {}}
for arm in ["original", "P", "C"]:
    ids = [i for i in targets if arm in fin[i]]
    res["arms"][arm] = {"n": len(ids), "evasion": round(ev(ids, arm), 3), "evasion_ci": ci(lambda b: ev(b, arm)),
                        "exec_final": round(float(np.mean([fin[i][arm]["exec_final"] for i in ids])), 3),
                        "score_mre": round(float(np.mean([fin[i][arm]["score_mre"] for i in ids])), 3),
                        "flagged": round(float(np.mean([fin[i][arm]["score_mre"] >= tau["mre"] for i in ids])), 3)}
d = lambda b: ev(b, "P") - ev(b, "C")  # noqa: E731
res["A1"] = {"P_minus_C": round(d(targets), 3), "ci": ci(d)}
res["A1"]["holds"] = res["A1"]["ci"][0] is not None and res["A1"]["ci"][0] > 0

rho = {}
for i in targets:
    c = [x for x in cands if x["id"] == i and x["arm"] == "P"]
    s, e = np.array([x["score_mre"] for x in c]), np.array([x["exec_search"] for x in c])
    if len(c) >= 3 and np.ptp(s) > 0 and np.ptp(e) > 0:
        rho[i] = spearmanr(s, e).statistic
mr = lambda b: float(np.mean([rho[i] for i in b if i in rho])) if any(i in rho for i in b) else np.nan  # noqa: E731
res["A2"] = {"mean_within_note_spearman": round(mr(targets), 3) if rho else None, "ci": ci(mr), "n_notes": len(rho)}
res["A2"]["holds"] = res["A2"]["ci"][0] is not None and res["A2"]["ci"][0] > 0

# pooled trade off: execution by score bin over all P candidates
sP = np.array([x["score_mre"] for x in cands if x["arm"] == "P"]); eP = np.array([x["exec_search"] / x["k_search"] for x in cands if x["arm"] == "P"])
if len(sP):
    q = np.quantile(sP, [0, .25, .5, .75, 1])
    res["tradeoff"] = [{"score_from": round(float(q[k]), 2), "score_to": round(float(q[k + 1]), 2),
                        "mean_exec": round(float(eP[(sP >= q[k]) & (sP <= q[k + 1])].mean()), 3)} for k in range(4)]

# transfer to unattacked monitors, including TF-IDF trained here on the training split's original prompts
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_union  # noqa: E402
reqs = {r["id"]: r for r in map(json.loads, open(SRC / "requests.jsonl"))}
beh = {b["id"]: b for b in map(json.loads, gzip.open(SRC / "behavior_namefix.jsonl.gz", "rt"))}
txt = lambda req: "\n".join((m.get("content") or "") for m in req["messages"])  # noqa: E731
trn = [i for i in beh if labels[i]["framing"] == "live" and labels[i]["split"] == "train" and i in reqs]
if args.dry_run:
    trn = trn[:120]
rt = np.array([beh[i]["fired_any"] / beh[i]["k"] for i in trn]); gt = np.array([labels[i]["case_id"] for i in trn])
T = [txt(reqs[i]["request"]) for i in trn]


def tfidf_fit(texts, r):
    v = make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                   TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2))
    A = v.fit_transform(texts); n = len(texts)
    rows = np.concatenate([np.arange(n)] * 2); yy = np.concatenate([np.ones(n, int), np.zeros(n, int)])
    w = np.concatenate([r, 1 - r]); k = w > 0
    return v, LogisticRegression(C=1.0, max_iter=5000).fit(A[rows[k]], yy[k], sample_weight=w[k])


oof = np.full(len(trn), np.nan)
for a, b in GroupKFold(5).split(T, groups=gt):
    v, m = tfidf_fit([T[k] for k in a], rt[a]); oof[b] = m.decision_function(v.transform([T[k] for k in b]))
tau_tfidf = float(np.percentile(oof[rt < 0.5], 95))
v, m = tfidf_fit(T, rt)
res["transfer"] = {"tau_tfidf": tau_tfidf}
for arm in ["original", "P", "C"]:
    ids = [i for i in targets if arm in fin[i]]
    tx = [txt(replace_note(reqs[i]["request"], labels[i]["note_text"], fin[i][arm]["text"])) for i in ids]
    st = m.decision_function(v.transform(tx)) if tx else []
    for i, s_ in zip(ids, st):
        fin[i][arm]["score_tfidf"] = float(s_)
    evd = [i for i in ids if fin[i][arm]["evaded"]]
    res["transfer"][arm] = {"n_evaded": len(evd),
                            **{f"also_below_{mm}": (round(float(np.mean([fin[i][arm][f"score_{mm}"] < tau[mm] for i in evd])), 3) if evd else None)
                               for mm in ["last_prompt", "dom"]},
                            "also_below_tfidf": round(float(np.mean([fin[i][arm]["score_tfidf"] < tau_tfidf for i in evd])), 3) if evd else None,
                            "tfidf_flags_executed": round(float(np.mean([fin[i][arm]["score_tfidf"] >= tau_tfidf for i in ids if fin[i][arm]["exec_final"] >= 0.5])), 3) if ids else None}

# A3: retraining with the attacker's candidates, cross fitted by halves of the target cases
try:
    z0 = np.load(D / f"originals{sfx}.npz"); X0 = z0["X"].astype(np.float32); ids0 = list(z0["ids"])
    feats = {}
    for p in sorted(D.glob("features_dryrun.npz" if args.dry_run else "features_round*.npz")):
        z = np.load(p)
        for k, t, x in zip(z["keys"], z["texts"], z["X"]):
            feats[(str(k), str(t))] = x.astype(np.float32)
    zf = np.load(D / f"final_features{sfx}.npz"); ffeat = {str(k): x.astype(np.float32) for k, x in zip(zf["keys"], zf["X"])}
    trm = np.array([labels[i]["split"] == "train" for i in ids0])
    r0 = np.array([beh[i]["fired_any"] / beh[i]["k"] for i in ids0]); g0 = np.array([labels[i]["case_id"] for i in ids0])
    halves = [uc[0::2], uc[1::2]]
    a3 = []
    for h in range(2):
        A_cases, B_cases = set(halves[h]), set(halves[1 - h])
        add = [x for x in cands if x["arm"] == "P" and case.get(x["id"]) in A_cases]
        Xa = [feats[(f"{x['id']}|P|{x['round']}", x["text"])][0] for x in add if (f"{x['id']}|P|{x['round']}", x["text"]) in feats]
        ra = [x["exec_search"] / x["k_search"] for x in add if (f"{x['id']}|P|{x['round']}", x["text"]) in feats]
        ga = [case[x["id"]] for x in add if (f"{x['id']}|P|{x['round']}", x["text"]) in feats]
        X = np.vstack([X0[trm, 0]] + ([np.stack(Xa)] if Xa else []))
        r = np.concatenate([r0[trm], ra]); g = np.concatenate([g0[trm], ga])
        p = fit_probe(X, r, g)
        t2 = float(np.percentile(p["oof"][r < 0.5], 95))
        evB = [i for i in targets if case[i] in B_cases and "P" in fin[i] and fin[i]["P"]["evaded"]]
        caught = [float(probe_score(p, ffeat[f"{i}|P"][None, 0])[0] >= t2) for i in evB if f"{i}|P" in ffeat]
        exB = [i for i in targets if case[i] in B_cases and "original" in fin[i]]
        rec_orig = [float(probe_score(p, ffeat[f"{i}|original"][None, 0])[0] >= t2) for i in exB if f"{i}|original" in ffeat
                    and fin[i]["original"]["exec_final"] >= 0.5]
        a3.append({"train_half": h, "added_candidates": len(Xa), "tau": t2, "evaded_in_other_half": len(evB),
                   "now_flagged": round(float(np.mean(caught)), 3) if caught else None,
                   "originals_still_flagged": round(float(np.mean(rec_orig)), 3) if rec_orig else None})
    res["A3"] = a3
except FileNotFoundError as e:
    res["A3"] = {"skipped": str(e)}

(D / "attack_analysis.json").write_text(json.dumps(res, indent=1))
A = res["arms"]
L = ["# Adaptive attacker against the execution probe (PLAN.md section 12)", "",
     f"{len(targets)} target notes from {len(uc)} cases (held out, reached execution in phase 1, flagged by the monitor). "
     f"Monitor threshold tau = {tau['mre']:.3f} (5% false alarms on training notes); held out recall before the attack "
     f"{mon['heldout_recall_before_attack']['mre']:.2f}. Final values from {F['k_final']} fresh rollouts.", "",
     "| | Notes | Evaded [95% CI] | Final execution | Still flagged | Mean monitor score |", "| --- | --- | --- | --- | --- | --- |"]
for arm, nm in [("original", "Original note"), ("P", "P: probe feedback"), ("C", "C: control, execution feedback only")]:
    a = A[arm]
    L.append(f"| {nm} | {a['n']} | {a['evasion']:.3f} [{a['evasion_ci'][0]}, {a['evasion_ci'][1]}] | {a['exec_final']:.3f} | "
             f"{a['flagged']:.3f} | {a['score_mre']:+.2f} |")
L += ["", f"**A1** (P minus C evasion): {res['A1']['P_minus_C']:+.3f} {res['A1']['ci']}, **{'holds' if res['A1']['holds'] else 'does not hold'}**.",
      f"**A2** (within note Spearman of monitor score and execution across P candidates, {res['A2']['n_notes']} notes): "
      f"{res['A2']['mean_within_note_spearman']} {res['A2']['ci']}, **{'holds' if res['A2']['holds'] else 'does not hold'}**.", ""]
if "tradeoff" in res:
    L += ["Execution of P candidates by quartile of monitor score:", "", "| Score range | Mean execution (search) |", "| --- | --- |"]
    L += [f"| {t['score_from']} to {t['score_to']} | {t['mean_exec']:.3f} |" for t in res["tradeoff"]] + [""]
L += ["Transfer (share of evaded finals also below each unattacked monitor's threshold):", "", "```", json.dumps(res["transfer"], indent=1), "```",
      "", "A3 (exploratory retraining, cross fitted by case halves):", "", "```", json.dumps(res["A3"], indent=1), "```"]
(D / "ATTACK_REPORT.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
