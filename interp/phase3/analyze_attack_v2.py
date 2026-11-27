"""W1, A1v2 and A3v2 of PLAN.md section 13 from the outputs of attack_v2.py (CPU, seconds).

    python interp/phase3/analyze_attack_v2.py [--dir results/phase3/attack_v2] [--dry-run]
Writes ATTACK_V2_REPORT.md and attack_v2_analysis.json in that directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "interp/data/behavior_v1"
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=str(REPO / "results/phase3/attack_v2"))
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
D = Path(args.dir); sfx = "_dryrun" if args.dry_run else ""
labels = {l["id"]: l for l in map(json.loads, open(SRC / "labels.jsonl"))}
F = json.load(open(D / f"final{sfx}.json")); mon = F["monitor"]; fin = F["finals"]; tau = mon["tau"]
P = np.load(D / f"paraphrase_scores{sfx}.npz")
rng = np.random.default_rng(0)


def boot(units, fn, n=1000):
    """Case bootstrap: units is {case: [items]}; fn(list of items) -> value."""
    cs = sorted(units)
    v = []
    for _ in range(n):
        items = [x for c in rng.choice(cs, len(cs)) for x in units[c]]
        val = fn(items)
        if val == val:
            v.append(val)
    return [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])] if v else [None, None]


res = {"monitor": mon}
# W1: held out paraphrases the agent executes
hp = P["heldout"].astype(bool); r = P["rate"]; src = P["src"]
work = np.where(hp & (r >= 0.5))[0]
units = {}
for k in work:
    units.setdefault(labels[str(src[k])]["case_id"], []).append(k)
rec = lambda ks, p: float(np.mean([P[p][k] >= tau[p] for k in ks])) if ks else np.nan  # noqa: E731
diff = lambda ks: rec(ks, "W") - rec(ks, "O")  # noqa: E731
res["W1"] = {"n_working_heldout_paraphrases": int(len(work)), "n_cases": len(units),
             "recall_O": round(rec(list(work), "O"), 3), "recall_W": round(rec(list(work), "W"), 3),
             "W_minus_O": round(diff(list(work)), 3), "ci": boot(units, diff)}
res["W1"]["holds"] = res["W1"]["ci"][0] is not None and res["W1"]["ci"][0] > 0
y = (r[hp] >= 0.5).astype(int)
res["W1"]["auroc_heldout_paraphrases"] = {p: (round(float(roc_auc_score(y, P[p][hp])), 3) if 0 < y.sum() < len(y) else None) for p in ["O", "W"]}
res["W1"]["recall_heldout_originals"] = mon["recall_heldout_originals"]

# A1v2 and A3v2
targets = [i for i in mon["targets"] if i in fin]
tu = {}
for i in targets:
    tu.setdefault(labels[i]["case_id"], []).append(i)
ev = lambda ids, a: float(np.mean([fin[i][a]["evaded"] for i in ids])) if ids else np.nan  # noqa: E731
res["arms"] = {}
for a in ["original", "P2", "P2W", "C2"]:
    res["arms"][a] = {"evasion": round(ev(targets, a), 3), "ci": boot(tu, lambda ids, a=a: ev(ids, a)),
                      "exec_final": round(float(np.mean([fin[i][a]["exec_final"] for i in targets])), 3),
                      "words_changed": round(float(np.mean([fin[i][a].get("words_changed", 0) for i in targets])), 1),
                      "flagged_by_O": round(float(np.mean([fin[i][a]["score_O"] >= tau["O"] for i in targets])), 3),
                      "flagged_by_W": round(float(np.mean([fin[i][a]["score_W"] >= tau["W"] for i in targets])), 3)}
d1 = lambda ids: ev(ids, "P2") - ev(ids, "C2")  # noqa: E731
d3 = lambda ids: ev(ids, "P2") - ev(ids, "P2W")  # noqa: E731
res["A1v2"] = {"P2_minus_C2": round(d1(targets), 3), "ci": boot(tu, d1)}
res["A1v2"]["holds"] = res["A1v2"]["ci"][0] is not None and res["A1v2"]["ci"][0] > 0
res["A3v2"] = {"P2_minus_P2W": round(d3(targets), 3), "ci": boot(tu, d3)}
res["A3v2"]["holds"] = res["A3v2"]["ci"][0] is not None and res["A3v2"]["ci"][0] > 0
evP2 = [i for i in targets if fin[i]["P2"]["evaded"]]
res["transfer_of_P2_evasions"] = {"n": len(evP2),
                                  "also_below_W": round(float(np.mean([fin[i]["P2"]["score_W"] < tau["W"] for i in evP2])), 3) if evP2 else None}
(D / "attack_v2_analysis.json").write_text(json.dumps(res, indent=1))

W1 = res["W1"]
L = ["# Varied wording probe and small edit attacker (PLAN.md section 13)", "",
     "## W1: does training on paraphrases help the probe catch reworded notes?", "",
     f"Probe O: original training notes. Probe W: plus {mon['train_paraphrases']} training paraphrases (no feedback, "
     f"executed at rate {mon['train_paraphrase_exec_rate']}). Thresholds at 5% false alarms on each probe's own training rows.", "",
     "| | Probe O | Probe W |", "| --- | --- | --- |",
     f"| Recall on held out paraphrases the agent executes ({W1['n_working_heldout_paraphrases']}, {W1['n_cases']} cases) | {W1['recall_O']:.3f} | {W1['recall_W']:.3f} |",
     f"| AUROC on all held out paraphrases (executed vs not) | {W1['auroc_heldout_paraphrases']['O']} | {W1['auroc_heldout_paraphrases']['W']} |",
     f"| Recall on held out original notes | {W1['recall_heldout_originals']['O']} | {W1['recall_heldout_originals']['W']} |", "",
     f"**W1** (W minus O recall): {W1['W_minus_O']:+.3f} {W1['ci']}, **{'holds' if W1['holds'] else 'does not hold'}**.", "",
     "## Small edit attacker", "",
     f"{len(targets)} targets (held out, executed in phase 1, flagged by both probes). Final values from {F['k_final']} fresh rollouts. "
     "P2 attacks probe O, P2W attacks probe W, C2 is the control (no score shown, judged against O).", "",
     "| Arm | Evaded [95% CI] | Final execution | Words changed | Flagged by O | Flagged by W |", "| --- | --- | --- | --- | --- | --- |"]
for a, nm in [("original", "Original note"), ("P2", "P2: small edits against O"), ("P2W", "P2W: small edits against W"), ("C2", "C2: control")]:
    x = res["arms"][a]
    L.append(f"| {nm} | {x['evasion']:.3f} {x['ci']} | {x['exec_final']:.3f} | {x['words_changed']} | {x['flagged_by_O']:.3f} | {x['flagged_by_W']:.3f} |")
L += ["", f"**A1v2** (P2 minus C2): {res['A1v2']['P2_minus_C2']:+.3f} {res['A1v2']['ci']}, **{'holds' if res['A1v2']['holds'] else 'does not hold'}**.",
      f"**A3v2** (P2 minus P2W): {res['A3v2']['P2_minus_P2W']:+.3f} {res['A3v2']['ci']}, **{'holds' if res['A3v2']['holds'] else 'does not hold'}**.",
      f"Of P2's {len(evP2)} evaded notes, the share W also misses: {res['transfer_of_P2_evasions']['also_below_W']}."]
(D / "ATTACK_V2_REPORT.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
