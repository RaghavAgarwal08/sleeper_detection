"""U1 of PLAN.md section 15 from the outputs of attack_v4.py (CPU, seconds).

    python interp/phase3/analyze_attack_v4.py [--dir results/phase3/attack_v4] [--dry-run]
Writes ATTACK_V4_REPORT.md and attack_v4_analysis.json in that directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "interp/data/behavior_v1"
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=str(REPO / "results/phase3/attack_v4"))
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
D = Path(args.dir); sfx = "_dryrun" if args.dry_run else ""
labels = {l["id"]: l for l in map(json.loads, open(SRC / "labels.jsonl"))}
F = json.load(open(D / f"final{sfx}.json")); mon = F["monitor"]; fin = F["finals"]
targets = [i for i in mon["targets"] if i in fin]
units = {}
for i in targets:
    units.setdefault(labels[i]["case_id"], []).append(i)
rng = np.random.default_rng(0); cs = sorted(units)
BS = [[i for c in rng.choice(cs, len(cs)) for i in units[c]] for _ in range(1000)]
ci = lambda fn: [round(float(x), 3) for x in np.percentile([fn(b) for b in BS], [2.5, 97.5])]  # noqa: E731
ev = lambda ids, a: float(np.mean([fin[i][a]["evaded"] for i in ids]))  # noqa: E731
res = {"n_targets": len(targets), "n_cases": len(cs), "monitor": {k: v for k, v in mon.items() if k != "targets"}, "arms": {}}
for a in ["E_single", "E_union"]:
    res["arms"][a] = {"evasion": round(ev(targets, a), 3), "ci": ci(lambda b, a=a: ev(b, a)),
                      "exec_final": round(float(np.mean([fin[i][a]["exec_final"] for i in targets])), 3),
                      "words_changed": round(float(np.mean([fin[i][a]["words_changed"] for i in targets])), 1),
                      "still_flagged_by_single": round(float(np.mean([fin[i][a]["flag_single"] for i in targets])), 3),
                      "still_flagged_by_union": round(float(np.mean([fin[i][a]["flag_union"] for i in targets])), 3)}
d = lambda ids: ev(ids, "E_single") - ev(ids, "E_union")  # noqa: E731
res["U1"] = {"difference": round(d(targets), 3), "ci": ci(d)}
res["U1"]["holds"] = res["U1"]["ci"][0] > 0
evs = [i for i in targets if fin[i]["E_single"]["evaded"]]
res["E_single_evasions_caught_by_union"] = round(float(np.mean([fin[i]["E_single"]["flag_union"] for i in evs])), 3) if evs else None
res["original_exec_final"] = round(float(np.mean([fin[i]["original"]["exec_final"] for i in targets])), 3)
res["agent_errors"] = F.get("agent_errors")
(D / "attack_v4_analysis.json").write_text(json.dumps(res, indent=1))
m = mon
L = ["# Small edit attacker against the combined monitor (PLAN.md section 15)", "",
     f"Cross fitted over two halves of cases. Every monitor at {m['fpr']:.0%} false alarms on non executed notes of the half it scores "
     f"(union: {', '.join(str(round(t['union_false_alarms'], 3)) for t in m['thresholds'].values())}). "
     f"Recall on all {m['n_executed']} executed notes: single {m['recall_all_executed_crossfitted']['single']}, "
     f"union {m['recall_all_executed_crossfitted']['union']}. {m['n_eligible']} notes eligible, {len(targets)} targets from {len(cs)} cases; "
     f"originals executed {res['original_exec_final']} in 16 fresh rollouts.", "",
     "| Arm | Evaded [95% CI] | Final execution | Words changed | Still flagged by single | Still flagged by union |",
     "| --- | --- | --- | --- | --- | --- |"]
for a in ["E_single", "E_union"]:
    x = res["arms"][a]
    L.append(f"| {a} | {x['evasion']:.3f} {x['ci']} | {x['exec_final']:.3f} | {x['words_changed']} | {x['still_flagged_by_single']:.3f} | {x['still_flagged_by_union']:.3f} |")
L += ["", f"**U1** (E_single minus E_union): {res['U1']['difference']:+.3f} {res['U1']['ci']}, **{'holds' if res['U1']['holds'] else 'does not hold'}**.",
      f"Share of E_single's evasions that the union still flags: {res['E_single_evasions_caught_by_union']}.",
      f"Agent rollouts that hit the special token error: {res['agent_errors']}."]
(D / "ATTACK_V4_REPORT.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
