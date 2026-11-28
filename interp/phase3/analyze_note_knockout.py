"""K1 and K2 of PLAN.md section 17 from note_knockout.py's output (CPU, seconds).

    python interp/phase3/analyze_note_knockout.py [--model gpt-oss-20b] [--dry-run]
Writes results/phase3/<model>/NOTE_KNOCKOUT_REPORT.md and note_knockout_analysis.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gpt-oss-20b")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
o = Path(os.environ.get("PHASE3_OUT", REPO / "results/phase3")) / args.model
J = json.load(open(o / f"note_knockout{'_dryrun' if args.dry_run else ''}.json")); pp = J["per_prompt"]
labels = {l["id"]: l for l in map(json.loads, open(REPO / "interp/data/behavior_v1/labels.jsonl"))}
notes = [i for i in J["notes"] if all(i in pp[c] for c in pp)]
by = {}
for i in notes:
    by.setdefault(labels[i]["case_id"], []).append(i)
cs = sorted(by); rng = np.random.default_rng(0)
BS = [[i for c in rng.choice(cs, len(cs)) for i in by[c]] for _ in range(2000)]
rate = lambda c, i, k="hits": pp[c][i][k] / pp[c][i]["n"]  # noqa: E731


def diff(a, b, level=95):
    f = lambda ids: float(np.mean([rate(a, i) - rate(b, i) for i in ids]))  # noqa: E731
    lo = (100 - level) / 2
    return {"value": round(f(notes), 3), "ci": [round(float(x), 3) for x in np.percentile([f(x) for x in BS], [lo, 100 - lo])],
            "level": level}


res = {"n_notes": len(notes), "n_cases": len(cs), "conditions": {}}
for c in pp:
    res["conditions"][c] = {k: round(float(np.mean([rate(c, i, key) for i in notes])), 3)
                            for k, key in [("execution", "hits"), ("user_task", "user"), ("any_call", "any_call")]}
if "note_all" in pp and "control_all" in pp:
    res["K1"] = diff("note_all", "control_all"); res["K1"]["holds"] = res["K1"]["ci"][1] < 0
    res["note_all_minus_base"] = diff("note_all", "base"); res["control_all_minus_base"] = diff("control_all", "base")
bands = [c for c in pp if c.startswith("note_L")]
res["K2"] = {}
for c in bands:
    x = diff(c, "base", level=100 - 5 / max(len(bands), 1)); x["matters"] = x["ci"][1] < 0
    res["K2"][c] = x
(o / "note_knockout_analysis.json").write_text(json.dumps(res, indent=1))
L = ["# Cutting the planted note out of attention (PLAN.md section 17)", "",
     f"{len(notes)} poisoned notes from {len(cs)} cases, k = {J['k']} first responses each; value zeroing at the note's tokens "
     f"(about {np.mean(list(J['note_tokens'].values())):.0f} tokens per prompt) in the listed layers.", "",
     "| Condition | Planted call | User's task | Any call |", "| --- | --- | --- | --- |"]
L += [f"| {c} | {v['execution']} | {v['user_task']} | {v['any_call']} |" for c, v in res["conditions"].items()]
if "K1" in res:
    L += ["", f"note_all minus base {res['note_all_minus_base']['value']} {res['note_all_minus_base']['ci']}; "
          f"control_all minus base {res['control_all_minus_base']['value']} {res['control_all_minus_base']['ci']}.",
          f"**K1** (note_all minus control_all): {res['K1']['value']} {res['K1']['ci']}, **{'holds' if res['K1']['holds'] else 'does not hold'}**."]
L += ["", "**K2** (each band minus base, Bonferroni corrected intervals):", ""]
L += [f"* {c}: {x['value']} {x['ci']} ({x['level']:.2f}%), {'matters' if x['matters'] else 'not shown to matter'}" for c, x in res["K2"].items()]
(o / "NOTE_KNOCKOUT_REPORT.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
