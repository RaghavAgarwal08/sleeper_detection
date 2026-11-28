"""G1, P1 and P2 of PLAN.md section 16 from the outputs of patch_heads.py (CPU, seconds).

    python interp/phase3/analyze_patch_heads.py [--model gpt-oss-20b] [--dry-run]
Writes PATCH_HEADS_REPORT.md and patch_heads_analysis.json in results/phase3/<model>/patch_heads/.
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
OUT = Path(os.environ.get("PHASE3_OUT", REPO / "results/phase3"))
o = OUT / args.model / "patch_heads"; sfx = "_dryrun" if args.dry_run else ""
labels = {l["id"]: l for l in map(json.loads, open(REPO / "interp/data/behavior_v1/labels.jsonl"))}
s0 = json.load(open(o / f"stage0{sfx}.json"))
res = {"G1": {k: s0[k] for k in ("n", "mean_within_wording_spearman", "overall_spearman", "G1")}}
rng = np.random.default_rng(0)


def boot(ids, fn):
    by = {}
    for i in ids:
        by.setdefault(labels[i]["case_id"], []).append(i)
    cs = sorted(by)
    v = [fn([i for c in rng.choice(cs, len(cs)) for i in by[c]]) for _ in range(1000)]
    return [round(float(x), 3) for x in np.nanpercentile(v, [2.5, 97.5])]


L = ["# Attention heads from the planted note to the action (PLAN.md section 16)", "",
     f"**G1** (forced decision tracks behaviour): mean within wording Spearman {s0['mean_within_wording_spearman']:.3f}, "
     f"overall {s0['overall_spearman']:.3f}, over {s0['n']} notes: **{'passes' if s0['G1'] else 'fails'}**."]
f1 = o / f"stage1{sfx}.json"
if f1.exists():
    s1 = json.load(open(f1))
    e1 = dict(zip(s1["joint_ids_half1"], s1["joint_effect_half1"]))
    m = lambda b: float(np.mean([e1[i] for i in b])) if b else np.nan  # noqa: E731
    res["P2"] = {"n_pairs": len(e1), "mean_fraction_of_gap": round(m(list(e1)), 3), "ci": boot(list(e1), m),
                 "half0_mean": round(float(np.mean(s1["joint_effect_half0"])), 3) if s1["joint_effect_half0"] else None}
    res["P2"]["holds"] = res["P2"]["ci"][0] > 0
    res["top_heads"] = s1["top"]
    res["top_head_layers"] = sorted({int(h.split(".")[0]) for h in s1["top"]})
    ex = s1["exact"]
    res["exact_top_mean"] = round(float(np.mean([ex[h] for h in s1["top"]])), 3)
    res["exact_random_mean"] = round(float(np.mean([ex[h] for h in s1["random_heads"]])), 3)
    att = s1.get("attention_to_note", {})
    if att and "skipped" not in att:
        res["attention"] = {h: {"to_note": round(float(np.mean([x["note"] for x in v])), 3),
                                "to_memory_result": round(float(np.mean([x["memory_result"] for x in v])), 3),
                                "note_share_of_memory_tokens": round(float(np.mean([x["note_share_of_memory_tokens"] for x in v])), 3)}
                            for h, v in att.items() if v}
    L += ["", f"Discovery on {s1['n_pairs'][0]} pairs, test on {s1['n_pairs'][1]} pairs. Top {len(s1['top'])} heads (layer.head): "
          f"{', '.join(s1['top'])}; layers {res['top_head_layers']}. Mean single head exact effect: top {res['exact_top_mean']}, "
          f"random {res['exact_random_mean']} (fraction of the note minus twin gap removed).", "",
          f"**P2** (patching the top heads together from the twin on held out pairs): mean fraction of the gap "
          f"{res['P2']['mean_fraction_of_gap']} {res['P2']['ci']} (half 0: {res['P2']['half0_mean']}), "
          f"**{'holds' if res['P2']['holds'] else 'does not hold'}**."]
    if "attention" in res:
        L += ["", "Attention from the decision position (descriptive):", "", "| Head | Onto the note | Onto the whole memory result | Note's share of memory result tokens |",
              "| --- | --- | --- | --- |"] + [f"| {h} | {v['to_note']} | {v['to_memory_result']} | {v['note_share_of_memory_tokens']} |" for h, v in res["attention"].items()]
fb = o / f"behaviour{sfx}.json"
if fb.exists():
    B = json.load(open(fb)); pp = B["per_prompt"]
    rate = lambda c, i, k="hits": pp[c][i][k] / pp[c][i]["n"]  # noqa: E731
    res["behaviour"] = {}
    for c in pp:
        ids = list(pp[c])
        res["behaviour"][c] = {k: round(float(np.mean([rate(c, i, key) for i in ids])), 3)
                               for k, key in [("execution", "hits"), ("user_task", "user"), ("any_call", "any_call")]}
    notes = [i for i in B["notes"] if all(i in pp.get(c, {}) for c in ("base", "ablate_top", "ablate_random"))]
    for nm, a, b in [("top_minus_base", "ablate_top", "base"), ("random_minus_base", "ablate_random", "base"),
                     ("top_minus_random", "ablate_top", "ablate_random")]:
        fn = lambda ids, a=a, b=b: float(np.mean([rate(a, i) - rate(b, i) for i in ids])) if ids else np.nan  # noqa: E731
        res["behaviour"][nm] = {"execution": round(fn(notes), 3), "ci": boot(notes, fn)}
    res["P1"] = {**res["behaviour"]["top_minus_random"], "holds": res["behaviour"]["top_minus_random"]["ci"][1] < 0}
    cl = [i for i in B["clean"] if all(i in pp.get(c, {}) for c in ("clean_base", "clean_ablate_top"))]
    if cl:
        fn = lambda ids: float(np.mean([rate("clean_ablate_top", i, "user") - rate("clean_base", i, "user") for i in ids]))  # noqa: E731
        res["clean_user_task_change"] = {"value": round(fn(cl), 3), "ci": boot(cl, fn)}
    L += ["", f"## Behaviour ({len(notes)} half 1 notes, k = 8 first responses; top heads vs layer matched random heads {B['random']})", "",
          "| Condition | Planted call | User's task | Any call |", "| --- | --- | --- | --- |"]
    L += [f"| {c} | {v['execution']} | {v['user_task']} | {v['any_call']} |" for c, v in res["behaviour"].items() if "user_task" in v]
    L += ["", f"Top minus base {res['behaviour']['top_minus_base']}; random minus base {res['behaviour']['random_minus_base']}.",
          f"**P1** (ablate_top minus ablate_random, execution): {res['P1']['execution']} {res['P1']['ci']}, **{'holds' if res['P1']['holds'] else 'does not hold'}**."]
    if "clean_user_task_change" in res:
        L.append(f"User's task on clean prompts under ablate_top minus base: {res['clean_user_task_change']}.")
(o / "patch_heads_analysis.json").write_text(json.dumps(res, indent=1))
(o / "PATCH_HEADS_REPORT.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
