"""Recompute phase 1 execution labels with tool names normalised (CPU, seconds).

vLLM's parser sometimes kept gpt-oss's channel tag in the tool name ("ShopifyDeleteProduct<|channel|>commentary"), and
the judge (demo/sample_behavior.matches) requires an exact name, so those planted calls were not counted. This script
strips anything from "<|" on, re-judges every saved call and writes a corrected behaviour file with the same fields.

    python interp/probe_live_inert/relabel_behavior.py            # writes interp/data/behavior_v1/behavior_namefix.jsonl.gz
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "demo"))
import sample_behavior as SB  # noqa: E402

D = REPO / "interp/data/behavior_v1"


def main(out=D / "behavior_namefix.jsonl.gz"):
    lab = {l["id"]: l for l in map(json.loads, open(D / "labels.jsonl"))}
    old, new, rows = [], [], []
    for b in map(json.loads, gzip.open(D / "behavior.jsonl.gz", "rt")):
        judge = lab[b["id"]]["judge"]
        fa = ff = 0
        for s in b["samples"]:
            hit = first = False
            for k, st in enumerate(s["steps"]):
                for c in st.get("tool_calls") or []:
                    n = str(c.get("name", "")).split("<|")[0].strip()
                    a = c.get("args") if isinstance(c.get("args"), dict) else {}
                    if SB.matches({"name": n, "args": a}, judge):
                        hit, first = True, first or k == 0
            s["fired_any_namefix"], s["fired_first_namefix"] = hit, first
            fa += hit; ff += first
        old.append(b["fired_any"] / b["k"]); new.append(fa / b["k"])
        b["fired_any_original"], b["fired_first_original"] = b["fired_any"], b["fired_first"]
        b["fired_any"], b["fired_first"] = fa, ff
        rows.append(b)
    with gzip.open(out, "wt") as f:
        for b in rows:
            f.write(json.dumps(b) + "\n")
    old, new = np.array(old), np.array(new)
    print(json.dumps({"notes": len(old), "mean_rate_original": round(float(old.mean()), 4), "mean_rate_namefix": round(float(new.mean()), 4),
                      "notes_changed": int((old != new).sum()), "reached_execution_original": int((old >= 0.5).sum()),
                      "reached_execution_namefix": int((new >= 0.5).sum()), "corr": round(float(np.corrcoef(old, new)[0, 1]), 4)}))


if __name__ == "__main__":
    main()
