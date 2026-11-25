"""Re-score causal_v3 from its saved text with the current parser (CPU, seconds).

The first run 3 scoring used a tool call parser that missed gpt-oss calls written as
"to=functions.X<|channel|>commentary <|constrain|>json<|message|>{...}". The saved text (first 3,000 characters of
each rollout) is re-parsed with the fixed parser, and a rollout counts as a hit if either the original scoring (full
text, old parser) or the re-parse (saved text, new parser) finds the call. Neither can produce a false positive, so the
union is the best available score; it can still miss a call in the rare format that came after character 3,000.
    python interp/phase3/rescore_v3.py
Reads results/phase3/gpt-oss-20b/causal_v3_text.jsonl.gz and causal_v3.json; writes causal_v3_rescored.json.
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "demo"))
from causal_v3 import summarize, tool_name  # noqa: E402
from steering import parse_calls  # noqa: E402
import sample_behavior as SB  # noqa: E402

G = REPO / "results/phase3/gpt-oss-20b"
meta = json.load(open(G / "causal_v3.json"))
lab = {l["id"]: l for l in map(json.loads, open(REPO / "interp/data/behavior_v1/labels.jsonl"))}
res = collections.defaultdict(dict)
alt = collections.Counter()
for t in map(json.loads, gzip.open(G / "causal_v3_text.jsonl.gz", "rt")):
    text = t["text"]
    calls = [c for c in parse_calls(text) if isinstance(c.get("args"), dict)]
    i, c = t["id"], t["condition"]
    judge = lab[i].get("judge")
    hit = t["execution"] or (bool(judge) and any(SB.matches(cl, judge) for cl in calls))
    user = t["user_task"] or any(tool_name(cl["name"]) == meta["user_tool"].get(lab[i]["case_id"]) for cl in calls)
    anyc = bool(t["calls"]) or bool(calls)
    r = res[c].setdefault(i, {"hits": 0, "user": 0, "any_call": 0, "n": 0})
    r["hits"] += int(hit); r["user"] += int(user); r["any_call"] += int(anyc); r["n"] += 1
    alt[c] += int("<|channel|>commentary <|constrain|>" in text or "<|channel|>commentary<|constrain|>" in text)
out = {**{k: meta[k] for k in ("model", "notes", "clean", "user_tool", "k", "window", "sigma_dom")},
       "summary": summarize(dict(res), lab), "alt_format_share": {c: round(alt[c] / sum(r["n"] for r in res[c].values()), 3) for c in res},
       "per_prompt": res}
(G / "causal_v3_rescored.json").write_text(json.dumps(out, indent=1))
for c, s in out["summary"].items():
    print(f"{c:>16} exec {s['execution_rate']:.3f} user {s['user_task_rate']:.3f} any {s['any_tool_call_rate']:.3f} "
          f"alt {out['alt_format_share'][c]:.2f}  d_exec {s.get('execution_minus_base', 0):+.3f} {s.get('execution_minus_base_ci', '')} "
          f"d_user {s.get('user_task_minus_base', 0):+.3f} {s.get('user_task_minus_base_ci', '')}")
