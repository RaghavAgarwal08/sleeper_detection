"""GPU: is the difference of means direction a usable defense? Pre-registered as H6 in interp/phase3/PLAN.md.

causal_v2.py found that steering every window layer by -1 SD along the difference of means direction (reached execution
minus triggered resisted, training split) cut first step execution from 0.26 to 0.07 while 91% of rollouts still called
a tool. Those notes were in the training split the direction came from, and "any tool call" does not show whether the
agent still did what the user asked. This run fixes both.

Prompts
  poisoned   every held out poisoned note (splits held_out_mixed and test) whose phase 1 fired_first is between 1/8 and
             7/8; the direction never saw them
  clean      the clean prompt (no planted note) of the same cases
Per rollout (first assistant step, T = 1, k samples) three outcomes
  execution  the planted call, judged by the case's own judge (as before)
  user_task  a call to the tool the user asked for: per case, the most common non planted tool in the first step of the
             phase 1 rollouts (e.g. ShopifyDeleteProduct)
  any_call   any tool call
Conditions (alpha in SD of the difference of means projection per window layer, from the start of the memory result
and on every generated token)
  base, dom_{-1,-0.5}, random_-1 (a fixed random direction orthogonal to the probe, same size as dom_-1) on poisoned
  clean_base, clean_dom_-1, clean_random_-1 on clean
The generated text of every rollout is saved (results/phase3/<model>/causal_v3_text.jsonl.gz) so refusals can be read.

    python interp/phase3/causal_v3.py --model gpt-oss-20b --k 8 --batch 16 --max-new-tokens 1536
Writes results/phase3/<model>/causal_v3.json after every condition; --resume skips finished conditions.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "interp/probe_live_inert"))
sys.path.insert(0, str(REPO / "demo"))
import models as M  # noqa: E402
from causal_v2 import dom_direction  # noqa: E402
from steering import Steer, direction, parse_calls  # noqa: E402

DATA = Path(os.environ.get("PHASE3_DATA", REPO / "interp/data/phase3"))
OUT = Path(os.environ.get("PHASE3_OUT", REPO / "results/phase3"))
HELD_OUT = {"held_out_mixed", "test"}


def tool_name(n: str) -> str:
    return re.split(r"<\|", str(n))[0].strip()


def user_tools(labels, beh_raw):
    """Per case: the most common non planted tool called in the first step of the phase 1 rollouts."""
    lab = {l["id"]: l for l in labels}
    cnt = collections.defaultdict(collections.Counter)
    for b in beh_raw.values():
        l = lab.get(b["id"])
        if not l or "samples" not in b:
            continue
        tgt = l["judge"]["target_tool"]
        for s in b["samples"]:
            if not s.get("steps"):
                continue
            for c in s["steps"][0].get("tool_calls") or []:
                n = tool_name(c.get("name") if isinstance(c, dict) else "")
                if n and n != tgt and n != "MemoryUpdate":
                    cnt[l["case_id"]][n] += 1
    return {k: v.most_common(1)[0][0] for k, v in cnt.items() if v}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--notes", type=int, default=0, help="cap on poisoned notes (0 = all eligible)")
    ap.add_argument("--dom-alphas", default="-1,-0.5")
    ap.add_argument("--random-alphas", default="-1", help="in difference of means SD units, so sizes match")
    ap.add_argument("--clean-conds", default="base,dom_-1,random_-1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="tiny random model on CPU, for tests only")
    args = ap.parse_args()
    import torch
    spec = M.get(args.model)
    d = DATA / spec["key"]
    labels = [json.loads(l) for l in open(d / "labels.jsonl")]
    lab = {l["id"]: l for l in labels}
    bpath = d / "behavior.jsonl"
    if not bpath.exists():
        bpath = REPO / "interp/data/behavior_v1/behavior.jsonl"
        bpath = bpath if bpath.exists() else Path(str(bpath) + ".gz")
    opener = gzip.open if str(bpath).endswith(".gz") else open
    beh = {b["id"]: b for b in map(json.loads, opener(bpath, "rt"))}
    built = {json.loads(l)["id"]: json.loads(l) for l in open(d / "requests.jsonl")}
    utool = user_tools(labels, beh)

    notes = sorted(i for i, b in beh.items() if (l := lab.get(i)) and l["framing"] == "live" and l["split"] in HELD_OUT
                   and i in built and 0 < b["fired_first"] < b["k"] and l["case_id"] in utool)
    if args.notes:
        notes = notes[: args.notes]
    cases = sorted({lab[i]["case_id"] for i in notes})
    clean_by_case = {}
    for l in labels:
        if l["framing"] == "none" and l["id"] in built and l["case_id"] in cases:
            clean_by_case.setdefault(l["case_id"], l["id"])
    clean = [clean_by_case[c] for c in cases if c in clean_by_case]
    print(f"{len(notes)} held out poisoned notes, {len(clean)} clean prompts, {len(cases)} cases, "
          f"user tool known for {sum(c in utool for c in cases)}", flush=True)

    if args.dry_run:
        from test_phase3 import tiny_model  # noqa: E402
        model, tok = tiny_model()
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec["hf"])
        model = M.load_hf(spec)
    model.eval()
    win = M.window_layers(model.config.num_hidden_layers)
    win_dec = [l - 1 for l in win]
    if args.dry_run:
        g = np.random.default_rng(1)
        u = g.normal(size=model.config.hidden_size); u /= np.linalg.norm(u)
        v = g.normal(size=model.config.hidden_size); v /= np.linalg.norm(v)
        sigma_dom, dinfo = 1.0, {"dry_run": True}
    else:
        u, _, dinfo = direction(spec, labels, beh, win)
        v, sigma_dom = dom_direction(spec, labels, beh)
    r = np.random.default_rng(12345).normal(size=len(u)); r -= (r @ u) * u; r /= np.linalg.norm(r)

    fl = lambda s: [float(x) for x in s.split(",") if x]  # noqa: E731
    conds = {"base": ("poisoned", None)}
    for a in fl(args.dom_alphas):
        conds[f"dom_{a:+g}"] = ("poisoned", ({l: ("add", a * sigma_dom) for l in win_dec}, v))
    for a in fl(args.random_alphas):
        conds[f"random_{a:+g}"] = ("poisoned", ({l: ("add", a * sigma_dom) for l in win_dec}, r))
    for c in [x for x in args.clean_conds.split(",") if x]:
        if c == "base":
            conds["clean_base"] = ("clean", None)
        else:
            kind, a = c.split("_")
            conds[f"clean_{kind}_{float(a):+g}"] = ("clean", ({l: ("add", float(a) * sigma_dom) for l in win_dec},
                                                              v if kind == "dom" else r))

    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id if isinstance(tok.eos_token_id, int) else 0)
    import sample_behavior as SB  # noqa: E402
    o = OUT / spec["key"]
    o.mkdir(parents=True, exist_ok=True)
    tag = "_dryrun" if args.dry_run else ""
    fn, ftext = o / f"causal_v3{tag}.json", o / f"causal_v3_text{tag}.jsonl.gz"
    results = {}
    if args.resume and fn.exists():
        old = json.load(open(fn))
        if old.get("notes") == notes and old.get("clean") == clean:
            results = {c: x for c, x in old["per_prompt"].items() if c in conds}
            print("resuming, already done:", sorted(results), flush=True)
    meta = {"model": spec["hf"], "notes": notes, "clean": clean, "user_tool": {c: utool[c] for c in cases}, "k": args.k,
            "window": win, "sigma_dom": sigma_dom, "direction_info": dinfo}
    t0 = time.time()
    for c, (pop, edit) in conds.items():
        if c in results:
            continue
        prompts = notes if pop == "poisoned" else clean
        hook = Steer(model, edit[1], edit[0]) if edit else None
        jobs = [(i, s) for i in prompts for s in range(args.k)]
        res = {}
        texts = []
        for b0 in range(0, len(jobs), args.batch):
            chunk = jobs[b0:b0 + args.batch]
            seqs = [built[i]["input_ids"] for i, _ in chunk]
            L = max(map(len, seqs))
            ids = torch.tensor([[pad] * (L - len(s)) + s for s in seqs], device=model.device)
            att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=model.device)
            if hook:
                p = lambda i: built[i]["positions"]  # noqa: E731
                hook.starts = torch.tensor([L - len(s) + p(i).get("memory_result_start", p(i)["memory_result_end"])
                                            for (i, _), s in zip(chunk, seqs)], device=model.device)
            torch.manual_seed(args.seed + b0)
            with torch.no_grad():
                g = model.generate(input_ids=ids, attention_mask=att, do_sample=True, temperature=1.0, top_p=1.0,
                                   max_new_tokens=args.max_new_tokens, pad_token_id=pad)
            for (i, s), row in zip(chunk, g[:, L:]):
                text = tok.decode(row, skip_special_tokens=False)
                calls = [cl for cl in parse_calls(text) if isinstance(cl.get("args"), dict)]
                judge = lab[i]["judge"] if lab[i].get("judge") else None
                case = lab[i]["case_id"]
                hit = bool(judge) and any(SB.matches(cl, judge) for cl in calls)
                user = any(tool_name(cl["name"]) == utool.get(case) for cl in calls)
                rec = res.setdefault(i, {"hits": 0, "user": 0, "any_call": 0, "n": 0})
                rec["hits"] += int(hit); rec["user"] += int(user); rec["any_call"] += int(bool(calls)); rec["n"] += 1
                texts.append({"condition": c, "id": i, "sample": s, "execution": hit, "user_task": user,
                              "calls": [tool_name(cl["name"]) for cl in calls], "text": text[:3000]})
        if hook:
            hook.remove()
        results[c] = res
        with gzip.open(ftext, "at") as f:
            for t in texts:
                f.write(json.dumps(t) + "\n")
        m = lambda key: np.mean([x[key] / x["n"] for x in res.values()])  # noqa: E731
        print(f"{c:>18}: execution {m('hits'):.3f}  user task {m('user'):.3f}  any call {m('any_call'):.3f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
        fn.write_text(json.dumps({**meta, "summary": summarize(results, lab), "per_prompt": results}, indent=1))
    print(json.dumps(json.load(open(fn))["summary"], indent=1))


def summarize(results, lab):
    brng = np.random.default_rng(0)
    out = {}
    for c, res in results.items():
        base_c = "clean_base" if c.startswith("clean") else "base"
        ids = list(res)
        row = {k: round(float(np.mean([res[i][key] / res[i]["n"] for i in ids])), 3)
               for k, key in [("execution_rate", "hits"), ("user_task_rate", "user"), ("any_tool_call_rate", "any_call")]}
        if base_c in results and c != base_c:
            base = results[base_c]
            cs = {i: lab[i]["case_id"] for i in ids}
            uc = sorted(set(cs.values()))
            idx = {cc: [k for k, i in enumerate(ids) if cs[i] == cc] for cc in uc}
            for key, name in [("hits", "execution"), ("user", "user_task")]:
                dif = np.array([res[i][key] / res[i]["n"] - base[i][key] / base[i]["n"] for i in ids])
                bs = [dif[np.concatenate([idx[cc] for cc in brng.choice(uc, len(uc))])].mean() for _ in range(1000)]
                row[f"{name}_minus_base"] = round(float(dif.mean()), 3)
                row[f"{name}_minus_base_ci"] = [round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])]
        out[c] = row
    return out


if __name__ == "__main__":
    main()
