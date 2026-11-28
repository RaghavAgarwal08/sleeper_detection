"""GPU: cut the planted note out of attention in bands of layers during real generation (PLAN.md section 17).

Value zeroing: in the chosen decoder layers, the value projection's output is set to zero at every token of the
planted note (all occurrences in the memory result) during the prompt pass; generated tokens then attend to zero
vectors there through the KV cache. Control: the same number of non note memory result tokens, random per prompt.

    python interp/phase3/note_knockout.py --model gpt-oss-20b
    python interp/phase3/note_knockout.py --model qwen2.5-7b --dry-run     # CPU test with the tiny model
Writes results/phase3/<model>/note_knockout.json after every condition and note_knockout_text.jsonl.gz; --resume.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "demo"))
import models as M  # noqa: E402
from causal_v3 import tool_name, user_tools  # noqa: E402
from steering import parse_calls  # noqa: E402

DATA = Path(os.environ.get("PHASE3_DATA", REPO / "interp/data/phase3"))
OUT = Path(os.environ.get("PHASE3_OUT", REPO / "results/phase3"))


class ValueZero:
    """Forward hooks on v_proj of the given layers: zero rows at positions pos[b] (a list per row) in the prompt pass."""

    def __init__(self, model, layers):
        self.pos = None
        self.hs = [model.model.layers[l].self_attn.v_proj.register_forward_hook(self._fn) for l in layers]

    def _fn(self, _m, _inp, out):
        if out.shape[1] > 1 and self.pos is not None:
            out = out.clone()
            for b, p in enumerate(self.pos):
                if len(p):
                    out[b, p] = 0
            return out
        return None

    def remove(self):
        for h in self.hs:
            h.remove()


def note_positions(src_req, built_req):
    """Token positions of every occurrence of the planted note (entry 0), from the dataset's token spans."""
    spans = src_req.get("entry_spans") or []
    if not spans or spans[0].get("entry_index", 0) != 0:
        return []
    return sorted({p for a, b in spans[0]["occurrences"] for p in range(a, b) if p < len(built_req["input_ids"])})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--notes", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    import torch
    import sample_behavior as SB
    spec = M.get(args.model)
    d = DATA / spec["key"]
    labels = [json.loads(l) for l in open(d / "labels.jsonl")]
    lab = {l["id"]: l for l in labels}
    beh = {b["id"]: b for b in map(json.loads, gzip.open(REPO / "interp/data/behavior_v1/behavior_namefix.jsonl.gz", "rt"))}
    built = {json.loads(l)["id"]: json.loads(l) for l in open(d / "requests.jsonl")}
    src = {json.loads(l)["id"]: json.loads(l) for l in open(REPO / "interp/data/behavior_v1/requests.jsonl")}
    utool = user_tools(labels, beh)
    if args.dry_run:
        from test_phase3 import tiny_model
        model, tok = tiny_model()
        # the tiny model's prompts are rendered differently: use the note span found by build_prompts (last token) only
        npos = lambda i: list(range(max(0, built[i]["positions"].get("note", 0) - 5), built[i]["positions"].get("note", 0) + 1))  # noqa: E731
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec["hf"])
        model = M.load_hf(spec)
        npos = lambda i: note_positions(src[i], built[i])  # noqa: E731
    model.eval()
    nL = model.config.num_hidden_layers
    eligible = sorted(i for i, b in beh.items() if (l := lab.get(i)) and l["framing"] == "live" and i in built
                      and b["fired_first"] >= 2 and l["case_id"] in utool and npos(i))
    notes = sorted(random.Random(0).sample(eligible, min(args.notes, len(eligible))))
    rng = random.Random(0)
    ctrl = {}
    for i in notes:
        p = built[i]["positions"]
        s, e = p.get("memory_result_start", p["memory_result_end"]), p["memory_result_end"]
        npi = set(npos(i))
        pool = [q for q in range(s, e + 1) if q not in npi]
        ctrl[i] = sorted(rng.sample(pool, min(len(npi), len(pool))))
    q = nL // 4
    bands = {f"note_L{a}_{min(a + q, nL) - 1}": list(range(a, min(a + q, nL))) for a in range(0, nL, q)}
    conds = {"base": (None, None), "note_all": ("note", list(range(nL))), "control_all": ("control", list(range(nL)))}
    conds.update({k: ("note", v) for k, v in bands.items()})
    print(f"{len(notes)} notes of {len(eligible)} eligible; note tokens per prompt "
          f"{np.mean([len(npos(i)) for i in notes]):.0f}; conditions {list(conds)}", flush=True)

    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id if isinstance(tok.eos_token_id, int) else 0)
    o = OUT / spec["key"]; o.mkdir(parents=True, exist_ok=True)
    tag = "_dryrun" if args.dry_run else ""
    fn, ftext = o / f"note_knockout{tag}.json", o / f"note_knockout_text{tag}.jsonl.gz"
    results = {}
    if args.resume and fn.exists():
        old = json.load(open(fn))
        if old.get("notes") == notes:
            results = {c: x for c, x in old["per_prompt"].items() if c in conds}
            print("resuming, already done:", sorted(results), flush=True)
    meta = {"model": spec["hf"], "notes": notes, "k": args.k, "conditions": {c: [w, ls] for c, (w, ls) in conds.items()},
            "note_tokens": {i: len(npos(i)) for i in notes}}
    t0 = time.time()
    for c, (what, layers) in conds.items():
        if c in results:
            continue
        hook = ValueZero(model, layers) if what else None
        jobs = [(i, s) for i in notes for s in range(args.k)]
        res, texts = {}, []
        for b0 in range(0, len(jobs), args.batch):
            chunk = jobs[b0:b0 + args.batch]
            seqs = [built[i]["input_ids"] for i, _ in chunk]
            L = max(map(len, seqs))
            ids = torch.tensor([[pad] * (L - len(s)) + s for s in seqs], device=model.device)
            att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=model.device)
            if hook:
                hook.pos = [[L - len(s) + p for p in (npos(i) if what == "note" else ctrl[i])] for (i, _), s in zip(chunk, seqs)]
            torch.manual_seed(b0)
            with torch.no_grad():
                g = model.generate(input_ids=ids, attention_mask=att, do_sample=True, temperature=1.0, top_p=1.0,
                                   max_new_tokens=args.max_new_tokens, pad_token_id=pad)
            for (i, s), row in zip(chunk, g[:, L:]):
                text = tok.decode(row, skip_special_tokens=False)
                calls = [cl for cl in parse_calls(text) if isinstance(cl.get("args"), dict)]
                case = lab[i]["case_id"]
                hit = bool(lab[i].get("judge")) and any(SB.matches(cl, lab[i]["judge"]) for cl in calls)
                user = any(tool_name(cl["name"]) == utool.get(case) for cl in calls)
                r = res.setdefault(i, {"hits": 0, "user": 0, "any_call": 0, "n": 0})
                r["hits"] += int(hit); r["user"] += int(user); r["any_call"] += int(bool(calls)); r["n"] += 1
                texts.append({"condition": c, "id": i, "sample": s, "execution": hit, "user_task": user,
                              "calls": [tool_name(cl["name"]) for cl in calls], "text": text})
        if hook:
            hook.remove()
        results[c] = res
        with gzip.open(ftext, "at") as f:
            for t in texts:
                f.write(json.dumps(t) + "\n")
        mm = lambda key: np.mean([x[key] / x["n"] for x in res.values()])  # noqa: E731
        print(f"{c:>14}: execution {mm('hits'):.3f}  user task {mm('user'):.3f}  any call {mm('any_call'):.3f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
        fn.write_text(json.dumps({**meta, "per_prompt": results}, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
