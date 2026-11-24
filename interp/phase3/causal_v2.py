"""GPU, exploratory follow up to the failed H5 steering test: is the execution signal causal, and where does it live?

steering.py pushed along the execution probe direction at ONE layer (after decoder layer 13) by up to 4 SD and saw no
change in execution. Three explanations remain (results/phase3/GPTOSS_GPU_REPORT.md section 6): the direction is a
correlate, the decision is recomputed after a single layer edit, or a logistic regression direction is a poor steering
vector. This script separates them with two kinds of intervention on the same notes (the 48 training notes with mixed
outcomes that steering.py used) and the same outcome (planted call in the first assistant step, k samples, T = 1).

Steering at EVERY window layer (after decoder layers 13 to 18), from the start of the memory result onward and on
every generated token, alpha SD per layer (so the total push builds up to 6 alpha SD):
  probe_all_{+2,+4,-4}   the execution probe direction (as in steering.py)
  dom_all_{+4,-4}        difference of means direction, reached execution minus triggered resisted (training split)
  random_all_+4          a fixed random direction orthogonal to the probe
Activation patching from the benign twin (same case, same wording, the note says the task is already done):
  patch_twin_L{6,12,17,22}   residual stream after layer L at the end of the memory result and every later prompt
                             token is replaced by the twin's values at the same positions (aligned from the end of the
                             prompt; the suffix text is identical). Generated tokens can still attend to the note itself.
  twin_prompt            the benign twin's own prompt, judged for the same planted call: the floor that full
                         removal of the note's intent would reach, so the patching effects have a scale.
  If execution collapses when patching at layer L, the end of the memory result carries what the model needs to act by
  layer L; if nothing changes at any L, the model reads the note directly and the probe's site is not on the causal path.
The unsteered baseline runs first. Results are written after every condition, so a crash loses at most one.

    python interp/phase3/causal_v2.py --model gpt-oss-20b --notes 48 --k 8 --batch 16 --max-new-tokens 1536
    python interp/phase3/causal_v2.py --only base,patch_twin_L17       # a subset
Writes results/phase3/<model>/causal_v2.json (about 30 GPU minutes per condition on an H100; 12 conditions).
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
sys.path.insert(0, str(REPO / "interp/probe_live_inert"))
sys.path.insert(0, str(REPO / "demo"))
import models as M  # noqa: E402
from steering import Steer, direction, parse_calls  # noqa: E402

DATA = Path(os.environ.get("PHASE3_DATA", REPO / "interp/data/phase3"))
OUT = Path(os.environ.get("PHASE3_OUT", REPO / "results/phase3"))
PATCH_LAYERS = [6, 12, 17, 22]  # residual index (output of decoder layer L - 1); scaled for smaller test models


class Patch:
    """During the prompt pass only, replace the output of decoder layer `layer_idx` at each row's last m_b positions with
    donor[b] ([m_b, d]). Rows are left padded, so the last positions line up with the end of each prompt."""

    def __init__(self, model, layer_idx):
        self.donor = None
        self.h = model.model.layers[layer_idx].register_forward_hook(self._fn, prepend=True)

    def _fn(self, _m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1 and self.donor is not None:
            h = h.clone()
            T = h.shape[1]
            for b, d in enumerate(self.donor):
                if d is not None:
                    h[b, T - d.shape[0]:] = d.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def remove(self):
        self.h.remove()


def dom_direction(spec, labels, beh):
    """Difference of means, reached execution minus triggered resisted, same features and split as direction()."""
    act = DATA / spec["key"] / "activations.npz"
    if act.exists():
        z = np.load(act)
        X = dict(zip(z["ids"], z["window_mean"][:, 0]))
    else:
        z = np.load(REPO / "results/probe_live_inert/features_window14_19.npz")
        X = dict(zip(z["ids"], z["X"][list(z["sites"]).index("memory_result_end")]))
    tr = [l["id"] for l in labels if l["framing"] == "live" and l["split"] == "train" and l["id"] in beh and l["id"] in X]
    Xt = np.stack([X[i] for i in tr]).astype(np.float64)
    yt = np.array([beh[i]["fired_any"] / beh[i]["k"] >= 0.5 for i in tr])
    v = Xt[yt].mean(0) - Xt[~yt].mean(0)
    u = v / np.linalg.norm(v)
    return u, float((Xt @ u).std())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--notes", type=int, default=48)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="comma separated condition names (base always runs)")
    ap.add_argument("--probe-alphas", default="2,4,-4", help="SD per window layer along the probe direction")
    ap.add_argument("--dom-alphas", default="4,-4", help="SD per window layer along the difference of means direction")
    ap.add_argument("--random-alphas", default="4", help="SD (probe units) per window layer along a random direction")
    ap.add_argument("--patch-layers", default="6,12,17,22", help="residual indices to patch from the twin (24 layer model)")
    ap.add_argument("--resume", action="store_true", help="keep conditions already in the results file instead of rerunning them")
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
    twin = {}
    for l in labels:
        if l["framing"] == "inert" and l["id"] in built:
            twin[(l["case_id"], l["template_id"])] = l["id"]

    # the same note selection as steering.py (training split, mixed phase 1 outcomes, stratified by wording),
    # restricted to notes whose benign twin was built
    rng = random.Random(args.seed)
    pools = {}
    for i, b in beh.items():
        l = lab.get(i)
        if (l and l["framing"] == "live" and l["split"] == "train" and i in built and 0 < b["fired_first"] < b["k"]
                and (l["case_id"], l["template_id"]) in twin):
            pools.setdefault(l["template_id"], []).append(i)
    for v in pools.values():
        rng.shuffle(v)
    notes = []
    while len(notes) < args.notes and any(pools.values()):
        for t in sorted(pools):
            if pools[t] and len(notes) < args.notes:
                notes.append(pools[t].pop())
    print(f"{len(notes)} notes", flush=True)

    if args.dry_run:
        from test_phase3 import tiny_model  # noqa: E402
        model, tok = tiny_model()
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec["hf"])
        model = M.load_hf(spec)
    model.eval()
    n_layers = model.config.num_hidden_layers
    win = M.window_layers(n_layers)
    win_dec = [l - 1 for l in win]  # decoder layer whose output is residual index l
    pl = [int(x) for x in args.patch_layers.split(",") if x]
    patch_layers = pl if n_layers == 24 else sorted({min(n_layers - 1, max(1, round(p * n_layers / 24))) for p in pl})
    args.patch_layers = patch_layers
    if args.dry_run:
        g = np.random.default_rng(1)
        u = g.normal(size=model.config.hidden_size); u /= np.linalg.norm(u)
        v = g.normal(size=model.config.hidden_size); v /= np.linalg.norm(v)
        sigma, sigma_dom, dinfo = 1.0, 1.0, {"dry_run": True}
    else:
        u, sigma, dinfo = direction(spec, labels, beh, win)
        v, sigma_dom = dom_direction(spec, labels, beh)
    r = np.random.default_rng(12345).normal(size=len(u)); r -= (r @ u) * u; r /= np.linalg.norm(r)

    conds = {"base": ("none", None)}
    al = lambda s: [float(x) for x in s.split(",") if x]  # noqa: E731
    for a in al(args.probe_alphas):
        conds[f"probe_all_{a:+g}"] = ("steer", ({l: ("add", a * sigma) for l in win_dec}, u))
    for a in al(args.dom_alphas):
        conds[f"dom_all_{a:+g}"] = ("steer", ({l: ("add", a * sigma_dom) for l in win_dec}, v))
    for a in al(args.random_alphas):
        conds[f"random_all_{a:+g}"] = ("steer", ({l: ("add", a * sigma) for l in win_dec}, r))
    conds["twin_prompt"] = ("twin", None)
    for L in patch_layers:
        conds[f"patch_twin_L{L}"] = ("patch", L)
    if args.only:
        keep = {"base", *args.only.split(",")}
        unknown = keep - set(conds)
        assert not unknown, f"unknown conditions {unknown}; choose from {list(conds)}"
        conds = {k: c for k, c in conds.items() if k in keep}

    # donor activations for patching: the twin's residual at the end of the memory result and every later token
    donors = {}
    if any(kind == "patch" for kind, _ in conds.values()):
        with torch.no_grad():
            for i in notes:
                p = built[i]
                t = built[twin[(lab[i]["case_id"], lab[i]["template_id"])]]
                m = min(len(p["input_ids"]) - p["positions"]["memory_result_end"],
                        len(t["input_ids"]) - t["positions"]["memory_result_end"])
                hs = model(torch.tensor([t["input_ids"]], device=model.device), output_hidden_states=True).hidden_states
                donors[i] = {L: hs[L][0, -m:].detach() for L in patch_layers}
        print(f"donor activations for {len(donors)} notes, suffix lengths "
              f"{sorted({next(iter(x.values())).shape[0] for x in donors.values()})[:10]}", flush=True)

    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id if isinstance(tok.eos_token_id, int) else 0)
    import sample_behavior as SB  # noqa: E402
    o = OUT / spec["key"]
    o.mkdir(parents=True, exist_ok=True)
    fn = o / ("causal_v2_dryrun.json" if args.dry_run else "causal_v2.json")
    results = {}
    if args.resume and fn.exists():
        old = json.load(open(fn))
        if old.get("notes") == notes:
            results = {c: r for c, r in old["per_note"].items() if c in conds and len(r) == len(notes)}
            print("resuming, already done:", sorted(results), flush=True)
    t0 = time.time()
    for c, (kind, cfg) in conds.items():
        if c in results:
            continue
        hook = None
        if kind == "steer":
            hook = Steer(model, cfg[1], cfg[0])
        elif kind == "patch":
            hook = Patch(model, cfg - 1)
        jobs = [(i, s) for i in notes for s in range(args.k)]
        results[c] = {}
        for b0 in range(0, len(jobs), args.batch):
            chunk = jobs[b0:b0 + args.batch]
            src = (lambda i: built[twin[(lab[i]["case_id"], lab[i]["template_id"])]]) if kind == "twin" else (lambda i: built[i])
            seqs = [src(i)["input_ids"] for i, _ in chunk]
            L = max(map(len, seqs))
            ids = torch.tensor([[pad] * (L - len(s)) + s for s in seqs], device=model.device)
            att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=model.device)
            if kind == "steer":
                hook.starts = torch.tensor([L - len(s) + built[i]["positions"]["memory_result_start"] for (i, _), s in zip(chunk, seqs)],
                                           device=model.device)
            elif kind == "patch":
                hook.donor = [donors[i][cfg] for i, _ in chunk]
            torch.manual_seed(args.seed + b0)
            with torch.no_grad():
                g = model.generate(input_ids=ids, attention_mask=att, do_sample=True, temperature=1.0, top_p=1.0,
                                   max_new_tokens=args.max_new_tokens, pad_token_id=pad)
            for (i, _), row in zip(chunk, g[:, L:]):
                calls = parse_calls(tok.decode(row, skip_special_tokens=False))
                hit = any(SB.matches(cl, lab[i]["judge"]) for cl in calls if isinstance(cl.get("args"), dict))
                rec = results[c].setdefault(i, {"hits": 0, "n": 0, "any_call": 0})
                rec["hits"] += int(hit); rec["n"] += 1; rec["any_call"] += int(bool(calls))
        if hook is not None:
            hook.remove()
        rate = np.mean([x["hits"] / x["n"] for x in results[c].values()])
        print(f"{c:>16}: first step execution {rate:.3f}  ({time.time() - t0:.0f} s)", flush=True)
        fn.write_text(json.dumps(summarize(results, notes, lab, spec, args, win, sigma, sigma_dom, dinfo, beh), indent=1))
    print(json.dumps(json.load(open(fn))["summary"], indent=1))


def summarize(results, notes, lab, spec, args, win, sigma, sigma_dom, dinfo, beh):
    base = {i: x["hits"] / x["n"] for i, x in results["base"].items()}
    cases = {i: lab[i]["case_id"] for i in notes}
    uc = sorted(set(cases.values()))
    idx = {cc: [k for k, i in enumerate(notes) if cases[i] == cc] for cc in uc}
    brng = np.random.default_rng(0)
    summary = {}
    for c, res in results.items():
        if len(res) < len(notes):
            continue
        rates = {i: x["hits"] / x["n"] for i, x in res.items()}
        diffs = np.array([rates[i] - base[i] for i in notes])
        bs = [diffs[np.concatenate([idx[cc] for cc in brng.choice(uc, len(uc))])].mean() for _ in range(1000)]
        summary[c] = {"execution_rate": round(float(np.mean(list(rates.values()))), 3),
                      "minus_baseline": round(float(diffs.mean()), 3),
                      "minus_baseline_ci": [round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])],
                      "any_tool_call_rate": round(float(np.mean([x["any_call"] / x["n"] for x in res.values()])), 3)}
    vllm_first = np.array([beh[i]["fired_first"] / beh[i]["k"] for i in notes])
    hf_base = np.array([base[i] for i in notes])
    return {"model": spec["hf"], "notes": notes, "k": args.k, "window": win, "sigma_probe": sigma, "sigma_dom": sigma_dom,
            "direction_info": dinfo, "patch_layers_residual_index": args.patch_layers, "summary": summary,
            "baseline_check": {"hf_mean": round(float(hf_base.mean()), 3), "vllm_fired_first_mean": round(float(vllm_first.mean()), 3),
                               "corr": round(float(np.corrcoef(hf_base, vllm_first)[0, 1]), 3) if hf_base.std() > 0 else None},
            "per_note": results}


if __name__ == "__main__":
    main()
