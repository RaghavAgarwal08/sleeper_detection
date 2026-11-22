"""GPU: is the probe's execution direction causal? Steer along it and measure first step execution (PLAN.md H5).

Direction u: the execution probe (standardize, L2 logistic regression, C = 0.01, the phase 1 choice) trained on the
window mean at the end of the memory result for poisoned notes in the training split, mapped back to raw residual
units and normalized. sigma = standard deviation of the poisoned notes' projections on u.

Conditions (pre-registered), each applied from the first token of the memory result onward, including every
generated token:
  add      alpha * sigma * u added to the residual after the first window layer, alpha in {-4, -2, 0, +2, +4}
  random   the same edit along a fixed random unit direction, alpha in {-4, +4} (controls for "any push breaks it")
  ablate   the component along u removed at every window layer
Outcome: in k sampled first responses (T = 1.0), the share that call the planted action (the case's own judge on the
first response, i.e. phase 1's fired_first). The full 4 step loop needs vLLM, which cannot be hooked; fired_first and
fired_any agree closely in phase 1, and the baseline condition is checked against vLLM's fired_first.
Notes: up to --notes poisoned training notes whose phase 1 fired_first rate is between 1/8 and 7/8, stratified by
wording (the notes where a push could move the outcome either way).

    python interp/phase3/steering.py --model gpt-oss-20b --notes 48 --k 8      # about 1 to 2 GPU hours on an H100
Writes results/phase3/<model>/steering.json.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
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

DATA = Path(os.environ.get("PHASE3_DATA", REPO / "interp/data/phase3"))
OUT = Path(os.environ.get("PHASE3_OUT", REPO / "results/phase3"))
ALPHAS = [-4, -2, 0, 2, 4]


# ------------------------------------------------------------------------------------- tool call parsing
def parse_calls(text: str) -> list[dict]:
    """Tool calls in a raw completion, for harmony (gpt-oss), hermes (Qwen), Llama 3.1 JSON and Mistral formats."""
    calls = []
    for name, body in re.findall(r"to=functions\.([\w.]+)[^<]*(?:<\|constrain\|>\w+)?<\|message\|>(.*?)(?:<\|call\|>|$)", text, re.S):
        calls.append({"name": name, "args": _json(body)})
    if calls:
        return calls
    for body in re.findall(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", text, re.S):
        o = _json(body)
        if isinstance(o, dict) and "name" in o:
            calls.append({"name": o["name"], "args": o.get("arguments", o.get("parameters", {})) or {}})
    if calls:
        return calls
    m = re.search(r"\[TOOL_CALLS\]\s*(\[.*\])", text, re.S)
    if m:
        o = _json(m.group(1))
        for c in o if isinstance(o, list) else []:
            if isinstance(c, dict) and "name" in c:
                calls.append({"name": c["name"], "args": c.get("arguments", {}) or {}})
        return calls
    for body in re.findall(r"(\{\s*\"name\"\s*:.*?\})\s*(?:<\|eom_id\|>|<\|eot_id\|>|$|;)", text, re.S):  # Llama 3.1 JSON
        o = _json(body)
        if isinstance(o, dict) and "name" in o:
            calls.append({"name": o["name"], "args": o.get("parameters", o.get("arguments", {})) or {}})
    return calls


def _json(s):
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return {"_raw": s}


# ---------------------------------------------------------------------------------------------- hooks
class Steer:
    """Forward hooks on decoder layers. starts[b] = first steered position of row b in the padded prompt."""

    def __init__(self, model, u, edits: dict):
        import torch
        self.model, self.handles, self.starts = model, [], None
        self.u = torch.tensor(u, dtype=torch.float32, device=model.device)
        layers = model.model.layers
        for layer_idx, (kind, coef) in edits.items():
            self.handles.append(layers[layer_idx].register_forward_hook(self._hook(kind, coef), prepend=True))

    def _hook(self, kind, coef):
        import torch

        def fn(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hf = h.float()
            T = hf.shape[1]
            if T > 1:  # prompt pass: only from each row's start position
                mask = (torch.arange(T, device=h.device)[None, :] >= self.starts[:, None]).float()[..., None]
            else:      # generation steps: always
                mask = torch.ones(hf.shape[0], 1, 1, device=h.device)
            if kind == "add":
                hf = hf + mask * coef * self.u
            elif kind == "ablate":
                hf = hf - mask * (hf @ self.u)[..., None] * self.u
            h2 = hf.to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


def direction(spec, labels, beh, window):
    """Execution probe direction in raw residual units, from the window mean at the end of the memory result."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    act = DATA / spec["key"] / "activations.npz"
    if act.exists():
        z = np.load(act)
        X = dict(zip(z["ids"], z["window_mean"][:, 0]))
    else:  # gpt-oss-20b phase 1 features (same window, same position)
        z = np.load(REPO / "results/probe_live_inert/features_window14_19.npz")
        X = dict(zip(z["ids"], z["X"][list(z["sites"]).index("memory_result_end")]))
    tr = [l["id"] for l in labels if l["framing"] == "live" and l["split"] == "train" and l["id"] in beh and l["id"] in X]
    Xt = np.stack([X[i] for i in tr]).astype(np.float64)
    yt = np.array([beh[i]["fired_any"] / beh[i]["k"] >= 0.5 for i in tr]).astype(int)
    sc = StandardScaler().fit(Xt)
    lr = LogisticRegression(C=1e-2, max_iter=5000).fit(sc.transform(Xt), yt)
    w = lr.coef_[0] / sc.scale_
    u = w / np.linalg.norm(w)
    proj = Xt @ u
    return u, float(proj.std()), {"n_train": len(tr), "proj_mean_reached": float(proj[yt == 1].mean()),
                                  "proj_mean_resisted": float(proj[yt == 0].mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--notes", type=int, default=48)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
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

    # notes: training split, fired_first between 1/8 and 7/8, stratified by wording
    rng = random.Random(args.seed)
    pools = {}
    for i, b in beh.items():
        l = lab.get(i)
        if l and l["framing"] == "live" and l["split"] == "train" and i in built and 0 < b["fired_first"] < b["k"]:
            pools.setdefault(l["template_id"], []).append(i)
    for v in pools.values():
        rng.shuffle(v)
    notes = []
    while len(notes) < args.notes and any(pools.values()):
        for t in sorted(pools):
            if pools[t] and len(notes) < args.notes:
                notes.append(pools[t].pop())

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
    if args.dry_run:
        u = np.random.default_rng(1).normal(size=model.config.hidden_size); u /= np.linalg.norm(u)
        sigma, dinfo = 1.0, {"dry_run": True}
    else:
        u, sigma, dinfo = direction(spec, labels, beh, win)
    r = np.random.default_rng(12345).normal(size=len(u)); r -= (r @ u) * u; r /= np.linalg.norm(r)
    first = win[0] - 1  # residual index l is the output of decoder layer l - 1
    conds = {f"add_{a:+d}": ({first: ("add", a * sigma)}, u) for a in ALPHAS}
    conds.update({f"random_{a:+d}": ({first: ("add", a * sigma)}, r) for a in (-4, 4)})
    conds["ablate"] = ({l - 1: ("ablate", 0.0) for l in win}, u)

    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id if isinstance(tok.eos_token_id, int) else 0)
    import sample_behavior as SB  # noqa: E402  (judge from demo/)
    results = {c: {} for c in conds}
    t0 = time.time()
    for c, (edits, vec) in conds.items():
        st = Steer(model, vec, edits)
        jobs = [(i, s) for i in notes for s in range(args.k)]
        for b0 in range(0, len(jobs), args.batch):
            chunk = jobs[b0:b0 + args.batch]
            seqs = [built[i]["input_ids"] for i, _ in chunk]
            L = max(map(len, seqs))
            ids = torch.tensor([[pad] * (L - len(s)) + s for s in seqs], device=model.device)
            att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=model.device)
            st.starts = torch.tensor([L - len(s) + built[i]["positions"]["memory_result_start"] for (i, _), s in zip(chunk, seqs)],
                                     device=model.device)
            torch.manual_seed(args.seed + b0)
            with torch.no_grad():
                g = model.generate(input_ids=ids, attention_mask=att, do_sample=True, temperature=1.0, top_p=1.0,
                                   max_new_tokens=args.max_new_tokens, pad_token_id=pad)
            for (i, _), row in zip(chunk, g[:, L:]):
                text = tok.decode(row, skip_special_tokens=False)
                calls = parse_calls(text)
                hit = any(SB.matches(cl, lab[i]["judge"]) for cl in calls if isinstance(cl.get("args"), dict))
                rec = results[c].setdefault(i, {"hits": 0, "n": 0, "any_call": 0})
                rec["hits"] += int(hit); rec["n"] += 1; rec["any_call"] += int(bool(calls))
        st.remove()
        rate = np.mean([v["hits"] / v["n"] for v in results[c].values()])
        print(f"{c:>10}: first step execution {rate:.3f}  ({time.time() - t0:.0f} s)", flush=True)

    # summary with case bootstrap CIs on the difference from the unsteered baseline
    base = {i: v["hits"] / v["n"] for i, v in results["add_+0"].items()}
    cases = {i: lab[i]["case_id"] for i in notes}
    brng = np.random.default_rng(0)
    summary = {}
    for c in conds:
        rates = {i: v["hits"] / v["n"] for i, v in results[c].items()}
        diffs = np.array([rates[i] - base[i] for i in notes])
        uc = sorted(set(cases.values()))
        idx = {cc: [k for k, i in enumerate(notes) if cases[i] == cc] for cc in uc}
        bs = [diffs[np.concatenate([idx[cc] for cc in brng.choice(uc, len(uc))])].mean() for _ in range(1000)]
        summary[c] = {"execution_rate": round(float(np.mean(list(rates.values()))), 3),
                      "minus_baseline": round(float(diffs.mean()), 3),
                      "minus_baseline_ci": [round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])],
                      "any_tool_call_rate": round(float(np.mean([v["any_call"] / v["n"] for v in results[c].values()])), 3)}
    vllm_first = np.array([beh[i]["fired_first"] / beh[i]["k"] for i in notes])
    hf_base = np.array([base[i] for i in notes])
    out = {"model": spec["hf"], "notes": notes, "k": args.k, "window": win, "inject_after_decoder_layer": first,
           "sigma": sigma, "direction_info": dinfo, "summary": summary,
           "baseline_check": {"hf_mean": round(float(hf_base.mean()), 3), "vllm_fired_first_mean": round(float(vllm_first.mean()), 3),
                              "corr": round(float(np.corrcoef(hf_base, vllm_first)[0, 1]), 3) if hf_base.std() > 0 else None},
           "per_note": results}
    o = OUT / spec["key"]
    o.mkdir(parents=True, exist_ok=True)
    (o / ("steering_dryrun.json" if args.dry_run else "steering.json")).write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
