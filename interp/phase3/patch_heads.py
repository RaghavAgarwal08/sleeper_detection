"""GPU: attention head patching from the planted note to the action, pre-registered in PLAN.md section 16.

Stages (each saved; --resume skips finished ones)
  0. decision score LD for every poisoned note and twin; gate G1
  1. discovery on half 0: attribution patching over all heads, exact patching of the 48 best and 16 random heads,
     top 16 heads; joint patching of the top 16 on half 1 pairs (P2); attention of the top heads onto the note
  2. behaviour on half 1: base, ablate_top, ablate_random, clean_base, clean_ablate_top (P1)

    python interp/phase3/patch_heads.py --model gpt-oss-20b
    python interp/phase3/patch_heads.py --model qwen2.5-7b --dry-run     # CPU test with the tiny model
Writes results/phase3/<model>/patch_heads/.
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
PREFIX = "<|channel|>commentary to=functions."
M_POS = 8


class Heads:
    """Forward pre hooks on every layer's output projection. The projection's input is the concatenated head outputs
    [b, T, H * dh]. Modes: capture the last M_POS positions; replace given heads at the last M_POS positions with donor
    values; replace given heads with a fixed vector at positions >= starts[b] (and on every generated token)."""

    def __init__(self, model):
        cfg = model.config
        self.H = cfg.num_attention_heads
        self.dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.layers = model.model.layers
        self.mode, self.cap, self.donor, self.heads, self.vec, self.starts, self.keep_grad = None, {}, None, [], None, None, False
        self.hs = [l.self_attn.o_proj.register_forward_pre_hook(self._fn(i)) for i, l in enumerate(self.layers)]

    def _fn(self, li):
        def fn(_m, args):
            x = args[0]
            if self.mode == "capture":
                if self.keep_grad and x.requires_grad:
                    x.retain_grad()
                self.cap[li] = x
                return None
            if self.mode == "patch" and self.donor is not None:
                hs = [h for (l, h) in self.heads if l == li]
                if hs and x.shape[1] > 1:
                    x = x.clone()
                    for h in hs:
                        x[0, -M_POS:, h * self.dh:(h + 1) * self.dh] = self.donor[li][-M_POS:, h * self.dh:(h + 1) * self.dh].to(x.dtype)
                    return (x,) + tuple(args[1:])
            if self.mode == "ablate":
                hs = [h for (l, h) in self.heads if l == li]
                if hs:
                    import torch
                    x = x.clone()
                    T = x.shape[1]
                    if T > 1:
                        mask = (torch.arange(T, device=x.device)[None, :] >= self.starts[:, None])[..., None]
                    else:
                        mask = torch.ones(x.shape[0], 1, 1, dtype=torch.bool, device=x.device)
                    for h in hs:
                        sl = slice(h * self.dh, (h + 1) * self.dh)
                        x[:, :, sl] = torch.where(mask, self.vec[(li, h)].to(x.dtype), x[:, :, sl])
                    return (x,) + tuple(args[1:])
            return None
        return fn

    def remove(self):
        for h in self.hs:
            h.remove()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--exact-top", type=int, default=48)
    ap.add_argument("--exact-random", type=int, default=16)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--notes", type=int, default=48)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--force", action="store_true", help="continue even if gate G1 fails (reported either way)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    import torch
    from scipy.stats import spearmanr
    from sklearn.model_selection import GroupKFold
    import sample_behavior as SB

    spec = M.get(args.model)
    d = DATA / spec["key"]
    o = OUT / spec["key"] / "patch_heads"; o.mkdir(parents=True, exist_ok=True)
    sfx = "_dryrun" if args.dry_run else ""
    t0 = time.time()
    log = lambda *a: print(f"[{(time.time() - t0) / 60:5.1f} min]", *a, flush=True)  # noqa: E731
    labels = [json.loads(l) for l in open(d / "labels.jsonl")]
    lab = {l["id"]: l for l in labels}
    beh = {b["id"]: b for b in map(json.loads, gzip.open(REPO / "interp/data/behavior_v1/behavior_namefix.jsonl.gz", "rt"))}
    built = {json.loads(l)["id"]: json.loads(l) for l in open(d / "requests.jsonl")}
    src = {json.loads(l)["id"]: json.loads(l) for l in open(REPO / "interp/data/behavior_v1/requests.jsonl")}
    utool = user_tools(labels, beh)
    twin = {(l["case_id"], l["template_id"]): l["id"] for l in labels if l["framing"] == "inert" and l["id"] in built}

    if args.dry_run:
        from test_phase3 import tiny_model
        model, tok = tiny_model()
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec["hf"])
        model = M.load_hf(spec)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dev = model.device
    H = Heads(model)
    nL, nH = len(H.layers), H.H
    pre = tok.encode(PREFIX if not args.dry_run else "to=functions.", add_special_tokens=False)

    def first_tok(name):
        full = tok.encode((PREFIX if not args.dry_run else "to=functions.") + name, add_special_tokens=False)
        return full[len(pre)] if full[:len(pre)] == pre and len(full) > len(pre) else None

    poisoned = sorted(i for i in beh if (l := lab.get(i)) and l["framing"] == "live" and i in built and l["case_id"] in utool
                      and (l["case_id"], l["template_id"]) in twin)
    if args.dry_run:
        poisoned = poisoned[:40]
    ptok = {i: (first_tok(lab[i]["judge"]["target_tool"]), first_tok(utool[lab[i]["case_id"]])) for i in poisoned}
    poisoned = [i for i in poisoned if ptok[i][0] is not None and ptok[i][1] is not None and ptok[i][0] != ptok[i][1]]
    rate = {i: beh[i]["fired_any"] / beh[i]["k"] for i in poisoned}
    ids_of = lambda i: built[i]["input_ids"] + pre  # noqa: E731

    def LD(i_prompt, i_note, grad=False):
        x = torch.tensor([ids_of(i_prompt)], device=dev)
        with torch.set_grad_enabled(grad):
            if grad:
                emb = model.get_input_embeddings()(x).detach().requires_grad_(True)
                logits = model(inputs_embeds=emb).logits[0, -1].float()
            else:
                logits = model(x).logits[0, -1].float()
        a, b = ptok[i_note]
        return logits[a] - logits[b]

    # ---------------- stage 0: gate
    f0 = o / f"stage0{sfx}.json"
    if args.resume and f0.exists():
        s0 = json.load(open(f0))
    else:
        log(f"stage 0: decision score for {len(poisoned)} notes and their twins")
        s0 = {"LD": {}, "LD_twin": {}}
        for n, i in enumerate(poisoned):
            s0["LD"][i] = float(LD(i, i))
            tw = twin[(lab[i]["case_id"], lab[i]["template_id"])]
            s0["LD_twin"][i] = float(LD(tw, i))
            if n % 200 == 0:
                log(f"  {n}/{len(poisoned)}")
        tm = np.array([lab[i]["template_id"] for i in poisoned]); ld = np.array([s0["LD"][i] for i in poisoned])
        rr = np.array([rate[i] for i in poisoned])
        rho = [spearmanr(ld[tm == t], rr[tm == t]).statistic for t in np.unique(tm) if (tm == t).sum() > 3 and np.ptp(rr[tm == t]) > 0]
        s0.update({"n": len(poisoned), "within_wording_spearman": [float(x) for x in rho],
                   "mean_within_wording_spearman": float(np.nanmean(rho)) if rho else None,
                   "overall_spearman": float(spearmanr(ld, rr).statistic)})
        s0["G1"] = bool(s0["mean_within_wording_spearman"] is not None and s0["mean_within_wording_spearman"] >= 0.3)
        f0.write_text(json.dumps(s0, indent=1))
    log(f"G1: mean within wording Spearman {s0['mean_within_wording_spearman']}, overall {s0['overall_spearman']:.3f}, "
        f"{'passes' if s0['G1'] else 'FAILS'}")
    if not s0["G1"] and not args.force and not args.dry_run:
        log("gate G1 failed: stopping as pre-registered"); return

    # ---------------- stage 1: discovery
    cases = np.array([lab[i]["case_id"] for i in poisoned])
    fold = {}
    for f, (_, b) in enumerate(GroupKFold(2).split(poisoned, groups=cases)):
        for k in b:
            fold[poisoned[k]] = f
    pairs = [i for i in poisoned if rate[i] >= 0.5 and s0["LD"][i] > s0["LD_twin"][i]]
    P0, P1 = [i for i in pairs if fold[i] == 0], [i for i in pairs if fold[i] == 1]
    if args.max_pairs:
        P0, P1 = P0[: args.max_pairs], P1[: args.max_pairs]
    f1 = o / f"stage1{sfx}.json"
    if args.resume and f1.exists():
        s1 = json.load(open(f1))
    else:
        log(f"stage 1: {len(P0)} discovery pairs, {len(P1)} test pairs")

        def twin_cache(i):
            tw = twin[(lab[i]["case_id"], lab[i]["template_id"])]
            H.mode, H.cap = "capture", {}
            with torch.no_grad():
                model(torch.tensor([ids_of(tw)], device=dev))
            H.mode = None
            return {li: x[0, -M_POS:].detach().clone() for li, x in H.cap.items()}

        attr = np.zeros((nL, nH)); cnt = 0
        for i in P0:
            don = twin_cache(i)
            H.mode, H.cap, H.keep_grad = "capture", {}, True
            ld = LD(i, i, grad=True)
            ld.backward()
            H.mode, H.keep_grad = None, False
            gap = s0["LD"][i] - s0["LD_twin"][i]
            for li, x in H.cap.items():
                g = x.grad[0, -M_POS:].float(); dx = (don[li].float() - x.detach()[0, -M_POS:].float())
                a = (g * dx).reshape(M_POS, nH, H.dh).sum((0, 2))  # first order change in LD from patching each head
                attr[li] += (-a / gap).cpu().numpy()
            cnt += 1
            H.cap = {}
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        attr /= max(cnt, 1)
        order = np.dstack(np.unravel_index(np.argsort(-attr, axis=None), attr.shape))[0]
        cand = [tuple(map(int, x)) for x in order[: args.exact_top]]
        rng = random.Random(0)
        rest = [(l, h) for l in range(nL) for h in range(nH) if (l, h) not in cand]
        rand = rng.sample(rest, min(args.exact_random, len(rest)))

        def patched_effect(heads, plist):
            eff = []
            for i in plist:
                don = twin_cache(i)
                H.mode, H.donor, H.heads = "patch", don, heads
                with torch.no_grad():
                    v = float(LD(i, i))
                H.mode, H.donor, H.heads = None, None, []
                eff.append((s0["LD"][i] - v) / (s0["LD"][i] - s0["LD_twin"][i]))
            return eff
        log(f"exact patching of {len(cand)} + {len(rand)} heads on {len(P0)} pairs")
        exact = {f"{l}.{h}": float(np.mean(patched_effect([(l, h)], P0))) for (l, h) in cand + rand}
        top = sorted(cand, key=lambda x: -exact[f"{x[0]}.{x[1]}"])[: args.top_k]
        joint0 = patched_effect(top, P0)
        joint1 = patched_effect(top, P1)
        s1 = {"n_pairs": [len(P0), len(P1)], "attribution": attr.tolist(), "exact": exact,
              "random_heads": [f"{l}.{h}" for l, h in rand], "top": [f"{l}.{h}" for l, h in top],
              "joint_effect_half0": joint0, "joint_effect_half1": joint1,
              "joint_ids_half1": P1}
        # attention of the top heads from the decision position onto the note's tokens (descriptive, optional)
        try:
            if args.dry_run:
                raise RuntimeError("skipped in dry run")
            model.set_attn_implementation("eager")
            frac = {f"{l}.{h}": [] for l, h in top}
            for i in P0[:30]:
                occ = src[i]["entry_spans"][0]["occurrences"] if src[i].get("entry_spans") else []
                note_pos = sorted({p for a, b in occ for p in range(a, b)})
                with torch.no_grad():
                    att = model(torch.tensor([ids_of(i)], device=dev), output_attentions=True).attentions
                ms, me = built[i]["positions"]["memory_result_start"], built[i]["positions"]["memory_result_end"]
                for l, h in top:
                    w = att[l][0, h, -1].float()
                    mem = w[ms:me + 1].sum().item()
                    nw = w[note_pos].sum().item() if note_pos else float("nan")
                    frac[f"{l}.{h}"].append({"note": nw, "memory_result": mem,
                                             "note_share_of_memory_tokens": len(note_pos) / max(me - ms + 1, 1)})
                del att
            s1["attention_to_note"] = frac
            model.set_attn_implementation("sdpa")
        except Exception as e:  # noqa: BLE001
            s1["attention_to_note"] = {"skipped": str(e)[:200]}
        f1.write_text(json.dumps(s1, indent=1))
    top = [tuple(map(int, x.split("."))) for x in s1["top"]]
    log(f"top heads {s1['top']}; joint patching recovers {np.mean(s1['joint_effect_half0']):.3f} (half 0), "
        f"{np.mean(s1['joint_effect_half1']) if s1['joint_effect_half1'] else float('nan'):.3f} (half 1)")

    # ---------------- stage 2: behaviour on half 1
    layers_top = [l for l, _ in top]
    rng = random.Random(0)
    rand_heads = []
    for l in layers_top:
        choices = [(l, h) for h in range(nH) if (l, h) not in top and (l, h) not in rand_heads]
        rand_heads.append(rng.choice(choices))
    notes = sorted(i for i in poisoned if fold[i] == 1 and beh[i]["fired_first"] >= 2)
    notes = sorted(random.Random(0).sample(notes, min(args.notes, len(notes))))
    cs = sorted({lab[i]["case_id"] for i in notes})
    clean_by = {}
    for l in labels:
        if l["framing"] == "none" and l["id"] in built and l["case_id"] in cs:
            clean_by.setdefault(l["case_id"], l["id"])
    clean = [clean_by[c] for c in cs if c in clean_by]
    # mean head outputs over benign twin prompts, positions from the start of the memory result
    vec = {}
    sums = {hd: None for hd in top + rand_heads}; nvec = 0
    for i in P0[:40] if P0 else notes[:10]:
        tw = twin[(lab[i]["case_id"], lab[i]["template_id"])]
        H.mode, H.cap = "capture", {}
        with torch.no_grad():
            model(torch.tensor([built[tw]["input_ids"]], device=dev))
        H.mode = None
        s = built[tw]["positions"]["memory_result_start"]
        for (l, h) in sums:
            v = H.cap[l][0, s:, h * H.dh:(h + 1) * H.dh].float().mean(0)
            sums[(l, h)] = v if sums[(l, h)] is None else sums[(l, h)] + v
        nvec += 1; H.cap = {}
    vec = {k: v / nvec for k, v in sums.items()}
    conds = {"base": ("poisoned", None), "ablate_top": ("poisoned", top), "ablate_random": ("poisoned", rand_heads),
             "clean_base": ("clean", None), "clean_ablate_top": ("clean", top)}
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id if isinstance(tok.eos_token_id, int) else 0)
    fn, ftext = o / f"behaviour{sfx}.json", o / f"behaviour_text{sfx}.jsonl.gz"
    results = json.load(open(fn))["per_prompt"] if args.resume and fn.exists() else {}
    log(f"stage 2: {len(notes)} notes, {len(clean)} clean prompts; random heads {rand_heads}")
    for c, (pop, heads) in conds.items():
        if c in results:
            continue
        prompts = notes if pop == "poisoned" else clean
        jobs = [(i, s) for i in prompts for s in range(args.k)]
        res, texts = {}, []
        for b0 in range(0, len(jobs), args.batch):
            chunk = jobs[b0:b0 + args.batch]
            seqs = [built[i]["input_ids"] for i, _ in chunk]
            L = max(map(len, seqs))
            ids = torch.tensor([[pad] * (L - len(s)) + s for s in seqs], device=dev)
            att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
            if heads:
                H.mode, H.heads, H.vec = "ablate", heads, vec
                H.starts = torch.tensor([L - len(s) + built[i]["positions"].get("memory_result_start", built[i]["positions"]["memory_result_end"])
                                         for (i, _), s in zip(chunk, seqs)], device=dev)
            torch.manual_seed(b0)
            with torch.no_grad():
                g = model.generate(input_ids=ids, attention_mask=att, do_sample=True, temperature=1.0, top_p=1.0,
                                   max_new_tokens=args.max_new_tokens, pad_token_id=pad)
            H.mode, H.heads = None, []
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
        results[c] = res
        with gzip.open(ftext, "at") as fo:
            for t in texts:
                fo.write(json.dumps(t) + "\n")
        mm = lambda key: np.mean([x[key] / x["n"] for x in res.values()]) if res else float("nan")  # noqa: E731
        log(f"{c:>18}: execution {mm('hits'):.3f}  user task {mm('user'):.3f}  any call {mm('any_call'):.3f}")
        fn.write_text(json.dumps({"notes": notes, "clean": clean, "top": s1["top"], "random": [f"{l}.{h}" for l, h in rand_heads],
                                  "per_prompt": results}, indent=1))
    log("DONE")


if __name__ == "__main__":
    main()
