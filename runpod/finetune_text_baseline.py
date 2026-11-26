"""GPU: a text classifier fine tuned end to end on the full prompt, as the baseline of PLAN.md section 10 (H8).

Same population, soft target (execution rate, corrected labels) and outer folds as text_baselines.py, read from
results/probe_live_inert/folds_heldout_cases.json. The classifier reads exactly the text TF-IDF reads. Inside each outer
training fold, 20% of the cases are held out to choose the epoch by validation loss; test notes are only predicted.

    python runpod/finetune_text_baseline.py --model answerdotai/ModernBERT-large --lr 2e-5 --epochs 4
    python runpod/finetune_text_baseline.py --model Qwen/Qwen3-8B --lora --lr 1e-4 --epochs 3
Writes results/probe_live_inert/ft_text_<name>.npz: ids, oof (logit of the chosen epoch), test logits for every epoch,
validation losses, chosen epochs, and the settings.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "interp/data/behavior_v1"
RES = Path(os.environ.get("FT_OUT", REPO / "results/probe_live_inert"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--limit", type=int, default=0, help="tests only: notes per split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from sklearn.model_selection import GroupShuffleSplit

    F = json.load(open(REPO / "results/probe_live_inert/folds_heldout_cases.json"))
    ids, fold, case = np.array(F["ids"]), np.array(F["fold"]), np.array(F["case"])
    rate = np.array(F["rate"], dtype=np.float32)
    reqs = {r["id"]: r for r in map(json.loads, open(DATA / "requests.jsonl"))}
    text = ["\n".join((m.get("content") or "") for m in reqs[i]["request"]["messages"]) for i in ids]

    tiny = bool(os.environ.get("FT_TINY"))
    if tiny:
        sys.path.insert(0, str(REPO / "interp/phase3"))
        from test_phase3 import tiny_model
        from transformers import LlamaForSequenceClassification
        _, tok = tiny_model()
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = [tok(t, truncation=True, max_length=args.max_tokens)["input_ids"] for t in text]
    print(f"{len(ids)} notes; tokens median {int(np.median([len(e) for e in enc]))}, max {max(len(e) for e in enc)}",
          flush=True)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    def build():
        torch.manual_seed(args.seed)
        if tiny:
            import copy
            cfg = copy.deepcopy(tiny_model()[0].config)
            cfg.num_labels, cfg.pad_token_id = 1, tok.pad_token_id
            m = LlamaForSequenceClassification(cfg)
        else:
            m = AutoModelForSequenceClassification.from_pretrained(
                args.model, num_labels=1, dtype=torch.bfloat16 if args.lora else torch.float32)
            m.config.pad_token_id = tok.pad_token_id
        if args.lora:
            from peft import LoraConfig, get_peft_model
            tm = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=tm,
                                             modules_to_save=["score"], task_type="SEQ_CLS"))
            for p in m.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
            if not tiny:
                m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                m.enable_input_require_grads()
        return m.to(dev)

    def logits(m, rows):
        m.eval(); out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev != "cpu"):
            for i in rows:
                x = torch.tensor([enc[i]], device=dev)
                out.append(float(m(input_ids=x).logits.float().reshape(-1)[0]))
        return np.array(out)

    def bce(z, r):
        p = 1 / (1 + np.exp(-z)); p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(np.mean(-(r * np.log(p) + (1 - r) * np.log(1 - p))))

    folds = [int(f) for f in args.folds.split(",")]
    oof = np.full(len(ids), np.nan, dtype=np.float32)
    per_epoch = np.full((args.epochs, len(ids)), np.nan, dtype=np.float32)
    val_loss, chosen = {}, {}
    t0 = time.time()
    for f in folds:
        tr_all, te = np.where(fold != f)[0], np.where(fold == f)[0]
        a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=f).split(tr_all, groups=case[tr_all]))
        tr, va = tr_all[a], tr_all[b]
        if args.limit:
            tr, va, te = tr[: args.limit], va[: max(2, args.limit // 2)], te[: max(2, args.limit // 2)]
        m = build()
        params = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        steps = args.epochs * math.ceil(len(tr) / args.accum); warm = max(1, int(0.1 * steps))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
        rng = random.Random(args.seed * 100 + f)
        val_loss[f] = []
        for ep in range(args.epochs):
            m.train(); order = list(tr); rng.shuffle(order); tot = 0.0
            for n, i in enumerate(order):
                x = torch.tensor([enc[i]], device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev != "cpu"):
                    z = m(input_ids=x).logits.float().reshape(-1)[:1]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(z, torch.tensor([rate[i]], device=dev))
                (loss / args.accum).backward(); tot += float(loss.detach())
                if (n + 1) % args.accum == 0 or n + 1 == len(order):
                    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step(); opt.zero_grad()
            zv = logits(m, va); vl = bce(zv, rate[va]); val_loss[f].append(vl)
            per_epoch[ep, te] = logits(m, te)
            print(f"fold {f} epoch {ep + 1}: train {tot / len(order):.4f} val {vl:.4f}  {time.time() - t0:.0f} s", flush=True)
        chosen[f] = int(np.argmin(val_loss[f]))
        oof[te] = per_epoch[chosen[f], te]
        print(f"fold {f}: chose epoch {chosen[f] + 1}", flush=True)
        del m, opt
        if dev != "cpu":
            torch.cuda.empty_cache()

    name = args.model.split("/")[-1].lower() + ("_lora" if args.lora else "")
    RES.mkdir(parents=True, exist_ok=True)
    out = RES / f"ft_text_{name}.npz"
    np.savez_compressed(out, ids=ids, oof=oof, per_epoch=per_epoch, folds=np.array(folds),
                        val_loss=json.dumps(val_loss), chosen=json.dumps(chosen),
                        settings=json.dumps(vars(args)))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
