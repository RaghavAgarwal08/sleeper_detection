"""GPU: features from OTHER language models reading the same prompt text, for the text baseline comparison.

The strongest objection to the probe is that any good reader of the prompt could predict execution. Here models other
than gpt-oss-20b read exactly the text the TF-IDF baseline sees (all message contents joined), and we keep the mean of
the residual stream over the same relative depth as gpt-oss's window (layers 14 to 19 of 24), at the last token and
averaged over all tokens. interp/probe_live_inert/text_baselines.py then fits the same probe on these features with the
same folds. If another model's reading of the text predicts gpt-oss's execution as well as gpt-oss's own state, the
signal is about the text; if gpt-oss's own state wins, it carries something about gpt-oss's own decision.

    python runpod/text_baseline_lm.py --model Qwen/Qwen3-8B            # about 5 minutes on an H100
Writes results/probe_live_inert/lm_text_<name>.npz with ids, last [n, d], mean [n, d].
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "interp/data/behavior_v1"
RES = Path(os.environ.get("TEXT_LM_OUT", REPO / "results/probe_live_inert"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import torch
    labels = [json.loads(l) for l in open(DATA / "labels.jsonl")]
    beh = {b["id"] for b in map(json.loads, gzip.open(DATA / "behavior.jsonl.gz", "rt"))}
    reqs = {r["id"]: r for r in map(json.loads, open(DATA / "requests.jsonl"))}
    ids = [l["id"] for l in labels if l["framing"] == "live" and l["id"] in beh]
    if args.limit:
        ids = ids[: args.limit]
    if os.environ.get("TEXT_LM_TINY"):  # tests only
        sys.path.insert(0, str(REPO / "interp/phase3"))
        from test_phase3 import tiny_model
        model, tok = tiny_model()
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    L = model.config.num_hidden_layers
    win = sorted({min(L, max(1, round(l * L / 24))) for l in range(14, 20)})
    last, mean = [], []
    t0 = time.time()
    with torch.no_grad():
        for n, i in enumerate(ids):
            text = "\n".join((m.get("content") or "") for m in reqs[i]["request"]["messages"])
            x = tok(text, return_tensors="pt", truncation=True, max_length=args.max_tokens).to(model.device)
            hs = model(**x, output_hidden_states=True).hidden_states
            h = torch.stack([hs[l][0] for l in win]).float().mean(0)  # [T, d]
            last.append(h[-1].cpu().numpy()); mean.append(h.mean(0).cpu().numpy())
            if n % 200 == 0:
                print(f"{n + 1}/{len(ids)} {time.time() - t0:.0f} s", flush=True)
    name = args.model.split("/")[-1].lower()
    RES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RES / f"lm_text_{name}.npz", ids=np.array(ids), last=np.stack(last).astype(np.float16),
                        mean=np.stack(mean).astype(np.float16), window=np.array(win), model=args.model)
    print("wrote", RES / f"lm_text_{name}.npz", np.stack(last).shape)


if __name__ == "__main__":
    main()
