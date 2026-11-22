"""GPU: capture one model's residual stream on the 1,924 prompts built by build_prompts.py.

Saves interp/data/phase3/<model>/activations.npz with
  ids            [n]
  layers         [n, 2, L+1, d] float16   residual after embeddings (0) and after every layer, at
                                          memory_result_end (slot 0) and at the note's last token (slot 1)
  window_mean    [n, 2, d] float32        mean over the pre-registered window (models.window_layers)
  window         the residual indices averaged
  sweep_window_mean [n, P, d] float16      window mean at the SWEEP positions (always saved; Ashok and May 2025 found
                                          that the position read matters), names in sweep_positions
The probes in analyze.py read window_mean; the layer sweep and trajectory.py read layers; sweeps.py reads sweep_*.

    python interp/phase3/capture.py --model qwen2.5-7b                   # about 10 to 30 minutes per model
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import models as M  # noqa: E402

DATA = Path(os.environ.get("PHASE3_DATA", REPO / "interp/data/phase3"))
SLOTS = ["memory_result_end", "note"]
# Label free positions for the position sweep: through the user request, the note, the memory result, the message
# end and the tokens just before the reply starts.
SWEEP = ["user_end", "memory_result_start", "note", "memory_result_end", "message_end", "last_minus_2", "last_minus_1",
         "last_prompt"]


def sweep_positions(r):
    p = r["positions"]
    mre = p["memory_result_end"]
    last = p["last_prompt"]
    return [p["user_end"], p.get("memory_result_start", mre), p.get("note", mre), mre, min(mre + 1, last),
            max(last - 2, 0), max(last - 1, 0), last]


def load(spec):
    import torch
    if os.environ.get("PHASE3_TINY"):  # tests only: tiny random model on CPU
        from test_phase3 import tiny_model
        return tiny_model()[0]
    return M.load_hf(spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import torch
    spec = M.get(args.model)
    d = DATA / spec["key"]
    rows = [json.loads(l) for l in open(d / "requests.jsonl")]
    if args.limit:
        rows = rows[: args.limit]
    t0 = time.time()
    m = load(spec)
    print(f"loaded {spec['hf']} in {time.time() - t0:.0f} s", flush=True)
    ids_out, layers, sweep = [], [], []
    win = M.window_layers(m.config.num_hidden_layers)
    with torch.no_grad():
        for n, r in enumerate(rows):
            pos = [r["positions"]["memory_result_end"], r["positions"].get("note", r["positions"]["memory_result_end"])]
            x = torch.tensor([r["input_ids"]], device=m.device)
            o = m(x, output_hidden_states=True)
            h = torch.stack([hs[0, pos] for hs in o.hidden_states], dim=1)  # [2, L+1, d]
            layers.append(h.float().cpu().numpy().astype(np.float16))
            sp = sweep_positions(r)
            sw = torch.stack([o.hidden_states[l][0, sp] for l in win]).float().mean(0)  # [P, d]
            sweep.append(sw.cpu().numpy().astype(np.float16))
            ids_out.append(r["id"])
            if n % 100 == 0:
                print(f"{n + 1}/{len(rows)}  {time.time() - t0:.0f} s", flush=True)
    A = np.stack(layers)
    L = A.shape[2] - 1
    np.savez(d / "activations.npz", ids=np.array(ids_out), layers=A, window=np.array(win),
             window_mean=A[:, :, win].astype(np.float32).mean(2), slots=np.array(SLOTS),
             sweep_window_mean=np.stack(sweep), sweep_positions=np.array(SWEEP))
    print(f"wrote {d / 'activations.npz'}: {A.shape}, window {win[0]} to {win[-1]} of {L}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
