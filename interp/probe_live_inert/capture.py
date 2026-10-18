"""Step 1 (GPU): cache gpt-oss-20b residual-stream activations for the behavior_v1 dataset.

Saves, for every request, the residual stream (embeddings + 24 layers, float16) at a few label-free positions:
  last_prompt, user_end, memory_result_end, and the last token of the FIRST occurrence of every memory entry
  (entry<i>.occ0.last). Router logits are skipped to save disk. The cache format matches gptoss.load_cache, so
  the existing repo tools can read it too.

Run from the repo's interp/ folder, under the memory guard, with the gpt-oss env:
    cd ~/sleeper_detection/interp
    ./run_guarded.sh ~/.conda/envs/gpt-oss-env/bin/python probe_live_inert/capture.py --limit 5   # smoke test
    ./run_guarded.sh ~/.conda/envs/gpt-oss-env/bin/python probe_live_inert/capture.py             # all 1,924

Resumable: requests already in cache/behavior_v1_probe/index.json are skipped, so you can stop and restart.
Reads only requests.jsonl (the defender view). Labels are never loaded here.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTERP = HERE.parent
sys.path.insert(0, str(INTERP))

import numpy as np  # noqa: E402

DEFAULT_DATA = INTERP / "data" / "behavior_v1"
DEFAULT_CACHE = INTERP / "cache" / "behavior_v1_probe"
MIN_FREE_GPU_MIB = 42_000  # gpt-oss-20b in bf16 needs about 39 GiB plus working memory


def gpu_free_mib() -> int | None:
    if shutil.which("nvidia-smi") is None:
        return None
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True)
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def probe_positions(record: dict) -> dict[str, int]:
    """Label-free positions to cache: the three fixed ones plus the first occurrence of every memory entry."""
    pos = dict(record["positions"])
    for e in record["entry_spans"]:
        if e["occurrences"]:
            a, b = e["occurrences"][0]
            pos[f"entry{e['entry_index']}.occ0.last"] = b - 1
    return pos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(DEFAULT_DATA))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--limit", type=int, default=None, help="only the first N requests (smoke test)")
    ap.add_argument("--skip-gpu-check", action="store_true", help="skip the free GPU memory check")
    args = ap.parse_args()

    if not args.skip_gpu_check:
        free = gpu_free_mib()
        if free is not None and free < MIN_FREE_GPU_MIB:
            sys.exit(f"Only {free} MiB of GPU memory is free; gpt-oss-20b needs about {MIN_FREE_GPU_MIB} MiB. "
                     "The A100 is shared with the CPDM training job: ask Huy to pause it, then run again.")

    import torch  # noqa: E402  (imported late so the GPU check can fail fast)
    import gptoss as G  # noqa: E402

    requests = [json.loads(l) for l in open(Path(args.dataset) / "requests.jsonl")]
    if args.limit:
        requests = requests[: args.limit]
    out = Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}

    todo = [r for r in requests if not (r["id"] in index and (out / f"{r['id']}.npz").exists())]
    print(f"{len(requests)} requests, {len(requests) - len(todo)} already cached, {len(todo)} to go", flush=True)
    if not todo:
        return

    t0 = time.time()
    G.model()  # load once (about 75 s)
    print(f"model loaded in {time.time() - t0:.0f} s", flush=True)

    t1 = time.time()
    for k, r in enumerate(todo, 1):
        pos = probe_positions(r)
        names, idx = list(pos), list(pos.values())
        assert max(idx) < r["n_tokens"], f"position past the end of request {r['id']}"
        o = G.forward(r["input_ids"], positions=idx, router=False)
        np.savez_compressed(out / f"{r['id']}.npz", resid=o["resid"].numpy().astype(np.float16), router=np.zeros(0))
        index[r["id"]] = {"position_names": names, "positions": idx, "n_tokens": r["n_tokens"]}
        if k % 25 == 0 or k == len(todo):
            index_path.write_text(json.dumps(index, indent=1))
            rate = (time.time() - t1) / k
            print(f"cached {k}/{len(todo)}  {rate:.2f} s/request  about {rate * (len(todo) - k) / 60:.0f} min left",
                  flush=True)
        del o
        if k % 100 == 0:
            torch.cuda.empty_cache()
    index_path.write_text(json.dumps(index, indent=1))
    print(f"done: {len(index)} requests in {out}", flush=True)


if __name__ == "__main__":
    main()
