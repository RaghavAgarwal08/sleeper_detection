"""Sample trigger turn behaviour for the LIVE notes of behavior_v1 (the only notes --task fired uses).

Wraps demo/sample_behavior.py without editing it. Two differences:
  1. Only live notes are sampled (888 of 1,924 requests), written to interp/data/behavior_v1_live/.
  2. The script's "today must equal the build date" check is replaced by the manifest's start_date. This is only
     correct if vLLM is started with VLLM_SYSTEM_START_DATE set to that same date, so the prompts vLLM renders are
     token for token the prompts the activations were captured on. Run check_vllm.py first; it must print MATCH.

    export VLLM_SYSTEM_START_DATE=$(python -c "import json;print(json.load(open('interp/data/behavior_v1/manifest.json'))['start_date'])")
    vllm serve openai/gpt-oss-20b ...   (see README.md)
    python interp/probe_live_inert/sample_live.py --k 8 --workers 64
    cp interp/data/behavior_v1_live/behavior.jsonl interp/data/behavior_v1/

Resumable. Start it ONCE: concurrent copies append duplicate rows (dedupe by id if that happens).
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
full, live = ROOT / "interp/data/behavior_v1", ROOT / "interp/data/behavior_v1_live"
live.mkdir(exist_ok=True)
start = json.loads((full / "manifest.json").read_text())["start_date"]
labels = {l["id"]: l for l in map(json.loads, open(full / "labels.jsonl"))}
with open(live / "requests.jsonl", "w") as f:
    for line in open(full / "requests.jsonl"):
        if labels[json.loads(line)["id"]]["framing"] == "live":
            f.write(line)
for n in ("labels.jsonl", "manifest.json"):
    shutil.copy(full / n, live / n)
script = ROOT / "demo/sample_behavior.py"
src = script.read_text()
assert src.count("dt.date.today().isoformat()") == 1
src = src.replace("dt.date.today().isoformat()", repr(start))
sys.argv = [str(script), "--data", str(live)] + sys.argv[1:]
exec(compile(src, str(script), "exec"), {"__name__": "__main__", "__file__": str(script)})
