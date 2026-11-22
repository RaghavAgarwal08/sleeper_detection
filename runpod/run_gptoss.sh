#!/usr/bin/env bash
# All gpt-oss-20b GPU work for phase 3, in order, on one pod (about 3 to 4 hours on an H100, roughly $10 to $12):
#   1. attention baselines (RQ1)          2. activations at every layer + pre-registered analysis and layer sweep
#   3. steering test (H5, the causal check)          4. report and figures
# Each step writes its results before the next starts, so a crash loses at most one step. Run inside tmux:
#   tmux new -s run   then   bash runpod/run_gptoss.sh 2>&1 | tee /workspace/run_gptoss.log
# When it prints DONE, download /workspace/results_gptoss.tar.gz (Jupyter file browser) and STOP THE POD.
set -euo pipefail
cd /workspace/sleeper_detection
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 MODEL_ID=openai/gpt-oss-20b PYTHONPATH=/workspace/sleeper_detection
STEP=${STEP:-1}
t0=$(date +%s); stamp() { echo "[$(( ($(date +%s) - t0) / 60 )) min] $*"; }

if (( STEP <= 1 )); then
  stamp "1/4 attention capture"
  (cd interp && python probe_live_inert/attention_capture.py --limit 4 --out /workspace/attn_smoke)
  (cd interp && python probe_live_inert/attention_capture.py --out /workspace/attn)
  python interp/probe_live_inert/attention_baselines.py --attn /workspace/attn/attn_features.npz
fi
if (( STEP <= 2 )); then
  stamp "2/4 activations and pre-registered analysis"
  python interp/phase3/build_prompts.py --model gpt-oss-20b
  python interp/phase3/capture.py --model gpt-oss-20b
  python interp/phase3/analyze.py --model gpt-oss-20b
fi
if (( STEP <= 3 )); then
  stamp "3/4 steering"
  python interp/phase3/steering.py --model gpt-oss-20b --notes 48 --k 8 --batch 32 --max-new-tokens 1536
fi
stamp "4/4 report"
python interp/phase3/report.py
tar czf /workspace/results_gptoss.tar.gz results/phase3 results/probe_live_inert/ATTENTION_BASELINES.md \
    results/probe_live_inert/attention_baselines.json /workspace/attn/at_calibration.json 2>/dev/null || true
stamp "DONE. Download /workspace/results_gptoss.tar.gz, then stop the pod."
