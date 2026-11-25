#!/usr/bin/env bash
# Neural text baselines: other language models read the same prompt text (runpod/text_baseline_lm.py).
# About 20 to 30 minutes on one GPU with 48 GB or more (an A40, L40S, A100 or H100 all work). Fresh pod with the
# sleeper-detection network volume at /workspace:
#   cd /workspace/sleeper_detection && git fetch -q origin raghav/phase2 && git reset -q --hard FETCH_HEAD
#   tmux new -d -s tb "bash runpod/run_text_baselines.sh 2>&1 | tee -a /workspace/run_text_baselines.log"
set -euo pipefail
cd /workspace/sleeper_detection
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HOME=/workspace/hf
t0=$(date +%s); stamp() { echo "[$(( ($(date +%s) - t0) / 60 )) min] $*"; }
stamp "setup"
python -c "import torch; print('torch==' + torch.__version__.split('+')[0])" > /tmp/pin.txt
pip install -q --ignore-installed PyJWT
pip install -q -U -c /tmp/pin.txt "transformers>=4.56" accelerate huggingface_hub
for M in Qwen/Qwen3-0.6B Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-8B; do
  stamp "$M"
  python runpod/text_baseline_lm.py --model "$M"
done
tar czf /workspace/results_text_lm.tar.gz results/probe_live_inert/lm_text_*.npz
stamp "DONE. Download /workspace/results_text_lm.tar.gz, then stop the pod."
