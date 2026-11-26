#!/usr/bin/env bash
# PLAN.md section 9: re-capture activations at all label free positions (including the last prompt token) for
# gpt-oss-20b, Qwen3-8B and Llama-3.1-8B, using the behaviour labels already in the repo (no new sampling), then run
# the pre-registered analysis at the last prompt token. About 30 to 40 minutes on one H100 or H200, no vLLM.
# Fresh pod (no volume needed):
#   cd /workspace && git clone -q --branch raghav/phase2 --single-branch https://github.com/pndh/sleeper_detection.git
#   apt-get update -qq && apt-get install -y -qq tmux
#   tmux new -d -s rc "bash /workspace/sleeper_detection/runpod/run_recapture.sh 2>&1 | tee -a /workspace/run_recapture.log"
set -euo pipefail
cd /workspace/sleeper_detection
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HOME=/workspace/hf PYTHONPATH=/workspace/sleeper_detection PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
t0=$(date +%s); stamp() { echo "[$(( ($(date +%s) - t0) / 60 )) min] $*"; }
stamp "setup"
python -c "import torch; print('torch==' + torch.__version__.split('+')[0])" > /tmp/pin.txt
pip install -q --ignore-installed PyJWT
pip install -q -U -c /tmp/pin.txt "transformers>=4.56" accelerate huggingface_hub scikit-learn scipy matplotlib
for M in qwen3-8b llama3.1-8b gpt-oss-20b; do
  export MODEL_HF_REPO=""; [[ $M == llama3.1-8b ]] && export MODEL_HF_REPO=unsloth/Meta-Llama-3.1-8B-Instruct
  [[ -z "$MODEL_HF_REPO" ]] && unset MODEL_HF_REPO
  D=interp/data/phase3/$M
  stamp "$M: prompts"
  python interp/phase3/build_prompts.py --model "$M"
  if [[ $M == gpt-oss-20b ]]; then zcat interp/data/behavior_v1/behavior_namefix.jsonl.gz > $D/behavior.jsonl
  else zcat $D/behavior.jsonl.gz > $D/behavior.jsonl; fi
  stamp "$M: capture"
  python interp/phase3/capture.py --model "$M"
  stamp "$M: analysis at the last prompt token"
  python interp/phase3/analyze.py --model "$M" --site last_prompt --no-sweep
  rm -f $D/activations.npz
done
tar czf /workspace/results_recapture.tar.gz results/phase3/*/analysis_last_prompt.json results/phase3/*/window_features_last_prompt.npz
stamp "DONE. Download /workspace/results_recapture.tar.gz, then terminate the pod."
