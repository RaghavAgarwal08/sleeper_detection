#!/usr/bin/env bash
# PLAN.md section 10 (H8): text classifiers fine tuned on the full prompt, same folds as the probe. Settings fixed in
# the pre-registration; do not tune them. About 1.5 to 2 hours on one H100 or H200, no vLLM, no Hugging Face login.
# Fresh pod (no volume needed):
#   cd /workspace && git clone -q --branch raghav/phase2 --single-branch https://github.com/pndh/sleeper_detection.git
#   apt-get update -qq && apt-get install -y -qq tmux
#   tmux new -d -s ft "bash /workspace/sleeper_detection/runpod/run_finetune_baselines.sh 2>&1 | tee -a /workspace/run_ft.log"
set -euo pipefail
cd /workspace/sleeper_detection
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
t0=$(date +%s); stamp() { echo "[$(( ($(date +%s) - t0) / 60 )) min] $*"; }
stamp "setup"
python -c "import torch; print('torch==' + torch.__version__.split('+')[0])" > /tmp/pin.txt
pip install -q --ignore-installed PyJWT
pip install -q -U -c /tmp/pin.txt "transformers>=4.56" accelerate huggingface_hub peft scikit-learn
stamp "dry run"
FT_OUT=/tmp/ft_dry python runpod/finetune_text_baseline.py --model Qwen/Qwen3-0.6B --lr 1e-5 --epochs 1 --limit 4 --folds 0
stamp "ModernBERT-large"
python runpod/finetune_text_baseline.py --model answerdotai/ModernBERT-large --lr 2e-5 --epochs 4
stamp "Qwen3-0.6B"
python runpod/finetune_text_baseline.py --model Qwen/Qwen3-0.6B --lr 1e-5 --epochs 4
stamp "Qwen3-8B LoRA"
python runpod/finetune_text_baseline.py --model Qwen/Qwen3-8B --lora --lr 1e-4 --epochs 3
tar czf /workspace/results_ft.tar.gz results/probe_live_inert/ft_text_*.npz
stamp "DONE. Download /workspace/results_ft.tar.gz, then terminate the pod."
