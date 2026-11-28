#!/usr/bin/env bash
# Attention head patching from the planted note to the action (PLAN.md section 16; interp/phase3/patch_heads.py).
# No vLLM. About 2 hours on one H200 (an H100 80GB should also fit), roughly $8. Fresh pod, no volume needed:
#   cd /workspace && git clone -q --branch raghav/phase2 --single-branch https://github.com/pndh/sleeper_detection.git
#   apt-get update -qq && apt-get install -y -qq tmux
#   tmux new -d -s ph "bash /workspace/sleeper_detection/runpod/run_patch_heads.sh 2>&1 | tee -a /workspace/run_patch_heads.log"
# If the pod restarts, run the same tmux line again: finished stages and conditions are kept.
set -euo pipefail
cd /workspace/sleeper_detection
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HOME=/workspace/hf PYTHONPATH=/workspace/sleeper_detection MODEL_ID=openai/gpt-oss-20b
t0=$(date +%s); stamp() { echo "[$(( ($(date +%s) - t0) / 60 )) min] $*"; }
stamp "setup"
python -c "import torch; print('torch==' + torch.__version__.split('+')[0])" > /tmp/pin.txt
pip install -q --ignore-installed PyJWT
pip install -q -U -c /tmp/pin.txt "transformers>=4.56" accelerate huggingface_hub scikit-learn scipy \
    openai "openai-agents>=0.14.0" pydantic json-repair
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
python -c "from huggingface_hub import snapshot_download as s; s('openai/gpt-oss-20b', ignore_patterns=['original/*','metal/*'])"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ -f interp/data/phase3/gpt-oss-20b/requests.jsonl ]] || { stamp "building prompts"; python interp/phase3/build_prompts.py --model gpt-oss-20b; }
stamp "patch heads (gate, discovery, behaviour)"
python interp/phase3/patch_heads.py --model gpt-oss-20b --resume ${PATCH_ARGS:-}
stamp "analysis"
python interp/phase3/analyze_patch_heads.py --model gpt-oss-20b || true
tar czf /workspace/results_patch_heads.tar.gz results/phase3/gpt-oss-20b/patch_heads
stamp "DONE. Download /workspace/results_patch_heads.tar.gz, then terminate the pod."
