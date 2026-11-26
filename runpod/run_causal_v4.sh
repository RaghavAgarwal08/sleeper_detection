#!/usr/bin/env bash
# H6b (interp/phase3/PLAN.md section 11): cross fitted confirmation of the difference of means defense, with
# CounterSteer style conditions, corrected scorer and all text saved (interp/phase3/causal_v4.py).
# About 5 to 7 hours on one H100 or H200 (roughly $25). Fresh pod, no volume needed:
#   cd /workspace && git clone -q --branch raghav/phase2 --single-branch https://github.com/pndh/sleeper_detection.git
#   apt-get update -qq && apt-get install -y -qq tmux
#   tmux new -d -s v4 "bash /workspace/sleeper_detection/runpod/run_causal_v4.sh 2>&1 | tee -a /workspace/run_causal_v4.log"
# If the pod restarts, run the same tmux line again: --resume keeps finished conditions.
set -euo pipefail
cd /workspace/sleeper_detection
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HOME=/workspace/hf PYTHONPATH=/workspace/sleeper_detection MODEL_ID=openai/gpt-oss-20b
t0=$(date +%s); stamp() { echo "[$(( ($(date +%s) - t0) / 60 )) min] $*"; }
stamp "setup"
python -c "import torch; print('torch==' + torch.__version__.split('+')[0])" > /tmp/pin.txt
pip install -q --ignore-installed PyJWT
pip install -q -U -c /tmp/pin.txt "transformers>=4.56" accelerate huggingface_hub scikit-learn scipy matplotlib \
    openai "openai-agents>=0.14.0" pydantic json-repair
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
python -c "from huggingface_hub import snapshot_download as s; s('openai/gpt-oss-20b', ignore_patterns=['original/*','metal/*'])"
if [[ -z "${SKIP_TESTS:-}" ]]; then
  python interp/phase3/test_phase3.py > /workspace/test_phase3.log 2>&1 && echo "CPU tests passed" || { tail -20 /workspace/test_phase3.log; exit 1; }
fi
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=interp/data/phase3/gpt-oss-20b
[[ -f $D/requests.jsonl ]] || { stamp "building prompts"; python interp/phase3/build_prompts.py --model gpt-oss-20b; }
stamp "causal v4 (H6b)"
python interp/phase3/causal_v4.py --model gpt-oss-20b --k 8 --batch 16 --max-new-tokens 1536 --resume ${CAUSAL_ARGS:-}
tar czf /workspace/results_causal_v4.tar.gz results/phase3/gpt-oss-20b/causal_v4.json results/phase3/gpt-oss-20b/causal_v4_text.jsonl.gz
stamp "DONE. Download /workspace/results_causal_v4.tar.gz, then terminate the pod."
