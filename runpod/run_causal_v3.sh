#!/usr/bin/env bash
# Third gpt-oss-20b GPU run, pre-registered H6 (interp/phase3/PLAN.md section 6): the difference of means direction
# on held out poisoned notes and on clean prompts, with the user's task scored (interp/phase3/causal_v3.py).
# About 4 hours on one H100, roughly $18 to $20. Fresh pod with the sleeper-detection volume at /workspace:
#   cd /workspace/sleeper_detection && git fetch -q origin raghav/phase2 && git reset -q --hard FETCH_HEAD
#   tmux new -d -s v3 "bash runpod/run_causal_v3.sh 2>&1 | tee -a /workspace/run_causal_v3.log"
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
if [[ ! -f $D/activations.npz ]]; then
  stamp "phase 3 data missing from the volume, rebuilding (about 10 minutes)"
  python interp/phase3/build_prompts.py --model gpt-oss-20b
  python interp/phase3/capture.py --model gpt-oss-20b
fi
stamp "causal v3 (H6)"
python interp/phase3/causal_v3.py --model gpt-oss-20b --k 8 --batch 16 --max-new-tokens 1536 ${CAUSAL_ARGS:-}
tar czf /workspace/results_causal_v3.tar.gz results/phase3/gpt-oss-20b/causal_v3.json results/phase3/gpt-oss-20b/causal_v3_text.jsonl.gz
stamp "DONE. Download /workspace/results_causal_v3.tar.gz, then stop the pod."
