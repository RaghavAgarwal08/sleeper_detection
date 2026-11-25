#!/usr/bin/env bash
# One open model through the phase 3 pipeline (PLAN.md): its own behaviour labels on all 888 poisoned notes, its own
# activations, the pre-registered analysis, and the files for the cross-model matrix.
#   MODEL=qwen3-8b bash runpod/run_model.sh        # about 2 to 4 hours on one H100 80GB
# Works on any fresh pod (no network volume needed). In the RunPod Web Terminal:
#   cd /workspace && git clone -q --branch raghav/phase2 --single-branch https://github.com/pndh/sleeper_detection.git
#   apt-get update -qq && apt-get install -y -qq tmux
#   tmux new -d -s m "MODEL=qwen3-8b bash /workspace/sleeper_detection/runpod/run_model.sh 2>&1 | tee -a /workspace/run_model.log"
# Steps (each resumable): install vLLM; render prompts with the model's chat template; serve with vLLM; tokenization
# check; pilot (120 notes, k = 4) and the pre-registered pilot gate; full sampling (888 notes, k = 8); stop vLLM;
# capture activations; analysis; archive.
set -euo pipefail
MODEL=${MODEL:?set MODEL, for example MODEL=qwen3-8b}
cd /workspace/sleeper_detection
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HOME=/workspace/hf PYTHONPATH=/workspace/sleeper_detection
t0=$(date +%s); stamp() { echo "[$(( ($(date +%s) - t0) / 60 )) min] $*"; }
HF=$(python -c "import sys; sys.path.insert(0,'interp/phase3'); import models as M; print(M.get('$MODEL')['hf'])")
VARGS=$(python -c "import sys; sys.path.insert(0,'interp/phase3'); import models as M; print(' '.join(M.get('$MODEL')['vllm']))")

stamp "install (vLLM brings its own torch)"
pip install -q --ignore-installed PyJWT
pip install -q vllm
pip install -q scikit-learn scipy matplotlib openai "openai-agents>=0.14.0" pydantic json-repair hf_transfer
python -c "import torch, vllm, transformers; print('torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__)"

stamp "render prompts for $MODEL"
python interp/phase3/build_prompts.py --model "$MODEL"

stamp "serve $HF with vLLM"
vllm serve "$HF" --served-model-name "$MODEL" --port 8000 --max-model-len 16384 --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice $VARGS > /workspace/vllm_$MODEL.log 2>&1 &
VPID=$!
trap 'kill $VPID 2>/dev/null || true' EXIT
for i in $(seq 1 120); do curl -sf localhost:8000/health >/dev/null && break; sleep 10; done
curl -sf localhost:8000/health >/dev/null || { echo "vLLM did not start"; tail -30 /workspace/vllm_$MODEL.log; exit 1; }

stamp "pilot (120 notes, k = 4) and pilot gate"
python interp/phase3/sample.py --model "$MODEL" --pilot
python - <<PY
import json, sys
s = json.load(open("interp/data/phase3/$MODEL/behavior_pilot_summary.json"))
print("pilot gate:", s["passes_pilot_gate"], s)
sys.exit(0 if s["passes_pilot_gate"] else 3)
PY

stamp "full sampling (888 notes, k = 8)"
python interp/phase3/sample.py --model "$MODEL" --skip-check
kill $VPID; wait $VPID 2>/dev/null || true; trap - EXIT; sleep 10

stamp "capture activations"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python interp/phase3/capture.py --model "$MODEL"
stamp "analysis (pre-registered H1, H2, H4 and the layer sweep)"
python interp/phase3/analyze.py --model "$MODEL"
python interp/phase3/sweeps.py --model "$MODEL" || true

D=interp/data/phase3/$MODEL
tar czf /workspace/results_$MODEL.tar.gz $D/behavior.jsonl $D/behavior_pilot.jsonl $D/behavior_pilot_summary.json \
    $D/behavior_summary.json results/phase3/$MODEL 2>/dev/null || true
stamp "DONE. Download /workspace/results_$MODEL.tar.gz, then terminate the pod."
