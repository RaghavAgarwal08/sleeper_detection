#!/usr/bin/env bash
# Adaptive attacker against the execution probe (PLAN.md section 12; interp/phase3/attack_probe.py).
# Needs an H200 (141 GB): two vLLM servers (agent gpt-oss-20b, attacker Qwen3-8B) and gpt-oss-20b in transformers.
# About 2 to 3 hours, roughly $10 to $15. Fresh pod, no volume needed:
#   cd /workspace && git clone -q --branch raghav/phase2 --single-branch https://github.com/pndh/sleeper_detection.git
#   apt-get update -qq && apt-get install -y -qq tmux
#   tmux new -d -s atk "bash /workspace/sleeper_detection/runpod/run_attack.sh 2>&1 | tee -a /workspace/run_attack.log"
# If the pod restarts, run the same tmux line again: the attack resumes from the last finished round.
set -euo pipefail
cd /workspace/sleeper_detection
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HOME=/workspace/hf PYTHONPATH=/workspace/sleeper_detection PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
t0=$(date +%s); stamp() { echo "[$(( ($(date +%s) - t0) / 60 )) min] $*"; }
stamp "install (vLLM brings its own torch)"
pip install -q --ignore-installed PyJWT
pip install -q vllm
pip install -q accelerate scikit-learn scipy openai "openai-agents>=0.14.0" pydantic json-repair hf_transfer
python -c "import torch, vllm, transformers; print('torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__); assert torch.cuda.get_device_properties(0).total_memory > 130e9, 'needs an H200 (141 GB)'"
python -c "from huggingface_hub import snapshot_download as s; s('openai/gpt-oss-20b', ignore_patterns=['original/*','metal/*']); s('Qwen/Qwen3-8B')"

stamp "serve the agent (gpt-oss-20b, port 8000) and the attacker (Qwen3-8B, port 8001)"
vllm serve openai/gpt-oss-20b --served-model-name gpt-oss-20b --port 8000 --max-model-len 16384 \
    --gpu-memory-utilization 0.35 --enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss \
    > /workspace/vllm_agent.log 2>&1 &
P1=$!
for i in $(seq 1 90); do curl -sf localhost:8000/health >/dev/null && break; sleep 10; done
curl -sf localhost:8000/health >/dev/null || { echo "agent vLLM did not start"; tail -30 /workspace/vllm_agent.log; exit 1; }
vllm serve Qwen/Qwen3-8B --served-model-name attacker --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.2 \
    > /workspace/vllm_attacker.log 2>&1 &
P2=$!
trap 'kill $P1 $P2 2>/dev/null || true' EXIT
for i in $(seq 1 90); do curl -sf localhost:8001/health >/dev/null && break; sleep 10; done
curl -sf localhost:8001/health >/dev/null || { echo "attacker vLLM did not start"; tail -30 /workspace/vllm_attacker.log; exit 1; }

stamp "attack (monitor, targets, 6 rounds, final evaluation)"
export HF_HUB_OFFLINE=1
python interp/phase3/attack_probe.py --resume ${ATTACK_ARGS:-}
stamp "analysis"
python interp/phase3/analyze_attack.py || true
tar czf /workspace/results_attack.tar.gz results/phase3/attack
stamp "DONE. Download /workspace/results_attack.tar.gz, then terminate the pod."
