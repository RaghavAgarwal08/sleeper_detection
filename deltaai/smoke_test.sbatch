#!/bin/bash
# Checks the GPU, loads gpt-oss-20b and runs the attention capture on 4 prompts. About 10 minutes of 1 GPU.
#   sbatch --account=<ALLOC> ~/sleeper_detection/deltaai/smoke_test.sbatch
#SBATCH --job-name=sd-smoke
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
export ALLOC="${SLURM_JOB_ACCOUNT}"
source "$HOME/sleeper_detection/deltaai/env.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$REPO/interp"
py -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
py probe_live_inert/test_attention_baselines.py
py probe_live_inert/attention_capture.py --limit 4 --out "$WORK/cache/behavior_v1_attn_smoke"
echo "smoke test ok"
