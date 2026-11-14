#!/bin/bash
# GPU step for the attention baselines on all 1,924 behavior_v1 prompts, then the CPU analysis.
# Expected about 25 to 40 minutes on one GH200; 2 hours reserved as a margin (you are charged for what you reserve,
# but the job ends, and stops charging, as soon as the script finishes).
#   sbatch --account=<ALLOC> ~/sleeper_detection/deltaai/attention_capture.sbatch
#SBATCH --job-name=sd-attn
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
export ALLOC="${SLURM_JOB_ACCOUNT}"
source "$HOME/sleeper_detection/deltaai/env.sh"
cd "$REPO/interp"
OUT="$WORK/cache/behavior_v1_attn"
py probe_live_inert/attention_capture.py --out "$OUT"
cd "$REPO"
py interp/probe_live_inert/attention_baselines.py --attn "$OUT/attn_features.npz"
cp "$OUT/attn_features.npz" "$OUT/at_calibration.json" results/probe_live_inert/ 2>/dev/null || true
echo "done: results/probe_live_inert/ATTENTION_BASELINES.md"
