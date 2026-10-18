#!/usr/bin/env bash
# Live vs inert linear probe on behavior_v1: capture activations (GPU), then train and evaluate (CPU).
# Run from the repo root on the A100 machine:   bash interp/probe_live_inert/run.sh
# Smoke test first (5 requests, a few minutes):  SMOKE=1 bash interp/probe_live_inert/run.sh
set -euo pipefail

GPT_ENV="${GPT_ENV:-$HOME/.conda/envs/gpt-oss-env/bin/python}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO/interp"

echo "== GPU check"
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv
echo "The A100 is shared with CPDM training. If less than about 42 GB is free, ask Huy to pause it first."

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "== Smoke test: caching 5 requests"
  ./run_guarded.sh "$GPT_ENV" probe_live_inert/capture.py --limit 5 --cache cache/behavior_v1_probe_smoke
  echo "Smoke test passed. Run again without SMOKE=1 for the full dataset."
  exit 0
fi

echo "== Step 1: caching activations for all 1,924 requests (resumable; rerun if it stops)"
./run_guarded.sh "$GPT_ENV" probe_live_inert/capture.py

echo "== Step 2: training and evaluating the probe (CPU)"
cd "$REPO"
"$GPT_ENV" interp/probe_live_inert/probe.py --task live_inert

echo "== Done. Results: results/probe_live_inert/REPORT_live_inert.md and report_live_inert.json"
echo "Remember to resume CPDM if it was paused."
