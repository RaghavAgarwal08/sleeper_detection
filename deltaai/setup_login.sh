#!/usr/bin/env bash
# One time setup, run on the DeltaAI login node (no GPU needed, about 15 minutes, mostly the model download).
#   ALLOC=<your allocation> bash ~/sleeper_detection/deltaai/setup_login.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
[[ "$ALLOC" != "CHANGE_ME" ]] || { echo "Set ALLOC first (run: accounts)"; exit 1; }
echo "architecture: $(uname -m)   container: $CONTAINER"
[[ -f "$CONTAINER" ]] || { echo "Container not found. Pick one from: ls /sw/user/NGC_containers/ | grep pytorch"; exit 1; }
# Python packages on top of the container's PyTorch, installed to $PYTHONUSERBASE on /work.
apptainer exec --bind /work,"$HOME" "$CONTAINER" env PYTHONUSERBASE="$PYTHONUSERBASE" \
  python3 -m pip install --user -q "transformers>=4.56" accelerate huggingface_hub scikit-learn scipy
# Model weights (login nodes have internet; compute nodes may not, so download here).
apptainer exec --bind /work,"$HOME" "$CONTAINER" env PYTHONUSERBASE="$PYTHONUSERBASE" HF_HOME="$HF_HOME" \
  python3 -c "from huggingface_hub import snapshot_download as s; print(s('openai/gpt-oss-20b', ignore_patterns=['original/*','metal/*']))"
apptainer exec --bind /work,"$HOME" "$CONTAINER" env PYTHONUSERBASE="$PYTHONUSERBASE" \
  python3 -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
echo "setup done. Next: sbatch --account=$ALLOC ~/sleeper_detection/deltaai/smoke_test.sbatch"
