# Source this on DeltaAI (login node or inside a job):   source ~/sleeper_detection/deltaai/env.sh
# Set ALLOC once you know it: run `accounts` after you are added to the fellowship group and copy the project name.
export ALLOC="${ALLOC:-CHANGE_ME}"                       # e.g. the 4 character code `accounts` prints
export WORK="/work/hdd/${ALLOC}/${USER}"                  # big files: model weights, caches, features
export REPO="${HOME}/sleeper_detection"                   # code lives in home (100 GB quota, backed by snapshots)
export HF_HOME="${WORK}/hf"                               # gpt-oss-20b weights, about 13 GB
export PYTHONUSERBASE="${WORK}/pyuser"                    # pip --user installs made inside the container
export APPTAINER_CACHEDIR="${WORK}/apptainer_cache"
# DeltaAI is ARM (aarch64) with GH200 GPUs, so we use NVIDIA's prebuilt PyTorch container instead of pip wheels.
# List what is installed with:  ls /sw/user/NGC_containers/ /sw/external/NGC/ 2>/dev/null | grep -i pytorch
export CONTAINER="${CONTAINER:-/sw/user/NGC_containers/pytorch_24.09-py3.sif}"
export PATH="${PYTHONUSERBASE}/bin:${PATH}"
mkdir -p "$WORK" "$HF_HOME" "$PYTHONUSERBASE" "$APPTAINER_CACHEDIR" 2>/dev/null
# Run python inside the container with the GPU visible and our folders mounted.
# HF_HUB_OFFLINE: weights are downloaded once on the login node; compute nodes may have no internet.
# MODEL_ID: gptoss.py now defaults to a catalogue model; behavior_v1 token ids are gpt-oss-20b's.
py() { apptainer exec --nv --bind /work,/projects,"$HOME" "$CONTAINER" env PYTHONUSERBASE="$PYTHONUSERBASE" HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 MODEL_ID=openai/gpt-oss-20b PYTHONPATH="$REPO" python3 "$@"; }
