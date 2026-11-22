#!/usr/bin/env bash
# One time setup on a fresh RunPod pod (PyTorch template, 1 x H100 80GB or A100 80GB, 100 GB+ volume at /workspace).
#   curl -sL https://raw.githubusercontent.com/pndh/sleeper_detection/raghav/phase2/runpod/setup_pod.sh | bash
# About 10 minutes, mostly the gpt-oss-20b download (13 GB).
set -euo pipefail
cd /workspace
if [[ ! -d sleeper_detection ]]; then
  git clone --branch raghav/phase2 --single-branch https://github.com/pndh/sleeper_detection.git
fi
cd sleeper_detection && git pull -q
export HF_HOME=/workspace/hf PIP_BREAK_SYSTEM_PACKAGES=1
# Keep the template's torch (no surprise CUDA upgrade) and replace Debian's PyJWT, which pip cannot uninstall.
python -c "import torch; print('torch==' + torch.__version__.split('+')[0])" > /tmp/pin.txt
pip install -q --ignore-installed PyJWT
pip install -q -U -c /tmp/pin.txt "transformers>=4.56" accelerate huggingface_hub scikit-learn scipy matplotlib \
    openai "openai-agents>=0.14.0" pydantic json-repair
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/2**30), 'GB')"
python -c "from huggingface_hub import snapshot_download as s; print(s('openai/gpt-oss-20b', ignore_patterns=['original/*','metal/*']))"
python interp/phase3/test_phase3.py > /workspace/test_phase3.log 2>&1 && echo "CPU tests passed" || { echo "CPU tests FAILED, see /workspace/test_phase3.log"; tail -20 /workspace/test_phase3.log; }
echo "setup done. Next: bash runpod/run_gptoss.sh"
