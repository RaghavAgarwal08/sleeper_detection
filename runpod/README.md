# Running phase 3 on RunPod (paste commands, no API key needed)

1. RunPod console, **Pods**, **Deploy**: pick **1 x H100 80GB** (or A100 80GB), template **RunPod PyTorch 2.x**,
   container disk 50 GB, volume 150 GB mounted at `/workspace`. Deploy, then **Connect**, **Web Terminal**.
2. Setup (about 10 minutes):
   ```bash
   curl -sL https://raw.githubusercontent.com/pndh/sleeper_detection/raghav/phase2/runpod/setup_pod.sh | bash
   ```
   It must end with `CPU tests passed` and `setup done`.
3. Run everything for gpt-oss-20b (about 3 to 4 hours):
   ```bash
   cd /workspace/sleeper_detection && tmux new -s run
   bash runpod/run_gptoss.sh 2>&1 | tee /workspace/run_gptoss.log
   ```
   Detach with `Ctrl+b` then `d`; reattach with `tmux attach -t run`. If a step fails, fix it and resume with
   `STEP=2 bash runpod/run_gptoss.sh` (or 3, 4).
4. When it prints `DONE`, download `/workspace/results_gptoss.tar.gz` from **Connect**, **Jupyter Lab** into
   `Desktop/AI Alignment`, then **Stop** the pod (and **Terminate** it if you do not need the volume).

Paste the last 30 lines of `/workspace/run_gptoss.log` into the chat if anything goes wrong.
