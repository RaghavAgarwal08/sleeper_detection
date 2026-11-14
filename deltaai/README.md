# Running this repo on DeltaAI

DeltaAI is NCSA's GH200 cluster. The CPUs are ARM (aarch64), each GPU is an H100 with 96 GB, so gpt-oss-20b in bf16
(about 39 GB) fits on one GPU. Jobs go through Slurm on partition `ghx4` and are charged to the fellowship allocation for
the time reserved.

## Once you are added to the fellowship group

1. ACCESS profile, scroll to "Resource Provider Site Usernames" for your NCSA username.
2. Set the NCSA password at identity.ncsa.illinois.edu/reset (not your NetID password) and enroll in NCSA Duo.
3. `ssh <ncsa-username>@dtai-login.delta.ncsa.illinois.edu`
4. `accounts` shows the allocation name and its remaining GPU hours. Use it below as `<ALLOC>`.

## First session (login node)

```bash
git clone https://github.com/RaghavAgarwal08/sleeper_detection.git ~/sleeper_detection
cd ~/sleeper_detection && git checkout raghav/phase2        # after push_phase2.sh has been run
ls /sw/user/NGC_containers/ | grep -i pytorch                 # if pytorch_24.09-py3.sif is missing, pick the newest
ALLOC=<ALLOC> bash deltaai/setup_login.sh                     # packages and model download, about 15 minutes
sbatch --account=<ALLOC> deltaai/smoke_test.sbatch            # about 10 GPU minutes
squeue -u $USER                                               # watch it; output lands in sd-smoke-<jobid>.out
```

If the smoke test prints `smoke test ok`:

```bash
sbatch --account=<ALLOC> deltaai/attention_capture.sbatch    # attention baselines, under 1 GPU hour expected
```

Results land in `results/probe_live_inert/ATTENTION_BASELINES.md`.

## Notes

* Put `export ALLOC=<ALLOC>` in `~/.bashrc` so `source deltaai/env.sh` works in every shell.
* Big files go to `/work/hdd/<ALLOC>/$USER` (1 TB default); home has 100 GB and is for code.
* Never train on the login node. For quick debugging use `srun --account=<ALLOC> --partition=ghx4-interactive
  --gpus-per-node=1 --cpus-per-task=8 --mem=32g --time=00:30:00 --pty bash` (charged at 2x, 2 hour maximum), and
  exit as soon as you are done.
* vLLM (only needed to sample new behaviour labels for behavior_v2) is a separate setup on ARM; check for an NGC vLLM
  container under /sw/user/NGC_containers first.
