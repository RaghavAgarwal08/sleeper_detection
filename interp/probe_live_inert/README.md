# Linear probes for planted memory notes (behavior_v1)

**Findings: [`results/probe_live_inert/FINDINGS.md`](../../results/probe_live_inert/FINDINGS.md)**

Two tasks on gpt-oss-20b's residual stream at the trigger turn:

* `--task live_inert`: is the planted note a pending instruction (live) or a closed record of the same action (inert)?
* `--task fired`: among live notes only, will the agent actually execute the planted action?

## Files

| File | Runs on | Does |
| --- | --- | --- |
| `capture.py` | GPU, about 42 GB | Caches the residual stream at label free positions for all 1,924 requests (about 10 min on an A100) |
| `probe.py` | CPU | Pre-registered probes, baselines and bootstrap CIs; writes `REPORT_<task>.md` and `report_<task>.json` |
| `export_features.py` | CPU | Saves the layer 14 to 19 window features (20 MB) so later analysis needs no cache |
| `check_vllm.py` | with vLLM running | Confirms vLLM renders the same prompt tokens as the cache (must print MATCH) |
| `sample_live.py` | GPU, vLLM | Samples trigger turn behaviour for the 888 live notes (k samples each), wrapping `demo/sample_behavior.py` |
| `controls.py` | CPU, about 2 min | Exploratory controls for the fired result (case and template confounds, stronger text baselines) |
| `run.sh` | Huy's A100 server | Capture plus live vs inert probe, using the repo's memory guard |

Outputs in `results/probe_live_inert/`, behaviour labels in `interp/data/behavior_v1/behavior.jsonl.gz`.

## What is fixed in advance

From `interp/data/behavior_v1/manifest.json`, written before any activations existed.

| Choice | Value |
| --- | --- |
| Train | split `train`: templates 0 to 3, 34 training domains |
| E1, new templates | `held_out_mixed`, templates 4 and 5, seen domains |
| E2, new domains | `held_out_mixed`, templates 0 to 3, 9 test domains |
| E3, both new (primary) | `test`, templates 4 and 5, test domains |
| Feature | residual stream at `memory_result_end`, mean of resid layers 14 to 19 |
| Secondary site | `entry0.occ0.last` (newest note; always the planted note in this dataset) |
| Probe | standardise, then L2 logistic regression; C by leave one template out CV on train only |
| Baselines | TF-IDF on the note text; difference of means; shuffled labels |
| Uncertainty | 95% bootstrap CIs resampling cases; paired CI for probe minus text |
| Fired label | `fired_any / k >= 0.5` with k = 8 |

The per layer sweep and everything in `controls.py` are exploratory.

## Reproduce only the analysis (laptop, no GPU)

```bash
pip install numpy scipy scikit-learn
gunzip -k interp/data/behavior_v1/behavior.jsonl.gz
python interp/probe_live_inert/controls.py          # uses results/probe_live_inert/features_window14_19.npz
python interp/probe_live_inert/probe.py --text-only --task fired
```

## Reproduce everything (one A100 80 GB, for example a RunPod A100 SXM with the PyTorch template)

Run each line separately. Background jobs (`&`) return immediately; start each one only once.

```bash
# 1. repo, model, activations
cd /workspace && git clone <this repo> sleeper_detection && cd sleeper_detection
pip install -q transformers accelerate scikit-learn huggingface_hub
export HF_HOME=/workspace/hf
hf download openai/gpt-oss-20b --exclude "original/*" --exclude "metal/*"
cd interp && nohup python probe_live_inert/capture.py > capture.log 2>&1 & cd ..
tail -3 interp/capture.log                           # wait for "done: 1924 requests"
python interp/probe_live_inert/probe.py --task live_inert
python interp/probe_live_inert/export_features.py

# 2. vLLM in its own environment (can be installed while capture runs; start it only after capture is done)
pip install -q uv
uv venv /workspace/vllm-env --python 3.12
uv pip install --python /workspace/vllm-env/bin/python "vllm==0.30.0" openai "openai-agents>=0.14.0" pydantic json-repair tqdm openai-harmony
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_SYSTEM_START_DATE=$(python -c "import json;print(json.load(open('interp/data/behavior_v1/manifest.json'))['start_date'])")
nohup /workspace/vllm-env/bin/vllm serve openai/gpt-oss-20b --port 8000 --served-model-name gpt-oss-20b --max-model-len 8192 --gpu-memory-utilization 0.92 --enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss --max-num-seqs 64 > vllm.log 2>&1 &
tail -f vllm.log                                     # wait for "Application startup complete", then Ctrl+C

# 3. behaviour labels and the fired probe
/workspace/vllm-env/bin/python interp/probe_live_inert/check_vllm.py        # must print MATCH
/workspace/vllm-env/bin/python interp/probe_live_inert/sample_live.py --limit 8 --k 8 --workers 8   # smoke test
nohup /workspace/vllm-env/bin/python interp/probe_live_inert/sample_live.py --k 8 --workers 64 > sample.log 2>&1 &
cp interp/data/behavior_v1_live/behavior.jsonl interp/data/behavior_v1/
python interp/probe_live_inert/probe.py --task fired
python interp/probe_live_inert/controls.py
```

Notes. `VLLM_SYSTEM_START_DATE` matters: vLLM writes the date into every gpt-oss prompt, and the dataset was rendered
with the manifest's date. The memory guard in `interp/run_guarded.sh` reads cgroup files that may not exist on cloud
pods, so the commands above call Python directly. The MXFP4 warning from transformers (dequantizing to bf16) is expected.
