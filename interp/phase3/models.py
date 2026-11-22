"""Models for the phase 3 multi-model runs, with what each needs to run (see PLAN.md for why each is here).

Every model must (a) call tools reliably through vLLM, so its behaviour can be sampled with the same agent loop
as gpt-oss-20b, and (b) load in Hugging Face transformers for activation capture. Gated models need a Hugging Face
token with the licence accepted on the model page (export HF_TOKEN=... on the pod; never commit it).

window: the pre-registered probe window, the same relative depth as gpt-oss-20b's layers 14 to 19 of 24
        (0.58 to 0.79 of the depth), computed by window_layers().
"""
from __future__ import annotations

MODELS = {
    # key: hf id, vLLM extra args, gated, approx bf16 GB, GPU to use, family/size for the figures
    "gpt-oss-20b": dict(hf="openai/gpt-oss-20b", vllm=["--tool-call-parser", "openai", "--reasoning-parser", "openai_gptoss"],
                        gated=False, gb=40, gpu="A100 80GB or H100", family="gpt-oss", size_b=21, reasoning=True),
    "qwen2.5-7b": dict(hf="Qwen/Qwen2.5-7B-Instruct", vllm=["--tool-call-parser", "hermes"],
                       gated=False, gb=16, gpu="A100 80GB or H100", family="Qwen2.5", size_b=7.6, reasoning=False),
    "qwen2.5-14b": dict(hf="Qwen/Qwen2.5-14B-Instruct", vllm=["--tool-call-parser", "hermes"],
                        gated=False, gb=30, gpu="A100 80GB or H100", family="Qwen2.5", size_b=14.7, reasoning=False),
    "qwen2.5-32b": dict(hf="Qwen/Qwen2.5-32B-Instruct", vllm=["--tool-call-parser", "hermes"],
                        gated=False, gb=66, gpu="H100 80GB (tight) or 2 GPUs", family="Qwen2.5", size_b=32.8, reasoning=False),
    "qwen3-8b": dict(hf="Qwen/Qwen3-8B", vllm=["--tool-call-parser", "hermes", "--reasoning-parser", "qwen3"],
                     gated=False, gb=17, gpu="A100 80GB or H100", family="Qwen3", size_b=8.2, reasoning=True),
    "llama3.1-8b": dict(hf="meta-llama/Llama-3.1-8B-Instruct", vllm=["--tool-call-parser", "llama3_json"],
                        gated=True, gb=16, gpu="A100 80GB or H100", family="Llama 3.1", size_b=8.0, reasoning=False),
    "mistral-7b": dict(hf="mistralai/Mistral-7B-Instruct-v0.3", vllm=["--tool-call-parser", "mistral"],
                       gated=True, gb=15, gpu="A100 80GB or H100", family="Mistral", size_b=7.2, reasoning=False),
}
PILOT = ["qwen2.5-7b", "qwen2.5-14b", "qwen3-8b", "llama3.1-8b", "mistral-7b"]
WINDOW_REL = (14 / 24, 19 / 24)


def window_layers(n_layers: int) -> list[int]:
    """Residual indices (1 = after decoder layer 0) at the same relative depth as gpt-oss-20b's 14 to 19 of 24."""
    lo = round(WINDOW_REL[0] * n_layers)
    hi = round(WINDOW_REL[1] * n_layers)
    return list(range(lo, hi + 1))


def get(key: str) -> dict:
    if key not in MODELS:
        raise SystemExit(f"unknown model {key}; choose from {', '.join(MODELS)}")
    return {"key": key, **MODELS[key]}


def load_hf(spec: dict, device: str = "cuda:0"):
    """Load a model in bf16. gpt-oss-20b ships MXFP4 weights; they are dequantized to bf16 so activations match the
    phase 1 capture (which ran on a stack that always dequantized)."""
    import torch
    from transformers import AutoModelForCausalLM
    kw = dict(torch_dtype=torch.bfloat16, device_map=device, low_cpu_mem_usage=True)
    if spec["key"].startswith("gpt-oss"):
        try:
            from transformers import Mxfp4Config
            kw["quantization_config"] = Mxfp4Config(dequantize=True)
        except ImportError:
            pass
    m = AutoModelForCausalLM.from_pretrained(spec["hf"], **kw)
    return m.eval()
