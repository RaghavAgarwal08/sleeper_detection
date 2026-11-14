from __future__ import annotations

import contextlib
from typing import Any, Callable, Iterable

ATTNDA_CATALOG = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen2-1.5b": "Qwen/Qwen2-1.5B-Instruct",
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma2-9b": "google/gemma-2-9b-it",
    "gemma2-2b": "google/gemma-2-2b-it",
}


def resolve_model_id(key_or_id: str) -> str:
    return ATTNDA_CATALOG.get(key_or_id.lower(), key_or_id)


def load_model_and_tokenizer(
    model_name_or_key: str = "qwen2-1.5b",
    device: str = "cuda:0",
    torch_dtype: Any = None,
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    model_id = resolve_model_id(model_name_or_key)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device if torch.cuda.is_available() and device.startswith("cuda") else "cpu",
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    model.eval()
    return model, tok


def locate_subspan_tokens(
    tokenizer: Any,
    full_text: str,
    target_substring: str,
) -> tuple[int, int]:
    """Finds [start, end) token indices of target_substring inside full_text.
    Uses add_special_tokens=False to prevent double-BOS offset corruption."""
    if target_substring not in full_text:
        raise ValueError("target_substring not present in full_text")

    char_start = full_text.index(target_substring)
    char_end = char_start + len(target_substring)

    prefix_tokens = len(tokenizer.encode(full_text[:char_start], add_special_tokens=False))
    target_tokens = len(tokenizer.encode(full_text[char_start:char_end], add_special_tokens=False))

    return (prefix_tokens, prefix_tokens + target_tokens)


def forward_pass(
    model: Any,
    input_ids: list[int] | Any,
    positions: Iterable[int] | None = None,
    output_attentions: bool = True,
    output_hidden_states: bool = True,
) -> dict:
    import torch

    with torch.no_grad():
        if isinstance(input_ids, list):
            x = torch.tensor([input_ids], device=model.device)
        else:
            x = input_ids.to(model.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)

        out = model(
            x,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        seq_len = x.shape[1]
        sel = torch.arange(seq_len) if positions is None else torch.tensor(list(positions))

        result = {
            "positions": sel.tolist(),
            "logits": out.logits[0, -1].float().cpu(),
        }

        if output_hidden_states and out.hidden_states is not None:
            result["resid"] = torch.stack([h[0, sel] for h in out.hidden_states], dim=1).float().cpu()

        if output_attentions and out.attentions is not None:
            result["attn"] = [a[0][:, sel].float().cpu().numpy() for a in out.attentions]

        return result


@contextlib.contextmanager
def hook_residual_stream(
    model: Any,
    edits: dict[int, Callable[[Any], Any]],
    recorded: dict[int, Any] | None = None,
):
    layers = getattr(model, "model", model).layers
    handles = []

    for layer_idx, fn in edits.items():
        def hook(_mod, _inp, output, fn=fn, lyr=layer_idx):
            h = output[0] if isinstance(output, tuple) else output
            h_mod = fn(h).to(h.dtype)
            if recorded is not None:
                recorded[lyr] = h_mod.detach().float().cpu()
            return (h_mod,) + tuple(output[1:]) if isinstance(output, tuple) else h_mod

        handles.append(layers[layer_idx].register_forward_hook(hook, prepend=True))

    try:
        yield
    finally:
        for h in handles:
            h.remove()

