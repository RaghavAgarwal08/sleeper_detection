"""CPU tests for the phase 3 pipeline with a tiny random model and a Qwen-style chat template (no downloads).

    python interp/phase3/test_phase3.py          # about 2 to 5 minutes
Covers: tool call parsing for all four formats, prompt rendering and positions, activation capture, the steering
hooks (edits land only from the start position on), a steering dry run, the analysis on fake labels, the report.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

_TINY = None
TEMPLATE = (
    "{%- if tools %}<|im_start|>system\n{{ messages[0]['content'] }}\n\n# Tools\n<tools>\n"
    "{%- for t in tools %}\n{{ t | tojson }}{%- endfor %}\n</tools><|im_end|>\n"
    "{%- endif %}"
    "{%- for m in messages %}"
    "{%- if m['role'] == 'system' %}{%- if not tools %}<|im_start|>system\n{{ m['content'] }}<|im_end|>\n{%- endif %}"
    "{%- elif m['role'] == 'user' %}<|im_start|>user\n{{ m['content'] }}<|im_end|>\n"
    "{%- elif m['role'] == 'assistant' %}<|im_start|>assistant\n{%- if m['content'] %}{{ m['content'] }}{%- endif %}"
    "{%- for tc in m.get('tool_calls') or [] %}\n<tool_call>\n{\"name\": \"{{ tc['function']['name'] }}\", \"arguments\": "
    "{{ tc['function']['arguments'] | tojson }}}\n</tool_call>{%- endfor %}<|im_end|>\n"
    "{%- elif m['role'] == 'tool' %}<|im_start|>user\n<tool_response>\n{{ m['content'] }}\n</tool_response><|im_end|>\n"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}")


def tiny_model():
    """A byte level BPE tokenizer trained on a few prompts plus a 6 layer random Llama model."""
    global _TINY
    if _TINY is None:
        import torch
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        reqs = [json.loads(l) for _, l in zip(range(60), open(REPO / "interp/data/behavior_v1/requests.jsonl"))]
        texts = ["\n".join((m.get("content") or "") for m in r["request"]["messages"]) for r in reqs]
        t = Tokenizer(models.BPE())
        t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        t.decoder = decoders.ByteLevel()
        specials = ["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "<pad>"]
        t.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=600, special_tokens=specials,
                                                         initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", eos_token="<|im_end|>")
        tok.chat_template = TEMPLATE
        torch.manual_seed(0)
        cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                          num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=8192)
        m = LlamaForCausalLM(cfg).eval()
        _TINY = (m, tok)
    return _TINY


def test_parse():
    from steering import parse_calls
    h = 'x<|start|>assistant<|channel|>commentary to=functions.GmailSendEmail <|constrain|>json<|message|>{"to": "a@b.c"}<|call|>'
    assert parse_calls(h) == [{"name": "GmailSendEmail", "args": {"to": "a@b.c"}}]
    q = '<tool_call>\n{"name": "Pay", "arguments": {"amount": 5}}\n</tool_call>'
    assert parse_calls(q) == [{"name": "Pay", "args": {"amount": 5}}]
    l = '{"name": "Pay", "parameters": {"amount": 5}}<|eom_id|>'
    assert parse_calls(l) == [{"name": "Pay", "args": {"amount": 5}}]
    mi = '[TOOL_CALLS] [{"name": "Pay", "arguments": {"amount": 5}}]'
    assert parse_calls(mi) == [{"name": "Pay", "args": {"amount": 5}}]
    assert parse_calls("I will not do that.") == []
    print("parse ok")


def test_hooks():
    import torch
    from steering import Steer
    m, tok = tiny_model()
    ids = torch.tensor([[5, 6, 7, 8, 9, 10], [4, 4, 5, 6, 7, 8]])
    u = np.zeros(64); u[3] = 1.0
    with torch.no_grad():
        base = m(ids, output_hidden_states=True).hidden_states[3]
        st = Steer(m, u, {2: ("add", 10.0)})
        st.starts = torch.tensor([3, 1])
        edit = m(ids, output_hidden_states=True).hidden_states[3]
        st.remove()
    d = (edit - base)[..., 3]
    assert torch.allclose(d[0, :3], torch.zeros(3)) and torch.allclose(d[0, 3:], torch.full((3,), 10.0), atol=1e-4)
    assert torch.allclose(d[1, :1], torch.zeros(1)) and torch.allclose(d[1, 1:], torch.full((5,), 10.0), atol=1e-4)
    with torch.no_grad():
        st = Steer(m, u, {2: ("ablate", 0.0)}); st.starts = torch.tensor([0, 0])
        ab = m(ids, output_hidden_states=True).hidden_states[3]
        st.remove()
    assert torch.allclose(ab[..., 3], torch.zeros_like(ab[..., 3]), atol=1e-5)
    print("hooks ok")


def test_patch_hook():
    import torch
    from causal_v2 import Patch
    m, tok = tiny_model()
    ids = torch.tensor([[5, 6, 7, 8, 9, 10], [4, 4, 5, 6, 7, 8]])
    donor = [torch.full((2, 64), 3.0), torch.full((3, 64), -1.0)]
    with torch.no_grad():
        p = Patch(m, 2); p.donor = donor
        hs = m(ids, output_hidden_states=True).hidden_states
        p.remove()
        base = m(ids, output_hidden_states=True).hidden_states
    assert torch.allclose(hs[3][0, -2:], donor[0]) and torch.allclose(hs[3][1, -3:], donor[1])
    assert torch.allclose(hs[3][0, :-2], base[3][0, :-2]) and torch.allclose(hs[3][1, :-3], base[3][1, :-3])
    print("patch hook ok")


def test_pipeline():
    d = tempfile.mkdtemp()
    env = dict(os.environ, PHASE3_DATA=f"{d}/data", PHASE3_OUT=f"{d}/out", PHASE3_TINY="1")
    run = lambda *a: subprocess.run([sys.executable, *a], env=env, check=True, capture_output=True, text=True)  # noqa: E731
    r = run(str(HERE / "build_prompts.py"), "--model", "qwen2.5-7b", "--limit", "390")
    print(r.stdout.strip().splitlines()[-1])
    rows = [json.loads(l) for l in open(f"{d}/data/qwen2.5-7b/requests.jsonl")]
    m, tok = tiny_model()
    for row in rows[:40]:
        p = row["positions"]
        assert p["user_end"] < p["memory_result_start"] <= p.get("note", p["memory_result_end"]) <= p["memory_result_end"] < p["last_prompt"]
        tail = tok.decode(row["input_ids"][p["memory_result_end"] - 3: p["memory_result_end"] + 1])
        assert tail.rstrip().endswith("}"), tail
    print("positions ok")
    run(str(HERE / "capture.py"), "--model", "qwen2.5-7b")
    z = np.load(f"{d}/data/qwen2.5-7b/activations.npz")
    import models as Mo
    assert z["layers"].shape[1:3] == (2, 7) and z["window"].tolist() == Mo.window_layers(6), (z["layers"].shape, z["window"])
    assert Mo.window_layers(24) == list(range(14, 20)), "gpt-oss-20b window must stay 14 to 19"
    print("capture ok", z["layers"].shape)
    # fake pilot labels from phase 1 behaviour, so the analysis has something to fit
    import gzip
    beh = {b["id"]: b for b in map(json.loads, gzip.open(REPO / "interp/data/behavior_v1/behavior.jsonl.gz", "rt"))}
    with open(f"{d}/data/qwen2.5-7b/behavior_pilot.jsonl", "w") as f:
        for row in rows:
            if row["id"] in beh:
                b = beh[row["id"]]
                f.write(json.dumps({"id": row["id"], "k": b["k"], "fired_any": b["fired_any"], "fired_first": b["fired_first"]}) + "\n")
    r = run(str(HERE / "analyze.py"), "--model", "qwen2.5-7b", "--pilot")
    a = json.load(open(f"{d}/out/qwen2.5-7b/analysis_pilot.json"))
    assert "H1_heldout_cases_rate" in a and a["n_poisoned"] > 50, a.keys()
    print("analyze ok", a["n_poisoned"], a["H1_heldout_cases_rate"]["probe_mre_minus_text_full"]["auroc_within_wording"])
    r = run(str(HERE / "sweeps.py"), "--model", "qwen2.5-7b", "--pilot")
    sw = json.load(open(f"{d}/out/qwen2.5-7b/sweeps.json"))
    assert len(sw["position_sweep_execution_auroc_within_wording"]) == 8 and "path_length" in sw["trajectory_auroc_within_wording_sign_free"]
    print("sweeps ok", sw["position_sweep_execution_auroc_within_wording"])
    r = run(str(HERE / "generic_probe.py"), "--model", "qwen2.5-7b")
    g = np.load(f"{d}/out/qwen2.5-7b/generic_probe_directions.npz")
    assert "dir_mean_of_six" in g.files and g["dir_mean_of_six"].shape == (64,)
    print("generic probe ok", sorted(g.files))
    r = run(str(HERE / "steering.py"), "--model", "qwen2.5-7b", "--dry-run", "--notes", "4", "--k", "2", "--batch", "4",
            "--max-new-tokens", "8")
    s = json.load(open(f"{d}/out/qwen2.5-7b/steering_dryrun.json"))
    assert set(s["summary"]) == {"add_-4", "add_-2", "add_+0", "add_+2", "add_+4", "random_-4", "random_+4", "ablate"}
    print("steering dry run ok")
    r = run(str(HERE / "causal_v2.py"), "--model", "qwen2.5-7b", "--dry-run", "--notes", "4", "--k", "2", "--batch", "4",
            "--max-new-tokens", "8")
    cv = json.load(open(f"{d}/out/qwen2.5-7b/causal_v2_dryrun.json"))
    assert {"base", "probe_all_+4", "dom_all_-4", "random_all_+4", "twin_prompt"} <= set(cv["summary"]), cv["summary"].keys()
    assert sum(k.startswith("patch_twin_L") for k in cv["summary"]) == 4, cv["summary"].keys()
    print("causal v2 dry run ok", sorted(cv["summary"]))
    # report on fake multi model results
    for k, shift in [("qwen2.5-7b", 0.0), ("llama3.1-8b", 0.05)]:
        o = Path(f"{d}/out/{k}"); o.mkdir(parents=True, exist_ok=True)
        a = json.load(open(f"{d}/out/qwen2.5-7b/analysis_pilot.json"))
        a["key"], a["pilot"], a["included"] = k, False, True
        a["layer_sweep_within_wording_auroc"] = list(np.linspace(0.5, 0.8 + shift, 7))
        (o / "analysis.json").write_text(json.dumps(a))
    s["model"] = "test"; Path(f"{d}/out/qwen2.5-7b/steering.json").write_text(json.dumps(s))
    r = run(str(HERE / "report.py"), "--out", f"{d}/report")
    assert Path(f"{d}/report/REPORT_PHASE3.md").exists() and len(list(Path(f"{d}/report/figures").glob("*.png"))) >= 3
    print("report ok", sorted(p.name for p in Path(f"{d}/report/figures").glob("*.png")))


if __name__ == "__main__":
    test_parse()
    test_hooks()
    test_patch_hook()
    test_pipeline()
    print("all phase 3 tests passed")
