"""Re-render the 1,924 behavior_v1 prompts in another model's own chat format, with label-free positions.

behavior_v1 stores each prompt as OpenAI-style messages (system, user, assistant tool call, tool result) plus tool
schemas. For each model this renders them with the model's Hugging Face chat template exactly as vLLM does
(tool call arguments parsed from JSON strings to dicts first, as vLLM does) and records, per prompt:
  input_ids, and positions  user_end (last token of the user request), note (last token of memory entry 0, the
  planted note), memory_result_end (last token of the tool result), last_prompt (the token the reply starts after).
sample.py later checks the rendering token for token against vLLM's /tokenize, and stops if they differ.

gpt-oss-20b is special-cased: its prompts were rendered with the harmony format when the dataset was built, so the
stored input_ids and positions are reused unchanged.

    python interp/phase3/build_prompts.py --model qwen2.5-7b
Writes interp/data/phase3/<model>/requests.jsonl (and copies labels.jsonl and manifest.json next to it).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import models as M  # noqa: E402

SRC = REPO / "interp/data/behavior_v1"


def vllm_messages(messages: list[dict]) -> list[dict]:
    """vLLM parses assistant tool call arguments into dicts before applying HF chat templates; do the same."""
    out = copy.deepcopy(messages)
    for m in out:
        for tc in m.get("tool_calls") or []:
            a = tc["function"].get("arguments")
            if isinstance(a, str):
                try:
                    tc["function"]["arguments"] = json.loads(a)
                except json.JSONDecodeError:
                    pass
    return out


def char_to_token(offsets: list[tuple[int, int]], char: int) -> int:
    """Index of the token whose span contains char (offsets from a fast tokenizer)."""
    for i, (a, b) in enumerate(offsets):
        if a <= char < b:
            return i
    raise ValueError(f"no token covers character {char}")


def find_end(text: str, needle: str, start: int = 0, last: bool = False) -> int | None:
    """Character index of the last character of needle in text (also tries its JSON-escaped form)."""
    for cand in (needle, json.dumps(needle)[1:-1]):
        i = text.rfind(cand, start) if last else text.find(cand, start)
        if i >= 0 and cand:
            return i + len(cand) - 1
    return None


def render(tok, req: dict) -> dict:
    msgs = vllm_messages(req["request"]["messages"])
    tools = req["request"]["tools"]
    text = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    ref = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=True)
    ref = list(ref["input_ids"] if hasattr(ref, "keys") else ref)
    if ids != ref:
        raise ValueError("re-tokenizing the rendered text does not reproduce the template's token ids")
    user = next(m["content"] for m in msgs if m["role"] == "user")
    tool = next(m["content"] for m in msgs if m["role"] == "tool")
    user_end = find_end(text, user)
    mre = find_end(text, tool, last=True)
    if user_end is None or mre is None:
        raise ValueError("user request or tool result not found verbatim in the rendered prompt")
    tool_start = mre - len(tool) + 1
    note_text = req["entry_spans"][0]["text"] if req.get("entry_spans") else None
    note = find_end(text, note_text, start=max(tool_start - 2, 0)) if note_text else None
    pos = {"user_end": char_to_token(offs, user_end), "memory_result_end": char_to_token(offs, mre),
           "last_prompt": len(ids) - 1}
    if note is not None and note <= mre:
        pos["note"] = char_to_token(offs, note)
    # first token of the tool result, where steering starts
    pos["memory_result_start"] = char_to_token(offs, tool_start)
    return {"id": req["id"], "input_ids": ids, "n_tokens": len(ids), "positions": pos}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=os.environ.get("PHASE3_DATA", str(REPO / "interp/data/phase3")))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    spec = M.get(args.model)
    out = Path(args.out) / spec["key"]
    out.mkdir(parents=True, exist_ok=True)
    reqs = [json.loads(l) for l in open(SRC / "requests.jsonl")]
    if args.limit:
        reqs = reqs[: args.limit]
    for n in ("labels.jsonl", "manifest.json"):
        shutil.copy(SRC / n, out / n)

    rows, bad = [], []
    if spec["key"] == "gpt-oss-20b":
        for r in reqs:
            pos = dict(r["positions"])
            if r["entry_spans"] and r["entry_spans"][0]["occurrences"]:
                pos["note"] = r["entry_spans"][0]["occurrences"][0][1] - 1
            tool_starts = [i for i, t in enumerate(r["input_ids"]) if t == 200008]  # <|message|> of each message
            pos["memory_result_start"] = tool_starts[4] + 1
            rows.append({"id": r["id"], "input_ids": r["input_ids"], "n_tokens": r["n_tokens"], "positions": pos})
    else:
        if os.environ.get("PHASE3_TINY"):  # tests only
            from test_phase3 import tiny_model
            tok = tiny_model()[1]
        else:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(spec["hf"])
        for r in reqs:
            try:
                rows.append(render(tok, r))
            except Exception as e:  # noqa: BLE001
                bad.append((r["id"], str(e)[:200]))
    with open(out / "requests.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    n_note = sum("note" in r["positions"] for r in rows)
    lens = sorted(r["n_tokens"] for r in rows)
    print(f"{spec['key']}: {len(rows)} prompts rendered, {len(bad)} failed, note position found in {n_note}; "
          f"tokens median {lens[len(lens) // 2] if lens else 0}, max {lens[-1] if lens else 0}")
    for b in bad[:5]:
        print("  failed", b)
    (out / "build_report.json").write_text(json.dumps({"rendered": len(rows), "failed": bad, "with_note": n_note}))
    if bad and len(bad) > 0.01 * len(reqs):
        sys.exit("more than 1% of prompts failed to render; inspect before sampling")


if __name__ == "__main__":
    main()
