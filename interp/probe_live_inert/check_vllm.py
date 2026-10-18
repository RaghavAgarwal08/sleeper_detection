"""Check that the running vLLM server renders a behavior_v1 request to the same number of prompt tokens as the
cached activations (interp/data/behavior_v1/requests.jsonl n_tokens). Must print MATCH before sampling."""
import json
from pathlib import Path

import openai

ROOT = Path(__file__).resolve().parents[2]
r = json.loads(open(ROOT / "interp/data/behavior_v1/requests.jsonl").readline())
c = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
o = c.chat.completions.create(model="gpt-oss-20b", messages=r["request"]["messages"], tools=r["request"]["tools"],
                              max_tokens=16)
print("vLLM prompt tokens", o.usage.prompt_tokens, "| cached activation tokens", r["n_tokens"])
print("MATCH" if o.usage.prompt_tokens == r["n_tokens"] else "MISMATCH: vLLM date or version differs from the build")
