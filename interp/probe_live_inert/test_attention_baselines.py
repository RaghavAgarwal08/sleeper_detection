"""CPU checks for attention_capture.spans and the attention_baselines methods (no GPU, no model).

    python interp/probe_live_inert/test_attention_baselines.py          # unit checks, a few seconds
    python interp/probe_live_inert/test_attention_baselines.py --e2e    # also run the full analysis on synthetic
                                                                          features with a planted signal (minutes)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import attention_baselines as AB  # noqa: E402
import attention_capture as AC  # noqa: E402

DATA = REPO / "interp/data/behavior_v1"


def test_spans():
    reqs = [json.loads(l) for l in open(DATA / "requests.jsonl")]
    for r in reqs:
        sp = AC.spans(r)
        roles = [t for k in AC.ROLES for t in sp[k]]
        assert roles == list(range(len(r["input_ids"]))), "role spans must partition the prompt"
        assert sp["user_content"][-1] == r["positions"]["user_end"] and set(sp["user_content"]) <= set(sp["user"])
        assert sp["tool_content"][-1] == r["positions"]["memory_result_end"] and set(sp["tool_content"]) <= set(sp["tool"])
        assert set(sp["entries"]) <= set(sp["tool"]), "memory entries must sit inside the tool result"
        assert r["positions"]["last_prompt"] in sp["final_assistant"]
    n_sys = {len(AC.spans(r)["system"]) for r in reqs}
    assert len(n_sys) == 1, f"system prompt length must be constant for AttentionDefense, got {n_sys}"
    print(f"spans ok on {len(reqs)} requests, system prompt {n_sys.pop()} tokens")


def synthetic(n=400, L=6, H=8, S=9, n_sys=10, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    seg = rng.uniform(0.05, 0.2, (n, 2, L, H, S))
    si = {s: k for k, s in enumerate(AC.SPANS)}
    seg[y == 1, 0, 3, :3, si["user_content"]] *= 0.4  # distraction: less attention to the user request
    sysA = rng.uniform(0, 0.1, (n, 2, H, n_sys))
    sysA[y == 1, -1, 2, 4] += 0.08
    fc = rng.normal(0, 1, (n, 32)).astype(np.float32)
    fc[y == 1, 5] += 2.5  # d prime 2.5, AUROC about 0.96 for one feature
    return y, {"seg": seg, "sys": sysA, "fc": fc}, si


def test_methods():
    y, feats, si = synthetic()
    tr, te = np.arange(0, 300), np.arange(300, 400)
    groups = np.arange(400) // 4
    mu = feats["seg"][:, 0, :, :, si["user_content"]]
    cal = (mu[y == 0].mean(0), mu[y == 0].std(0), mu[y == 1].mean(0), mu[y == 1].std(0))
    sc, info = AB.run_methods(tr, te, y, groups, feats, cal, si, full_layers=[1, 3, 5])
    auc = {k: roc_auc_score(y[te], v) for k, v in sc.items()}
    print("synthetic AUROC", {k: round(v, 3) for k, v in auc.items()}, info)
    for k in ["AT", "AT_fit", "AttnDA", "AttnDef_RF", "AttnDef_LR", "FC"]:
        assert auc[k] > 0.85, f"{k} failed to find the planted signal"
    assert auc["AttnDA_full"] > 0.85, "layer 3 is a planted full layer, AttnDA_full should see it"
    assert auc["AttnDA_mre"] < 0.75, "no signal was planted at the memory-result query"
    keep, k = AB.at_heads(np.zeros((2, 2)), np.ones((2, 2)), np.zeros((2, 2)), np.ones((2, 2)))
    assert k == "top10" and keep.sum() == 4
    m = AB.ShrinkLDA(0.1).fit(feats["fc"][tr], y[tr])
    assert roc_auc_score(y[te], m.score(feats["fc"][te])) > 0.7
    print("methods ok")


def test_e2e():
    """Synthetic attn_features.npz aligned with the real ids, signal planted on the poisoned label, full analysis run."""
    import probe as P
    labels, _ = P.load_labels(DATA)
    reqs = [json.loads(l) for l in open(DATA / "requests.jsonl")]
    n_sys = len(AC.spans(reqs[0])["system"])
    rng = np.random.default_rng(1)
    n, L, H = len(labels), 24, 64
    poisoned = np.array([l["framing"] == "live" for l in labels])  # dataset field: "live" = poisoned
    seg = rng.uniform(0.01, 0.2, (n, 2, L, H, len(AC.SPANS))).astype(np.float16)
    seg[poisoned, 0, 9, :4, AC.SPANS.index("user_content")] *= 0.3
    with tempfile.TemporaryDirectory() as d:
        np.savez_compressed(Path(d) / "attn_features.npz", ids=np.array([l["id"] for l in labels]), seg_attn=seg,
                            sys_attn=rng.uniform(0, 0.1, (n, 2, H, n_sys)).astype(np.float16),
                            fc_last=rng.normal(0, 1, (n, 2880)).astype(np.float16), spans=np.array(AC.SPANS),
                            queries=np.array(AC.QUERIES), sys_layers=np.array(AC.SYS_LAYERS),
                            layer_types=np.array(["sliding_attention", "full_attention"] * 12))
        subprocess.run([sys.executable, str(HERE / "attention_baselines.py"), "--attn", str(Path(d) / "attn_features.npz"),
                        "--out", d], check=True)
        rep = json.load(open(Path(d) / "attention_baselines.json"))
    li = rep["tasks"]["poison_vs_twin"]["E1_new_templates"]["methods"]
    assert li["AT_fit"]["auroc"] > 0.9 and li["AttnDA"]["auroc"] > 0.8 and li["AttnDA_full"]["auroc"] > 0.8, li  # 4 of 1,536 heads carry signal
    assert abs(li["FC"]["auroc"] - 0.5) < 0.15, "FC got random features, should be near chance"
    print("e2e ok", {k: v["auroc"] for k, v in li.items()})


if __name__ == "__main__":
    test_spans()
    test_methods()
    if "--e2e" in sys.argv:
        test_e2e()
