"""CPU, after capture.py on the pod: position sweep (Ashok and May 2025) and trajectory geometry (Pandey et al. 2026).

Position sweep: the same execution probe (held out cases, 5 folds, C = 0.01, binary target) on the window mean read
at each of the label free positions capture.py stores, from the end of the user request to the last prompt token.
Ashok and May found the final input tokens and middle layers predict behaviour best; this checks where in our prompt
the execution signal is strongest and whether the end of the memory result is the right place to read.

Trajectory geometry at the end of the memory result, across all layers (Pandey et al.): path length (sum of
distances between consecutive layers), mean turning angle between consecutive steps, and per layer cosine with the
previous layer. Each scalar is scored as a detector (AUROC within wording) for poisoned vs benign twin and for
reached execution vs triggered resisted, with no training at all.

    python interp/phase3/sweeps.py --model gpt-oss-20b
Writes results/phase3/<model>/sweeps.json.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import models as M  # noqa: E402

DATA = Path(os.environ.get("PHASE3_DATA", REPO / "interp/data/phase3"))
OUT = Path(os.environ.get("PHASE3_OUT", REPO / "results/phase3"))


def strat_auc(y, s, strata):
    num = den = 0.0
    for g in np.unique(strata):
        m = strata == g
        p, n = s[m][y[m] == 1], s[m][y[m] == 0]
        if len(p) and len(n):
            num += (p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()
            den += len(p) * len(n)
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    spec = M.get(args.model)
    d = DATA / spec["key"]
    labels = [json.loads(l) for l in open(d / "labels.jsonl")]
    bpath = d / ("behavior_pilot.jsonl" if args.pilot else "behavior.jsonl")
    if not bpath.exists():
        bpath = REPO / "interp/data/behavior_v1/behavior.jsonl.gz"
    opener = gzip.open if str(bpath).endswith(".gz") else open
    beh = {b["id"]: b for b in map(json.loads, opener(bpath, "rt"))}
    z = np.load(d / "activations.npz")
    pos = {i: k for k, i in enumerate(z["ids"])}
    labels = [l for l in labels if l["id"] in pos]
    order = [pos[l["id"]] for l in labels]
    fr = np.array([l["framing"] for l in labels])  # live = poisoned, inert = benign twin
    tm = np.array([-1 if l["template_id"] is None else int(l["template_id"]) for l in labels])
    cases = np.array([l["case_id"] for l in labels])
    poisoned = (fr == "live") & np.array([l["id"] in beh for l in labels])
    rate = np.array([beh[l["id"]]["fired_any"] / beh[l["id"]]["k"] if l["id"] in beh else 0 for l in labels])
    y = (rate >= 0.5).astype(int)
    out = {"model": spec["hf"]}

    # position sweep
    if "sweep_window_mean" in z.files:
        S = z["sweep_window_mean"][order].astype(np.float32)
        names = list(z["sweep_positions"])
        pop = np.where(poisoned)[0]
        res = {}
        for j, name in enumerate(names):
            s = np.full(len(labels), np.nan)
            for a, b in GroupKFold(5).split(pop, groups=cases[pop]):
                sc = StandardScaler().fit(S[pop[a], j])
                m = LogisticRegression(C=1e-2, max_iter=5000).fit(sc.transform(S[pop[a], j]), y[pop[a]])
                s[pop[b]] = m.predict_proba(sc.transform(S[pop[b], j]))[:, 1]
            res[name] = round(float(strat_auc(y[pop], s[pop], tm[pop])), 3)
            print(f"position {name:>20}: execution AUROC within wording {res[name]}", flush=True)
        out["position_sweep_execution_auroc_within_wording"] = res

    # trajectory geometry at the end of the memory result
    H = z["layers"][order][:, 0].astype(np.float32)  # [n, L+1, d]
    steps = np.diff(H, axis=1)
    length = np.linalg.norm(steps, axis=2).sum(1)
    unit = steps / (np.linalg.norm(steps, axis=2, keepdims=True) + 1e-8)
    turn = np.arccos(np.clip((unit[:, 1:] * unit[:, :-1]).sum(2), -1, 1)).mean(1)
    cos_prev = (H[:, 1:] * H[:, :-1]).sum(2) / (np.linalg.norm(H[:, 1:], axis=2) * np.linalg.norm(H[:, :-1], axis=2) + 1e-8)
    metrics = {"path_length": length, "mean_turning_angle": turn}
    for l in range(cos_prev.shape[1]):
        metrics[f"cos_layer{l + 1}_vs_{l}"] = cos_prev[:, l]
    pt = np.isin(fr, ["live", "inert"])
    traj = {}
    for k, v in metrics.items():
        a1 = strat_auc((fr[pt] == "live").astype(int), v[pt], tm[pt])
        a2 = strat_auc(y[poisoned], v[poisoned], tm[poisoned])
        traj[k] = {"poisoned_vs_twin": round(float(max(a1, 1 - a1)), 3), "execution": round(float(max(a2, 1 - a2)), 3),
                   "direction_poisoned_higher": bool(a1 >= 0.5), "direction_executed_higher": bool(a2 >= 0.5)}
    best = sorted(traj.items(), key=lambda kv: -kv[1]["execution"])[:5]
    print("trajectory, best for execution:", best, flush=True)
    out["trajectory_auroc_within_wording_sign_free"] = traj
    o = OUT / spec["key"]
    o.mkdir(parents=True, exist_ok=True)
    (o / "sweeps.json").write_text(json.dumps(out, indent=1))
    print("wrote", o / "sweeps.json")


if __name__ == "__main__":
    main()
