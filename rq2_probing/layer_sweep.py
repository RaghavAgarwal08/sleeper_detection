from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class LayerSweepResult:
    layers: list[int]
    reached_aurocs: list[float]
    unreached_aurocs: list[float]
    best_mid_layer: int
    best_mid_auroc: float


def run_layer_sweep(
    h_clean: np.ndarray,
    h_unreached: np.ndarray,
    h_reached: np.ndarray,
) -> LayerSweepResult:
    n_samples, n_layers, d_model = h_clean.shape
    split = n_samples // 2

    reached_aurocs = []
    unreached_aurocs = []

    for l in range(n_layers):
        x_clean_tr = h_clean[:split, l, :]
        x_reached_tr = h_reached[:split, l, :]
        v_l = np.mean(x_reached_tr, axis=0) - np.mean(x_clean_tr, axis=0)
        norm_v = np.linalg.norm(v_l)
        if norm_v > 1e-12:
            v_l = v_l / norm_v

        # Eval on held-out reached execution vs clean
        s_clean_te = np.dot(h_clean[split:, l, :], v_l)
        s_reached_te = np.dot(h_reached[split:, l, :], v_l)
        auroc_r = _roc(s_reached_te, s_clean_te)
        reached_aurocs.append(auroc_r)

        # Eval on held-out unreached vs clean
        s_unreached_te = np.dot(h_unreached[split:, l, :], v_l)
        auroc_u = _roc(s_unreached_te, s_clean_te)
        unreached_aurocs.append(auroc_u)

    mid_start = int(0.55 * n_layers)
    mid_end = int(0.80 * n_layers)
    mid_slice = unreached_aurocs[mid_start:mid_end]
    best_offset = int(np.argmax(mid_slice))
    best_layer = mid_start + best_offset
    best_auroc = mid_slice[best_offset]

    return LayerSweepResult(
        layers=list(range(n_layers)),
        reached_aurocs=reached_aurocs,
        unreached_aurocs=unreached_aurocs,
        best_mid_layer=best_layer,
        best_mid_auroc=best_auroc,
    )


def _roc(pos: np.ndarray, neg: np.ndarray) -> float:
    diffs = pos[:, None] - neg[None, :]
    return float(np.mean((diffs > 0).astype(float) + 0.5 * (diffs == 0).astype(float)))
