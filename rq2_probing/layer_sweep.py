from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class LayerSweepResult:
    layers: list[int]
    fired_aurocs: list[float]
    dormant_aurocs: list[float]
    best_mid_layer: int
    best_mid_auroc: float


def run_layer_sweep(
    h_clean: np.ndarray,
    h_dormant: np.ndarray,
    h_fired: np.ndarray,
) -> LayerSweepResult:
    n_samples, n_layers, d_model = h_clean.shape
    split = n_samples // 2

    fired_aurocs = []
    dormant_aurocs = []

    for l in range(n_layers):
        x_clean_tr = h_clean[:split, l, :]
        x_fired_tr = h_fired[:split, l, :]
        v_l = np.mean(x_fired_tr, axis=0) - np.mean(x_clean_tr, axis=0)
        norm_v = np.linalg.norm(v_l)
        if norm_v > 1e-12:
            v_l = v_l / norm_v

        # Eval on held-out fired vs clean
        s_clean_te = np.dot(h_clean[split:, l, :], v_l)
        s_fired_te = np.dot(h_fired[split:, l, :], v_l)
        auroc_f = _roc(s_fired_te, s_clean_te)
        fired_aurocs.append(auroc_f)

        # Eval on held-out dormant vs clean
        s_dormant_te = np.dot(h_dormant[split:, l, :], v_l)
        auroc_d = _roc(s_dormant_te, s_clean_te)
        dormant_aurocs.append(auroc_d)

    # Mid layers defined as 55% - 80% of depth
    mid_start = int(0.55 * n_layers)
    mid_end = int(0.80 * n_layers)
    mid_slice = dormant_aurocs[mid_start:mid_end]
    best_offset = int(np.argmax(mid_slice))
    best_layer = mid_start + best_offset
    best_auroc = mid_slice[best_offset]

    return LayerSweepResult(
        layers=list(range(n_layers)),
        fired_aurocs=fired_aurocs,
        dormant_aurocs=dormant_aurocs,
        best_mid_layer=best_layer,
        best_mid_auroc=best_auroc,
    )


def _roc(pos: np.ndarray, neg: np.ndarray) -> float:
    diffs = pos[:, None] - neg[None, :]
    return float(np.mean((diffs > 0).astype(float) + 0.5 * (diffs == 0).astype(float)))
