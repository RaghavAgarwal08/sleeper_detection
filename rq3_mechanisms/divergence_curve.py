from __future__ import annotations

import numpy as np


def compute_divergence_curve(
    h_state_a: np.ndarray,
    h_state_b: np.ndarray,
) -> np.ndarray:
    diffs = h_state_a - h_state_b
    return np.mean(np.linalg.norm(diffs, axis=-1), axis=0)


def find_bifurcation_layer(
    divergence_curve: np.ndarray,
    threshold_ratio: float = 0.5,
) -> int:
    max_div = np.max(divergence_curve)
    threshold = threshold_ratio * max_div
    for l_idx, val in enumerate(divergence_curve):
        if val >= threshold:
            return l_idx
    return len(divergence_curve) - 1
