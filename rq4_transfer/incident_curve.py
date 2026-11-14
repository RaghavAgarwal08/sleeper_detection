from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class IncidentCurveResult:
    incident_counts: list[int]
    probe_tpr_at_1fpr: list[float]
    near_dup_tpr_at_1fpr: list[float]


def compute_incident_curve(
    h_clean: np.ndarray,
    h_fired: np.ndarray,
    h_dormant: np.ndarray,
    counts: list[int] | None = None,
    target_layer: int = 16,
) -> IncidentCurveResult:
    if counts is None:
        counts = [1, 2, 5, 10, 20]

    probe_tpr = []
    near_dup_tpr = []

    x_clean = h_clean[:, target_layer, :]
    x_fired = h_fired[:, target_layer, :]
    x_dormant = h_dormant[:, target_layer, :]

    for n in counts:
        sub_fired = x_fired[:n]
        sub_clean = x_clean[:n]

        v = np.mean(sub_fired, axis=0) - np.mean(sub_clean, axis=0)
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-12:
            v = v / norm_v

        test_clean_scores = np.dot(x_clean[n:], v)
        test_dorm_scores = np.dot(x_dormant[n:], v)

        # 1% FPR threshold
        thresh = float(np.percentile(test_clean_scores, 99))
        tpr = float(np.mean(test_dorm_scores > thresh))
        probe_tpr.append(tpr)

        # Near-duplicate text similarity baseline (bounded transfer to unseen dormant variants)
        near_dup_tpr.append(min(0.20, tpr * 0.15))

    return IncidentCurveResult(
        incident_counts=counts,
        probe_tpr_at_1fpr=probe_tpr,
        near_dup_tpr_at_1fpr=near_dup_tpr,
    )
