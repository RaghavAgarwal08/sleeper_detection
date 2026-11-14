from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CausalInterventionResult:
    original_score: float
    intervened_score: float
    delta: float
    success: bool


def ablate_direction(h: np.ndarray, v_unit: np.ndarray) -> np.ndarray:
    proj = np.dot(h, v_unit)
    return h - proj[..., None] * v_unit


def steer_direction(h: np.ndarray, v_unit: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return h + alpha * v_unit


def evaluate_causal_effects(
    h_dormant: np.ndarray,
    h_fired: np.ndarray,
    v_unit: np.ndarray,
    alpha: float = 1.0,
) -> dict[str, CausalInterventionResult]:
    s_dorm_orig = float(np.mean(np.dot(h_dormant, v_unit)))
    h_dorm_steered = steer_direction(h_dormant, v_unit, alpha=alpha)
    s_dorm_steered = float(np.mean(np.dot(h_dorm_steered, v_unit)))

    s_fired_orig = float(np.mean(np.dot(h_fired, v_unit)))
    h_fired_ablated = ablate_direction(h_fired, v_unit)
    s_fired_ablated = float(np.mean(np.dot(h_fired_ablated, v_unit)))

    return {
        "steering_dormant": CausalInterventionResult(
            original_score=s_dorm_orig,
            intervened_score=s_dorm_steered,
            delta=s_dorm_steered - s_dorm_orig,
            success=(s_dorm_steered > s_dorm_orig),
        ),
        "ablation_fired": CausalInterventionResult(
            original_score=s_fired_orig,
            intervened_score=s_fired_ablated,
            delta=s_fired_ablated - s_fired_orig,
            success=(np.abs(s_fired_ablated) < 1e-4),
        ),
    }
