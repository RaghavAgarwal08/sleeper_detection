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
    h_unreached: np.ndarray,
    h_reached: np.ndarray,
    v_unit: np.ndarray,
    alpha: float = 1.0,
) -> dict[str, CausalInterventionResult]:
    s_unreached_orig = float(np.mean(np.dot(h_unreached, v_unit)))
    h_unreached_steered = steer_direction(h_unreached, v_unit, alpha=alpha)
    s_unreached_steered = float(np.mean(np.dot(h_unreached_steered, v_unit)))

    s_reached_orig = float(np.mean(np.dot(h_reached, v_unit)))
    h_reached_ablated = ablate_direction(h_reached, v_unit)
    s_reached_ablated = float(np.mean(np.dot(h_reached_ablated, v_unit)))

    return {
        "steering_unreached": CausalInterventionResult(
            original_score=s_unreached_orig,
            intervened_score=s_unreached_steered,
            delta=s_unreached_steered - s_unreached_orig,
            success=bool(s_unreached_steered > s_unreached_orig),
        ),
        "ablation_reached": CausalInterventionResult(
            original_score=s_reached_orig,
            intervened_score=s_reached_ablated,
            delta=s_reached_ablated - s_reached_orig,
            success=bool(np.abs(s_reached_ablated) < 1e-4),
        ),
    }
