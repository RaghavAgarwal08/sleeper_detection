from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SubspaceResult:
    commitment_vector: np.ndarray
    surface_vector: np.ndarray
    divergence_curve: np.ndarray
    scores: np.ndarray
    auroc: float


class SubspaceDivergenceProbe:
    def __init__(self, target_layers: tuple[int, int] = (14, 20)):
        self.target_layers = target_layers
        self.commitment_vector: np.ndarray | None = None
        self.surface_vector: np.ndarray | None = None
        self.layer_weights: np.ndarray | None = None

    def fit(
        self,
        hidden_states_dormant: np.ndarray,
        hidden_states_twin: np.ndarray,
        hidden_states_clean: np.ndarray,
    ) -> SubspaceDivergenceProbe:
        n_layers = hidden_states_dormant.shape[1]
        l_start, l_end = self.target_layers

        delta_commitment = hidden_states_dormant[:, l_start:l_end, :] - hidden_states_twin[:, l_start:l_end, :]
        v_commit = np.mean(delta_commitment, axis=(0, 1))
        norm_c = np.linalg.norm(v_commit)
        if norm_c > 1e-12:
            v_commit = v_commit / norm_c

        delta_surface = hidden_states_twin[:, l_start:l_end, :] - hidden_states_clean[:, l_start:l_end, :]
        v_surface = np.mean(delta_surface, axis=(0, 1))
        norm_s = np.linalg.norm(v_surface)
        if norm_s > 1e-12:
            v_surface = v_surface / norm_s

        v_commit_orth = v_commit - np.dot(v_commit, v_surface) * v_surface
        norm_orth = np.linalg.norm(v_commit_orth)
        if norm_orth > 1e-12:
            v_commit_orth = v_commit_orth / norm_orth

        self.commitment_vector = v_commit_orth
        self.surface_vector = v_surface

        layer_div = np.mean(
            np.linalg.norm(hidden_states_dormant - hidden_states_twin, axis=-1),
            axis=0,
        )
        norm_div = np.sum(layer_div[l_start:l_end])
        weights = np.zeros(n_layers, dtype=np.float32)
        if norm_div > 1e-12:
            weights[l_start:l_end] = layer_div[l_start:l_end] / norm_div
        else:
            weights[l_start:l_end] = 1.0 / (l_end - l_start)
        self.layer_weights = weights

        return self

    def compute_divergence_curve(
        self,
        hidden_states_a: np.ndarray,
        hidden_states_b: np.ndarray,
    ) -> np.ndarray:
        diffs = hidden_states_a - hidden_states_b
        return np.mean(np.linalg.norm(diffs, axis=-1), axis=0)

    def score(self, hidden_states: np.ndarray) -> np.ndarray:
        if self.commitment_vector is None or self.layer_weights is None:
            raise RuntimeError("SubspaceDivergenceProbe must be fit before score().")

        projections = np.dot(hidden_states, self.commitment_vector)
        weighted_scores = np.sum(projections * self.layer_weights[None, :], axis=1)
        return weighted_scores

    def evaluate(self, hidden_states: np.ndarray, labels: np.ndarray) -> SubspaceResult:
        scores = self.score(hidden_states)
        auroc = _compute_auroc(scores, labels)
        dummy_curve = np.zeros(hidden_states.shape[1], dtype=np.float32)
        return SubspaceResult(
            commitment_vector=self.commitment_vector,
            surface_vector=self.surface_vector,
            divergence_curve=dummy_curve,
            scores=scores,
            auroc=auroc,
        )


def _compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diffs = pos[:, None] - neg[None, :]
    return float(np.mean((diffs > 0).astype(float) + 0.5 * (diffs == 0).astype(float)))
