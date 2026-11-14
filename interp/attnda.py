from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AttnDAResult:
    weights: np.ndarray
    scores: np.ndarray
    auroc: float


class AttnDA:
    def __init__(self, epsilon: float = 1e-4):
        self.epsilon = float(epsilon)
        self.weights: np.ndarray | None = None
        self.threshold: float = 0.0

    def fit(self, focus_scores: np.ndarray, labels: np.ndarray) -> AttnDA:
        n_samples = focus_scores.shape[0]
        flat_fs = focus_scores.reshape(n_samples, -1)

        benign = flat_fs[labels == 0]
        malicious = flat_fs[labels == 1]

        u_b = np.mean(benign, axis=0)
        u_m = np.mean(malicious, axis=0)
        var_b = np.var(benign, axis=0)
        var_m = np.var(malicious, axis=0)

        scatter_diag = (1.0 - self.epsilon) * (var_b + var_m) + self.epsilon
        self.weights = (u_m - u_b) / scatter_diag
        self.threshold = float(0.5 * np.dot(self.weights, u_m + u_b))
        return self

    def score(self, focus_scores: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("AttnDA must be fit before calling score().")
        flat_fs = focus_scores.reshape(focus_scores.shape[0], -1)
        return np.dot(flat_fs, self.weights)

    def predict(self, focus_scores: np.ndarray) -> np.ndarray:
        scores = self.score(focus_scores)
        return (scores > self.threshold).astype(int)

    def evaluate(self, focus_scores: np.ndarray, labels: np.ndarray) -> AttnDAResult:
        scores = self.score(focus_scores)
        auroc = _compute_auroc(scores, labels)
        return AttnDAResult(
            weights=self.weights,
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
