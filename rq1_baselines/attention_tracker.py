from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AttentionTrackerResult:
    selected_heads: list[tuple[int, int]]
    scores: np.ndarray
    auroc: float


class AttentionTracker:
    def __init__(self, k: float = 4.0):
        self.k = float(k)
        self.selected_heads: list[tuple[int, int]] = []
        self.num_layers = 0
        self.num_heads = 0

    @staticmethod
    def compute_focus_scores(
        attn_weights: list[np.ndarray],
        instruction_span: tuple[int, int],
        data_span: tuple[int, int],
        readout_pos: int = -1,
    ) -> np.ndarray:
        num_layers = len(attn_weights)
        num_heads = attn_weights[0].shape[0]
        focus = np.zeros((num_layers, num_heads), dtype=np.float32)

        i_start, i_end = instruction_span
        d_start, d_end = data_span

        for l_idx, layer_attn in enumerate(attn_weights):
            for h_idx in range(num_heads):
                head_w = layer_attn[h_idx, readout_pos, :]
                a_i = float(np.sum(head_w[i_start:i_end]))
                a_d = float(np.sum(head_w[d_start:d_end]))
                denom = a_i + a_d
                focus[l_idx, h_idx] = (a_i / denom) if denom > 1e-12 else 0.5
        return focus

    def fit(self, focus_scores: np.ndarray, labels: np.ndarray) -> AttentionTracker:
        benign_mask = labels == 0
        malicious_mask = labels == 1

        benign_data = focus_scores[benign_mask]
        malicious_data = focus_scores[malicious_mask]

        mu_b = np.mean(benign_data, axis=0)
        var_b = np.var(benign_data, axis=0)
        mu_m = np.mean(malicious_data, axis=0)
        var_m = np.var(malicious_data, axis=0)

        num_layers, num_heads = mu_b.shape
        self.num_layers = num_layers
        self.num_heads = num_heads

        denom = np.sqrt(var_b) + np.sqrt(var_m)
        denom = np.where(denom < 1e-8, 1e-8, denom)
        stat = (mu_b - mu_m) / denom

        self.selected_heads = [
            (l, h)
            for l in range(num_layers)
            for h in range(num_heads)
            if stat[l, h] > self.k
        ]
        return self

    def score(self, focus_scores: np.ndarray) -> np.ndarray:
        if not self.selected_heads:
            return np.zeros(focus_scores.shape[0], dtype=np.float32)

        indices = tuple(zip(*self.selected_heads))
        selected_vals = focus_scores[:, indices[0], indices[1]]
        aggregate_fs = np.mean(selected_vals, axis=-1)
        return -aggregate_fs

    def evaluate(self, focus_scores: np.ndarray, labels: np.ndarray) -> AttentionTrackerResult:
        scores = self.score(focus_scores)
        auroc = _compute_auroc(scores, labels)
        return AttentionTrackerResult(
            selected_heads=self.selected_heads,
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
