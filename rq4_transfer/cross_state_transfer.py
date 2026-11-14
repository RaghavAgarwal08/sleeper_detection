from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TransferResult:
    transfer_auroc: float
    oracle_auroc: float
    retention_ratio: float


def train_fired_test_dormant(
    h_clean_train: np.ndarray,
    h_fired_train: np.ndarray,
    h_clean_test: np.ndarray,
    h_dormant_test: np.ndarray,
    target_layer: int = 16,
) -> TransferResult:
    x_c_tr = h_clean_train[:, target_layer, :]
    x_f_tr = h_fired_train[:, target_layer, :]
    v_fired = np.mean(x_f_tr, axis=0) - np.mean(x_c_tr, axis=0)
    norm_f = np.linalg.norm(v_fired)
    if norm_f > 1e-12:
        v_fired = v_fired / norm_f

    x_c_te = h_clean_test[:, target_layer, :]
    x_d_te = h_dormant_test[:, target_layer, :]

    s_c = np.dot(x_c_te, v_fired)
    s_d = np.dot(x_d_te, v_fired)
    auroc_transfer = _roc(s_d, s_c)

    # Oracle trained on dormant
    v_dormant = np.mean(x_d_te, axis=0) - np.mean(x_c_te, axis=0)
    norm_d = np.linalg.norm(v_dormant)
    if norm_d > 1e-12:
        v_dormant = v_dormant / norm_d
    s_c_oracle = np.dot(x_c_te, v_dormant)
    s_d_oracle = np.dot(x_d_te, v_dormant)
    auroc_oracle = _roc(s_d_oracle, s_c_oracle)

    retention = float(auroc_transfer / auroc_oracle) if auroc_oracle > 1e-12 else 0.0

    return TransferResult(
        transfer_auroc=auroc_transfer,
        oracle_auroc=auroc_oracle,
        retention_ratio=retention,
    )


def _roc(pos: np.ndarray, neg: np.ndarray) -> float:
    diffs = pos[:, None] - neg[None, :]
    return float(np.mean((diffs > 0).astype(float) + 0.5 * (diffs == 0).astype(float)))
