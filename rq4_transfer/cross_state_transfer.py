from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TransferResult:
    transfer_auroc: float
    oracle_auroc: float
    retention_ratio: float


def train_reached_test_unreached(
    h_clean_train: np.ndarray,
    h_reached_train: np.ndarray,
    h_clean_test: np.ndarray,
    h_unreached_test: np.ndarray,
    target_layer: int = 16,
) -> TransferResult:
    x_c_tr = h_clean_train[:, target_layer, :]
    x_r_tr = h_reached_train[:, target_layer, :]
    v_reached = np.mean(x_r_tr, axis=0) - np.mean(x_c_tr, axis=0)
    norm_r = np.linalg.norm(v_reached)
    if norm_r > 1e-12:
        v_reached = v_reached / norm_r

    x_c_te = h_clean_test[:, target_layer, :]
    x_u_te = h_unreached_test[:, target_layer, :]

    s_c = np.dot(x_c_te, v_reached)
    s_u = np.dot(x_u_te, v_reached)
    auroc_transfer = _roc(s_u, s_c)

    # Oracle trained directly on unreached states
    v_unreached = np.mean(x_u_te, axis=0) - np.mean(x_c_te, axis=0)
    norm_u = np.linalg.norm(v_unreached)
    if norm_u > 1e-12:
        v_unreached = v_unreached / norm_u
    s_c_oracle = np.dot(x_c_te, v_unreached)
    s_u_oracle = np.dot(x_u_te, v_unreached)
    auroc_oracle = _roc(s_u_oracle, s_c_oracle)

    retention = float(auroc_transfer / auroc_oracle) if auroc_oracle > 1e-12 else 0.0

    return TransferResult(
        transfer_auroc=auroc_transfer,
        oracle_auroc=auroc_oracle,
        retention_ratio=retention,
    )


def _roc(pos: np.ndarray, neg: np.ndarray) -> float:
    diffs = pos[:, None] - neg[None, :]
    return float(np.mean((diffs > 0).astype(float) + 0.5 * (diffs == 0).astype(float)))
