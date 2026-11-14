from __future__ import annotations

import numpy as np


def evaluate_lodo_transfer(
    family_data: dict[str, dict[str, np.ndarray]],
    target_layer: int = 16,
) -> dict[str, float]:
    family_aurocs = {}
    families = list(family_data.keys())

    for held_out in families:
        train_fired = []
        train_clean = []
        for fam in families:
            if fam == held_out:
                continue
            train_fired.append(family_data[fam]["fired"][:, target_layer, :])
            train_clean.append(family_data[fam]["clean"][:, target_layer, :])

        x_f = np.concatenate(train_fired, axis=0)
        x_c = np.concatenate(train_clean, axis=0)
        v = np.mean(x_f, axis=0) - np.mean(x_c, axis=0)
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-12:
            v = v / norm_v

        test_clean = family_data[held_out]["clean"][:, target_layer, :]
        test_dormant = family_data[held_out]["dormant"][:, target_layer, :]

        s_c = np.dot(test_clean, v)
        s_d = np.dot(test_dormant, v)

        diffs = s_d[:, None] - s_c[None, :]
        auroc = float(np.mean((diffs > 0).astype(float) + 0.5 * (diffs == 0).astype(float)))
        family_aurocs[held_out] = auroc

    family_aurocs["mean_lodo_auroc"] = float(np.mean(list(family_aurocs.values())))
    return family_aurocs
