from __future__ import annotations

import numpy as np


def evaluate_lodo_transfer(
    family_data: dict[str, dict[str, np.ndarray]],
    target_layer: int = 16,
) -> dict[str, float]:
    family_aurocs = {}
    families = list(family_data.keys())

    for held_out in families:
        train_reached = []
        train_clean = []
        for fam in families:
            if fam == held_out:
                continue
            train_reached.append(family_data[fam]["reached"][:, target_layer, :])
            train_clean.append(family_data[fam]["clean"][:, target_layer, :])

        x_r = np.concatenate(train_reached, axis=0)
        x_c = np.concatenate(train_clean, axis=0)
        v = np.mean(x_r, axis=0) - np.mean(x_c, axis=0)
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-12:
            v = v / norm_v

        test_clean = family_data[held_out]["clean"][:, target_layer, :]
        test_unreached = family_data[held_out]["unreached"][:, target_layer, :]

        s_c = np.dot(test_clean, v)
        s_u = np.dot(test_unreached, v)

        diffs = s_u[:, None] - s_c[None, :]
        auroc = float(np.mean((diffs > 0).astype(float) + 0.5 * (diffs == 0).astype(float)))
        family_aurocs[held_out] = auroc

    family_aurocs["mean_lodo_auroc"] = float(np.mean(list(family_aurocs.values())))
    return family_aurocs
