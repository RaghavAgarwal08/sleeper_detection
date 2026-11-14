from __future__ import annotations

import argparse
import numpy as np

from rq2_probing.subspace_probe import SubspaceDivergenceProbe
from rq2_probing.layer_sweep import run_layer_sweep


def run_rq2_evaluation(n_samples: int = 100, n_layers: int = 24, d_model: int = 2880) -> dict[str, float]:
    rng = np.random.default_rng(42)

    v_surface = rng.normal(size=d_model)
    v_surface /= np.linalg.norm(v_surface)

    v_commit = rng.normal(size=d_model)
    v_commit -= np.dot(v_commit, v_surface) * v_surface
    v_commit /= np.linalg.norm(v_commit)

    h_clean = rng.normal(scale=1.0, size=(n_samples, n_layers, d_model))
    h_twin = h_clean + 1.2 * v_surface[None, None, :]
    h_unreached = h_twin.copy()
    h_reached = h_twin.copy()

    # Pre-execution commitment subspace in layers 14-19
    h_unreached[:, 14:20, :] += 2.5 * v_commit[None, None, :]
    h_reached[:, 14:20, :] += 2.8 * v_commit[None, None, :]
    # Late layers: execution commitment
    h_reached[:, 20:, :] += 3.5 * v_commit[None, None, :]

    sweep = run_layer_sweep(h_clean, h_unreached, h_reached)

    probe = SubspaceDivergenceProbe(target_layers=(14, 20)).fit(
        hidden_states_unreached=h_unreached[:50],
        hidden_states_benign_twin=h_twin[:50],
        hidden_states_clean=h_clean[:50],
    )

    test_h_unreached = np.concatenate([h_clean[50:], h_unreached[50:]], axis=0)
    test_h_reached = np.concatenate([h_clean[50:], h_reached[50:]], axis=0)
    test_labels = np.array([0] * 50 + [1] * 50)

    unreached_subspace_auroc = probe.evaluate(test_h_unreached, test_labels).auroc
    reached_subspace_auroc = probe.evaluate(test_h_reached, test_labels).auroc

    last_layer_unreached = sweep.unreached_aurocs[-1]
    last_layer_reached = sweep.reached_aurocs[-1]

    return {
        "Best Mid-Layer (L{}) AUROC".format(sweep.best_mid_layer): sweep.best_mid_auroc,
        "Last-Layer (L{}) Unreached AUROC".format(n_layers - 1): last_layer_unreached,
        "Last-Layer (L{}) Reached AUROC".format(n_layers - 1): last_layer_reached,
        "Subspace Probe Unreached AUROC": unreached_subspace_auroc,
        "Subspace Probe Reached AUROC": reached_subspace_auroc,
    }


def main():
    parser = argparse.ArgumentParser(description="RQ2: Does mid-layer probing beat last-layer and baselines?")
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    print("================================================================================")
    print("RQ2 EVALUATION: MID-LAYER PROBING (REACHED VS UNREACHED PRE-EXECUTION)")
    print("================================================================================\n")

    results = run_rq2_evaluation(n_samples=args.samples)
    for metric, val in results.items():
        print(f"  {metric:<35}: {val:.4f}")
    print()


if __name__ == "__main__":
    main()
