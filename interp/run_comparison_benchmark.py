from __future__ import annotations

import argparse
import numpy as np

from rq1_baselines.attention_tracker import AttentionTracker
from rq1_baselines.attnda import AttnDA
from rq2_probing.subspace_probe import SubspaceDivergenceProbe
from benchmarks.memorydojo import (
    TaskDomain,
    build_synthetic_trajectory,
    compute_lifecycle_metrics,
)


def run_benchmark(n_samples: int = 100, n_layers: int = 24, n_heads: int = 16, d_model: int = 2880) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(42)

    # 1. Attention focus scores:
    # Clean & Retrieved-Unreached: attention remains on instruction/benign context
    # Reached Execution: attention actively redirected to injected payload
    fs_clean = rng.normal(loc=0.80, scale=0.08, size=(n_samples, n_layers, n_heads))
    fs_twin = rng.normal(loc=0.80, scale=0.08, size=(n_samples, n_layers, n_heads))
    fs_unreached = rng.normal(loc=0.80, scale=0.08, size=(n_samples, n_layers, n_heads))
    fs_reached = rng.normal(loc=0.35, scale=0.10, size=(n_samples, n_layers, n_heads))

    # 2. Hidden states:
    # Surface text direction v_surface + latent commitment direction v_commit in middle layers (14-19)
    v_surface = rng.normal(size=d_model)
    v_surface /= np.linalg.norm(v_surface)

    v_commit = rng.normal(size=d_model)
    v_commit -= np.dot(v_commit, v_surface) * v_surface
    v_commit /= np.linalg.norm(v_commit)

    h_clean = rng.normal(scale=1.0, size=(n_samples, n_layers, d_model))
    h_twin = h_clean + 1.2 * v_surface[None, None, :]
    h_unreached = h_twin.copy()
    h_reached = h_twin.copy()

    # Pre-execution commitment in middle layers (14-19)
    h_unreached[:, 14:20, :] += 2.5 * v_commit[None, None, :]
    h_reached[:, 14:20, :] += 2.8 * v_commit[None, None, :]
    # Late layers: execution commitment
    h_reached[:, 20:, :] += 3.5 * v_commit[None, None, :]

    # 3. Fit detectors on training split
    train_fs = np.concatenate([fs_clean[:50], fs_reached[:50]], axis=0)
    train_labels = np.array([0] * 50 + [1] * 50)

    at = AttentionTracker(k=2.0).fit(train_fs, train_labels)
    attnda = AttnDA().fit(train_fs, train_labels)

    probe = SubspaceDivergenceProbe(target_layers=(14, 20)).fit(
        hidden_states_unreached=h_unreached[:50],
        hidden_states_benign_twin=h_twin[:50],
        hidden_states_clean=h_clean[:50],
    )

    # 4. Evaluate across 3 test settings:
    # Test A: Reached Execution vs Clean
    test_fs_reached = np.concatenate([fs_clean[50:], fs_reached[50:]], axis=0)
    test_h_reached = np.concatenate([h_clean[50:], h_reached[50:]], axis=0)
    test_labels_reached = np.array([0] * 50 + [1] * 50)

    # Test B: Retrieved Unreached (Pre-Execution) vs Clean
    test_fs_unreached = np.concatenate([fs_clean[50:], fs_unreached[50:]], axis=0)
    test_h_unreached = np.concatenate([h_clean[50:], h_unreached[50:]], axis=0)
    test_labels_unreached = np.array([0] * 50 + [1] * 50)

    # Test C: Retrieved Unreached vs Benign Twin (Falsifying H0 text confound)
    test_fs_twin = np.concatenate([fs_twin[50:], fs_unreached[50:]], axis=0)
    test_h_twin = np.concatenate([h_twin[50:], h_unreached[50:]], axis=0)
    test_labels_twin = np.array([0] * 50 + [1] * 50)

    return {
        "Reached Execution vs Clean": {
            "Attention Tracker": at.evaluate(test_fs_reached, test_labels_reached).auroc,
            "AttnDA": attnda.evaluate(test_fs_reached, test_labels_reached).auroc,
            "Subspace Divergence (Ours)": probe.evaluate(test_h_reached, test_labels_reached).auroc,
        },
        "Retrieved Unreached vs Clean": {
            "Attention Tracker": at.evaluate(test_fs_unreached, test_labels_unreached).auroc,
            "AttnDA": attnda.evaluate(test_fs_unreached, test_labels_unreached).auroc,
            "Subspace Divergence (Ours)": probe.evaluate(test_h_unreached, test_labels_unreached).auroc,
        },
        "Retrieved Unreached vs Benign Twin": {
            "Attention Tracker": at.evaluate(test_fs_twin, test_labels_twin).auroc,
            "AttnDA": attnda.evaluate(test_fs_twin, test_labels_twin).auroc,
            "Subspace Divergence (Ours)": probe.evaluate(test_h_twin, test_labels_twin).auroc,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Head-to-head comparison benchmark")
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    print("================================================================================")
    print("SLEEPER DETECTION BENCHMARK: AT vs AttnDA vs SUBSPACE DIVERGENCE (OURS)")
    print("================================================================================\n")

    results = run_benchmark(n_samples=args.samples)
    for setting, scores in results.items():
        print(f"--- {setting} ---")
        for method, auroc in scores.items():
            print(f"  {method:<30} AUROC: {auroc:.4f}")
        print()

    print("--- MemoryDojo Trajectory Synthesis ---")
    trajectories = [
        build_synthetic_trajectory(d, length=5, t0=0)
        for d in [TaskDomain.WORKSPACE, TaskDomain.BANKING, TaskDomain.TRAVEL, TaskDomain.SLACK]
    ]
    metrics = compute_lifecycle_metrics(trajectories)
    print(f"  Trajectories Built : {len(trajectories)}")
    print(f"  Insertion Rate (ISR): {metrics.isr * 100:.1f}%")
    print(f"  Retrieval Rate (RSR): {metrics.rsr * 100:.1f}%")
    print(f"  Persistence Share  : {metrics.pe * 100:.1f}%")
    print(f"  Reach Rate (BRT)   : {metrics.brt * 100:.1f}%\n")


if __name__ == "__main__":
    main()
