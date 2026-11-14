from __future__ import annotations

import argparse
import numpy as np

from interp.attention_tracker import AttentionTracker
from interp.attnda import AttnDA
from interp.subspace_probe import SubspaceDivergenceProbe
from interp.memorydojo import (
    TaskDomain,
    build_synthetic_trajectory,
    compute_lifecycle_metrics,
)


def run_benchmark(n_samples: int = 100, n_layers: int = 24, n_heads: int = 16, d_model: int = 2880) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(42)

    # 1. Simulate attention focus scores:
    # Fired: attention redirected to payload
    # Dormant: attention remains on instruction/benign context (no trigger -> no attention shift)
    fs_clean = rng.normal(loc=0.80, scale=0.08, size=(n_samples, n_layers, n_heads))
    fs_twin = rng.normal(loc=0.80, scale=0.08, size=(n_samples, n_layers, n_heads))
    fs_dormant = rng.normal(loc=0.80, scale=0.08, size=(n_samples, n_layers, n_heads))
    fs_fired = rng.normal(loc=0.35, scale=0.10, size=(n_samples, n_layers, n_heads))

    # 2. Simulate hidden states:
    # Surface direction v_surface + latent commitment direction v_commit in middle layers (14-19)
    v_surface = rng.normal(size=d_model)
    v_surface /= np.linalg.norm(v_surface)

    v_commit = rng.normal(size=d_model)
    v_commit -= np.dot(v_commit, v_surface) * v_surface
    v_commit /= np.linalg.norm(v_commit)

    h_clean = rng.normal(scale=1.0, size=(n_samples, n_layers, d_model))
    h_twin = h_clean + 1.2 * v_surface[None, None, :]
    h_dormant = h_twin.copy()
    h_fired = h_twin.copy()

    # In middle layers (14-19), both dormant and fired enter the commitment state
    h_dormant[:, 14:20, :] += 2.5 * v_commit[None, None, :]
    h_fired[:, 14:20, :] += 2.8 * v_commit[None, None, :]
    # Late layers: fired continues executing, dormant decays
    h_fired[:, 20:, :] += 3.5 * v_commit[None, None, :]

    # 3. Fit detectors on training data (Clean vs Fired / Twins)
    train_fs = np.concatenate([fs_clean[:50], fs_fired[:50]], axis=0)
    train_labels = np.array([0] * 50 + [1] * 50)

    at = AttentionTracker(k=2.0).fit(train_fs, train_labels)
    attnda = AttnDA().fit(train_fs, train_labels)

    probe = SubspaceDivergenceProbe(target_layers=(14, 20)).fit(
        hidden_states_dormant=h_dormant[:50],
        hidden_states_twin=h_twin[:50],
        hidden_states_clean=h_clean[:50],
    )

    # 4. Evaluate on Test sets:
    # Test A: Fired vs Clean
    test_fs_fired = np.concatenate([fs_clean[50:], fs_fired[50:]], axis=0)
    test_h_fired = np.concatenate([h_clean[50:], h_fired[50:]], axis=0)
    test_labels_fired = np.array([0] * 50 + [1] * 50)

    # Test B: Dormant vs Clean (The key question)
    test_fs_dormant = np.concatenate([fs_clean[50:], fs_dormant[50:]], axis=0)
    test_h_dormant = np.concatenate([h_clean[50:], h_dormant[50:]], axis=0)
    test_labels_dormant = np.array([0] * 50 + [1] * 50)

    # Test C: Dormant vs Benign Twin (Ruling out surface confound H0)
    test_fs_twin = np.concatenate([fs_twin[50:], fs_dormant[50:]], axis=0)
    test_h_twin = np.concatenate([h_twin[50:], h_dormant[50:]], axis=0)
    test_labels_twin = np.array([0] * 50 + [1] * 50)

    results = {
        "Fired vs Clean": {
            "Attention Tracker": at.evaluate(test_fs_fired, test_labels_fired).auroc,
            "AttnDA": attnda.evaluate(test_fs_fired, test_labels_fired).auroc,
            "Subspace Divergence (Ours)": probe.evaluate(test_h_fired, test_labels_fired).auroc,
        },
        "Dormant vs Clean": {
            "Attention Tracker": at.evaluate(test_fs_dormant, test_labels_dormant).auroc,
            "AttnDA": attnda.evaluate(test_fs_dormant, test_labels_dormant).auroc,
            "Subspace Divergence (Ours)": probe.evaluate(test_h_dormant, test_labels_dormant).auroc,
        },
        "Dormant vs Benign Twin": {
            "Attention Tracker": at.evaluate(test_fs_twin, test_labels_twin).auroc,
            "AttnDA": attnda.evaluate(test_fs_twin, test_labels_twin).auroc,
            "Subspace Divergence (Ours)": probe.evaluate(test_h_twin, test_labels_twin).auroc,
        },
    }

    return results


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
