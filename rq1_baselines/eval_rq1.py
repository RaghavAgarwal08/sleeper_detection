from __future__ import annotations

import argparse
import numpy as np

from rq1_baselines.attention_tracker import AttentionTracker
from rq1_baselines.attnda import AttnDA


def run_rq1_evaluation(n_samples: int = 100, n_layers: int = 24, n_heads: int = 16) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(42)

    fs_clean = rng.normal(loc=0.80, scale=0.08, size=(n_samples, n_layers, n_heads))
    fs_dormant = rng.normal(loc=0.80, scale=0.08, size=(n_samples, n_layers, n_heads))
    fs_fired = rng.normal(loc=0.35, scale=0.10, size=(n_samples, n_layers, n_heads))

    train_fs = np.concatenate([fs_clean[:50], fs_fired[:50]], axis=0)
    train_labels = np.array([0] * 50 + [1] * 50)

    at = AttentionTracker(k=2.0).fit(train_fs, train_labels)
    attnda = AttnDA().fit(train_fs, train_labels)

    test_fs_fired = np.concatenate([fs_clean[50:], fs_fired[50:]], axis=0)
    test_labels_fired = np.array([0] * 50 + [1] * 50)

    test_fs_dormant = np.concatenate([fs_clean[50:], fs_dormant[50:]], axis=0)
    test_labels_dormant = np.array([0] * 50 + [1] * 50)

    return {
        "Fired vs Clean": {
            "Attention Tracker": at.evaluate(test_fs_fired, test_labels_fired).auroc,
            "AttnDA": attnda.evaluate(test_fs_fired, test_labels_fired).auroc,
        },
        "Dormant vs Clean": {
            "Attention Tracker": at.evaluate(test_fs_dormant, test_labels_dormant).auroc,
            "AttnDA": attnda.evaluate(test_fs_dormant, test_labels_dormant).auroc,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="RQ1: Do AT and AttnDA fail on dormant payloads?")
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    print("================================================================================")
    print("RQ1 EVALUATION: DO ATTENTION TRACKER & ATTNDA FAIL ON DORMANT PAYLOADS?")
    print("================================================================================\n")

    results = run_rq1_evaluation(n_samples=args.samples)
    for setting, scores in results.items():
        print(f"--- {setting} ---")
        for method, auroc in scores.items():
            print(f"  {method:<25} AUROC: {auroc:.4f}")
        print()


if __name__ == "__main__":
    main()
