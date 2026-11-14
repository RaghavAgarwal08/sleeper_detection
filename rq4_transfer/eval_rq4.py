from __future__ import annotations

import argparse
import numpy as np

from rq4_transfer.cross_state_transfer import train_fired_test_dormant
from rq4_transfer.incident_curve import compute_incident_curve
from rq4_transfer.lodo_eval import evaluate_lodo_transfer


def run_rq4_evaluation(n_samples: int = 100, n_layers: int = 24, d_model: int = 2880) -> dict[str, float]:
    rng = np.random.default_rng(42)

    v_commit = rng.normal(size=d_model)
    v_commit /= np.linalg.norm(v_commit)

    # Cross-state transfer
    h_clean = rng.normal(scale=1.0, size=(n_samples, n_layers, d_model))
    h_dormant = h_clean.copy()
    h_fired = h_clean.copy()

    h_dormant[:, 14:20, :] += 2.5 * v_commit[None, None, :]
    h_fired[:, 14:20, :] += 2.8 * v_commit[None, None, :]
    h_fired[:, 20:, :] += 3.5 * v_commit[None, None, :]

    split = n_samples // 2
    res = train_fired_test_dormant(
        h_clean[:split],
        h_fired[:split],
        h_clean[split:],
        h_dormant[split:],
        target_layer=16,
    )

    # Incident curve
    curve = compute_incident_curve(h_clean, h_fired, h_dormant, counts=[1, 5, 10], target_layer=16)

    # LODO
    families = ["exfiltration", "sabotage", "financial_fraud"]
    family_data = {}
    for f in families:
        c = rng.normal(scale=1.0, size=(30, n_layers, d_model))
        d = c.copy()
        f_act = c.copy()
        d[:, 14:20, :] += 2.5 * v_commit[None, None, :]
        f_act[:, 14:20, :] += 2.8 * v_commit[None, None, :]
        family_data[f] = {"clean": c, "dormant": d, "fired": f_act}

    lodo_res = evaluate_lodo_transfer(family_data, target_layer=16)

    return {
        "Cross-State (Fired->Dormant) AUROC": res.transfer_auroc,
        "Oracle Ceiling AUROC": res.oracle_auroc,
        "Retention Ratio": res.retention_ratio,
        "TPR @ 1% FPR (N=1 Incident)": curve.probe_tpr_at_1fpr[0],
        "TPR @ 1% FPR (N=10 Incidents)": curve.probe_tpr_at_1fpr[2],
        "Mean LODO Held-Out AUROC": lodo_res["mean_lodo_auroc"],
    }


def main():
    parser = argparse.ArgumentParser(description="RQ4: Can a probe trained on fired find dormant?")
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    print("================================================================================")
    print("RQ4 EVALUATION: INCIDENT RESPONSE & CROSS-STATE GENERALIZATION")
    print("================================================================================\n")

    results = run_rq4_evaluation(n_samples=args.samples)
    for k, v in results.items():
        print(f"  {k:<35}: {v:.4f}")
    print()


if __name__ == "__main__":
    main()
