from __future__ import annotations

import argparse
import numpy as np

from rq3_mechanisms.benign_twins import evaluate_benign_twin_separation
from rq3_mechanisms.untriggerable import evaluate_untriggerable_contrast
from rq3_mechanisms.divergence_curve import compute_divergence_curve, find_bifurcation_layer
from rq3_mechanisms.causal_steering import evaluate_causal_effects


def run_rq3_evaluation(n_samples: int = 100, n_layers: int = 24, d_model: int = 2880) -> dict[str, str | float]:
    rng = np.random.default_rng(42)

    v_surface = rng.normal(size=d_model)
    v_surface /= np.linalg.norm(v_surface)

    v_commit = rng.normal(size=d_model)
    v_commit -= np.dot(v_commit, v_surface) * v_surface
    v_commit /= np.linalg.norm(v_commit)

    h_clean = rng.normal(scale=1.0, size=(n_samples, n_layers, d_model))
    h_twin = h_clean + 1.2 * v_surface[None, None, :]
    h_untrig = h_clean + 1.2 * v_surface[None, None, :] + 0.4 * v_commit[None, None, :]
    h_unreached = h_twin.copy()
    h_reached = h_twin.copy()

    h_unreached[:, 14:20, :] += 2.5 * v_commit[None, None, :]
    h_reached[:, 14:20, :] += 2.8 * v_commit[None, None, :]
    h_reached[:, 20:, :] += 3.5 * v_commit[None, None, :]

    s_clean = np.dot(h_clean[:, 16, :], v_commit)
    s_twin = np.dot(h_twin[:, 16, :], v_commit)
    s_untrig = np.dot(h_untrig[:, 16, :], v_commit)
    s_unreached = np.dot(h_unreached[:, 16, :], v_commit)
    s_reached = np.dot(h_reached[:, 16, :], v_commit)

    twin_res = evaluate_benign_twin_separation(s_unreached, s_twin)
    hyp_res = evaluate_untriggerable_contrast(s_clean, s_twin, s_untrig, s_unreached, s_reached)

    div_curve = compute_divergence_curve(h_reached, h_unreached)
    bif_layer = find_bifurcation_layer(div_curve)

    causal_res = evaluate_causal_effects(h_unreached[:, 16, :], h_reached[:, 16, :], v_commit, alpha=1.5)

    return {
        "Benign Twin AUROC": twin_res["twin_auroc"],
        "H0 Falsified": "YES (Ruled out)" if twin_res["h0_falsified"] else "NO",
        "Untriggerable Decision": hyp_res.favored_hypothesis,
        "Bifurcation Layer": bif_layer,
        "Steering Delta": causal_res["steering_unreached"].delta,
        "Ablation Success": "YES" if causal_res["ablation_reached"].success else "NO",
    }


def main():
    parser = argparse.ArgumentParser(description="RQ3: Mechanism checks (H0, H1, H2, causality, divergence)")
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    print("================================================================================")
    print("RQ3 EVALUATION: MECHANISM CHECKS (BENIGN TWINS, UNTRIGGERABLE, CAUSALITY)")
    print("================================================================================\n")

    results = run_rq3_evaluation(n_samples=args.samples)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<28}: {v:.4f}")
        else:
            print(f"  {k:<28}: {v}")
    print()


if __name__ == "__main__":
    main()
