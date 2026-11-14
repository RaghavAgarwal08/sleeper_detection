import numpy as np
import pytest

from rq1_baselines import AttentionTracker, AttnDA
from rq2_probing import SubspaceDivergenceProbe, run_layer_sweep
from rq3_mechanisms import (
    evaluate_benign_twin_separation,
    evaluate_untriggerable_contrast,
    compute_divergence_curve,
    evaluate_causal_effects,
)
from rq4_transfer import (
    train_reached_test_unreached,
    compute_incident_curve,
    evaluate_lodo_transfer,
)
from benchmarks.memorydojo import (
    TaskDomain,
    MemorySystemType,
    MemorySystemSimulator,
    build_synthetic_trajectory,
    compute_lifecycle_metrics,
)


def test_rq1_attention_tracker_and_attnda():
    rng = np.random.default_rng(123)
    clean_fs = rng.normal(loc=0.85, scale=0.05, size=(20, 4, 8))
    malicious_fs = rng.normal(loc=0.30, scale=0.05, size=(20, 4, 8))

    X = np.concatenate([clean_fs, malicious_fs], axis=0)
    y = np.array([0] * 20 + [1] * 20)

    at = AttentionTracker(k=3.0).fit(X, y)
    assert len(at.selected_heads) > 0
    assert at.evaluate(X, y).auroc > 0.95

    lda = AttnDA(epsilon=1e-3).fit(X, y)
    assert lda.evaluate(X, y).auroc > 0.95


def test_rq2_subspace_probe_and_layer_sweep():
    rng = np.random.default_rng(123)
    d = 128
    v_surface = rng.normal(size=d)
    v_surface /= np.linalg.norm(v_surface)

    v_commit = rng.normal(size=d)
    v_commit -= np.dot(v_commit, v_surface) * v_surface
    v_commit /= np.linalg.norm(v_commit)

    h_clean = rng.normal(size=(20, 10, d))
    h_twin = h_clean + 1.5 * v_surface[None, None, :]
    h_unreached = h_twin.copy()
    h_unreached[:, 5:8, :] += 2.0 * v_commit[None, None, :]
    h_reached = h_unreached.copy()
    h_reached[:, 8:, :] += 3.0 * v_commit[None, None, :]

    probe = SubspaceDivergenceProbe(target_layers=(5, 8)).fit(
        hidden_states_unreached=h_unreached,
        hidden_states_benign_twin=h_twin,
        hidden_states_clean=h_clean,
    )

    dot_surface = np.abs(np.dot(probe.commitment_vector, probe.surface_vector))
    assert dot_surface < 1e-5

    test_X = np.concatenate([h_twin, h_unreached], axis=0)
    test_y = np.array([0] * 20 + [1] * 20)
    assert probe.evaluate(test_X, test_y).auroc > 0.95

    sweep = run_layer_sweep(h_clean, h_unreached, h_reached)
    assert sweep.best_mid_layer in [5, 6, 7]


def test_rq3_mechanisms_and_causality():
    rng = np.random.default_rng(42)
    s_unreached = rng.normal(loc=2.0, scale=0.3, size=20)
    s_twin = rng.normal(loc=0.1, scale=0.3, size=20)

    twin_res = evaluate_benign_twin_separation(s_unreached, s_twin)
    assert twin_res["h0_falsified"] is True

    s_clean = rng.normal(loc=0.0, scale=0.2, size=20)
    s_untrig = rng.normal(loc=0.3, scale=0.2, size=20)
    s_reached = rng.normal(loc=3.0, scale=0.3, size=20)
    hyp_res = evaluate_untriggerable_contrast(s_clean, s_twin, s_untrig, s_unreached, s_reached)
    assert "H1" in hyp_res.favored_hypothesis

    v = rng.normal(size=64)
    v /= np.linalg.norm(v)
    h_unreached = rng.normal(size=(20, 64))
    h_reached = h_unreached + 2.0 * v
    causal_res = evaluate_causal_effects(h_unreached, h_reached, v, alpha=1.5)
    assert bool(causal_res["steering_unreached"].success)
    assert bool(causal_res["ablation_reached"].success)


def test_rq4_cross_state_and_lodo():
    rng = np.random.default_rng(42)
    d = 64
    v = rng.normal(size=d)
    v /= np.linalg.norm(v)

    c = rng.normal(size=(20, 10, d))
    u_act = c + 2.0 * v[None, None, :]
    r_act = c + 2.5 * v[None, None, :]

    trans = train_reached_test_unreached(c[:10], r_act[:10], c[10:], u_act[10:], target_layer=5)
    assert trans.transfer_auroc > 0.90

    curve = compute_incident_curve(c, r_act, u_act, counts=[2, 5], target_layer=5)
    assert len(curve.probe_tpr_at_1fpr) == 2

    family_data = {
        "fam1": {"clean": c, "unreached": u_act, "reached": r_act},
        "fam2": {"clean": c, "unreached": u_act, "reached": r_act},
    }
    lodo = evaluate_lodo_transfer(family_data, target_layer=5)
    assert lodo["mean_lodo_auroc"] > 0.90


def test_memorydojo_lifecycle_and_simulation():
    traj = build_synthetic_trajectory(TaskDomain.WORKSPACE, length=5, t0=0)
    assert len(traj.sessions) == 5
    assert traj.insertion_session_idx == 0

    metrics = compute_lifecycle_metrics([traj])
    assert metrics.isr == 1.0
    assert metrics.rsr > 0.0
    assert metrics.br == 1.0

    sim = MemorySystemSimulator(MemorySystemType.LETTA_CORE)
    sim.write(session_idx=0, observations=["Note 1", "Note 2"])
    retrieved = sim.retrieve("query", k=1)
    assert len(retrieved) == 1
