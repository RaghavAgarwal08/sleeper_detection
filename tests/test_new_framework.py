import numpy as np
import pytest

from interp.attention_tracker import AttentionTracker
from interp.attnda import AttnDA
from interp.subspace_probe import SubspaceDivergenceProbe
from interp.memorydojo import (
    TaskDomain,
    MemorySystemType,
    MemorySystemSimulator,
    build_synthetic_trajectory,
    compute_lifecycle_metrics,
)


def test_attention_tracker_fit_and_score():
    rng = np.random.default_rng(123)
    clean_fs = rng.normal(loc=0.85, scale=0.05, size=(20, 4, 8))
    malicious_fs = rng.normal(loc=0.30, scale=0.05, size=(20, 4, 8))

    X = np.concatenate([clean_fs, malicious_fs], axis=0)
    y = np.array([0] * 20 + [1] * 20)

    at = AttentionTracker(k=3.0).fit(X, y)
    assert len(at.selected_heads) > 0

    eval_res = at.evaluate(X, y)
    assert eval_res.auroc > 0.95


def test_attnda_fisher_discriminant():
    rng = np.random.default_rng(123)
    clean_fs = rng.normal(loc=0.85, scale=0.05, size=(20, 4, 8))
    malicious_fs = rng.normal(loc=0.30, scale=0.05, size=(20, 4, 8))

    X = np.concatenate([clean_fs, malicious_fs], axis=0)
    y = np.array([0] * 20 + [1] * 20)

    model = AttnDA(epsilon=1e-3).fit(X, y)
    eval_res = model.evaluate(X, y)
    assert eval_res.auroc > 0.95


def test_subspace_probe_orthogonal_commitment():
    rng = np.random.default_rng(123)
    d = 128
    v_surface = rng.normal(size=d)
    v_surface /= np.linalg.norm(v_surface)

    v_commit = rng.normal(size=d)
    v_commit -= np.dot(v_commit, v_surface) * v_surface
    v_commit /= np.linalg.norm(v_commit)

    h_clean = rng.normal(size=(20, 10, d))
    h_twin = h_clean + 1.5 * v_surface[None, None, :]
    h_dormant = h_twin.copy()
    h_dormant[:, 5:8, :] += 2.0 * v_commit[None, None, :]

    probe = SubspaceDivergenceProbe(target_layers=(5, 8)).fit(
        hidden_states_dormant=h_dormant,
        hidden_states_twin=h_twin,
        hidden_states_clean=h_clean,
    )

    dot_surface = np.abs(np.dot(probe.commitment_vector, probe.surface_vector))
    assert dot_surface < 1e-5

    test_X = np.concatenate([h_twin, h_dormant], axis=0)
    test_y = np.array([0] * 20 + [1] * 20)
    res = probe.evaluate(test_X, test_y)
    assert res.auroc > 0.95


def test_memorydojo_lifecycle():
    traj = build_synthetic_trajectory(TaskDomain.WORKSPACE, length=5, t0=0)
    assert len(traj.sessions) == 5
    assert traj.insertion_session_idx == 0

    metrics = compute_lifecycle_metrics([traj])
    assert metrics.isr == 1.0
    assert metrics.rsr > 0.0
    assert metrics.br == 1.0


def test_memory_simulator():
    sim = MemorySystemSimulator(MemorySystemType.LETTA_CORE)
    sim.write(session_idx=0, observations=["Note 1", "Note 2"])
    retrieved = sim.retrieve("query", k=1)
    assert len(retrieved) == 1
    assert retrieved[0].session_id == 0
