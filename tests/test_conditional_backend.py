from __future__ import annotations

import numpy as np
import pytest

from rr_gid_cn.rrgid import psd_project
from rr_gid_cn.conditional_backend import (
    FixedConditionalMean,
    DirectConditionalMean,
    GroupedPanelData,
    ReferenceMomentCache,
    build_conditional_feature_basis,
    build_direct_panel_information_basis,
    build_full_law_moment_cache,
    build_panel_information_basis,
    cholesky_solve,
    evaluate_conditional_basis,
    uncached_qmc_mean,
)
from rr_gid_cn.conditional_backend import build_shared_score_bases
from rr_gid_cn.p4_integrity import compute_pilot_budget
from rr_gid_cn.synthetic_oracle import (
    feature_map,
    full_target_kl,
    make_frozen_mixture,
    reference_scale,
    sample_full,
)


def _cpu_qmc(monkeypatch):
    import rr_gid_cn.synthetic_oracle as oracle

    monkeypatch.setattr(oracle, "_cuda_device", lambda: None)


def test_t1_cached_matches_uncached_qmc(monkeypatch):
    _cpu_qmc(monkeypatch)
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=400, seed=7)
    panel = (0, 1)
    rows = sample_full(mixture, 8, seed=19)[:3][:, list(panel)]
    beta = np.linspace(-0.3, 0.4, 12)
    order, seed = 4, 123
    basis = build_conditional_feature_basis(
        mixture, rows, panel, order=order, seed=seed, scale=scale, dtype="float64",
    )
    cached = evaluate_conditional_basis(beta, basis)
    fresh = evaluate_conditional_basis(beta, basis)
    assert np.max(np.abs(cached - fresh)) <= 1e-12
    uncached = uncached_qmc_mean(mixture, beta, rows, panel, order, seed, scale)
    assert np.max(np.abs(cached - uncached)) <= 1e-10


def test_disk_backed_basis_streaming_matches_in_memory(monkeypatch, tmp_path):
    _cpu_qmc(monkeypatch)
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=400, seed=7)
    panel = (0, 1)
    rows = sample_full(mixture, 8, seed=19)[:5][:, list(panel)]
    beta = np.linspace(-0.3, 0.4, 12)
    kwargs = dict(order=5, seed=123, scale=scale, dtype="float32")
    memory = build_conditional_feature_basis(mixture, rows, panel, **kwargs)
    disk = build_conditional_feature_basis(
        mixture,
        rows,
        panel,
        storage_path=tmp_path / "score.dat",
        **kwargs,
    )
    assert isinstance(disk.phi_nodes, np.memmap)
    mean_memory, cov_memory = memory.conditional_mean_and_cov(beta)
    mean_disk, cov_disk = disk.conditional_mean_and_cov(beta)
    # The disk-backed basis is deliberately stored in the configured hot
    # dtype (float32 here); reductions still accumulate in float64, so the
    # comparison is against storage precision rather than bit identity.
    assert np.max(np.abs(mean_memory - mean_disk)) <= 5e-7
    assert np.max(np.abs(cov_memory - cov_disk)) <= 5e-7


def test_fixed_qmc_stream_matches_cached_basis(monkeypatch):
    _cpu_qmc(monkeypatch)
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=400, seed=7)
    panel = (0, 1)
    rows = sample_full(mixture, 8, seed=19)[:5][:, list(panel)]
    beta = np.linspace(-0.3, 0.4, 12)
    basis = build_conditional_feature_basis(
        mixture, rows, panel, order=5, seed=123, scale=scale, dtype="float64",
    )
    streamed = FixedConditionalMean(
        mixture, rows, panel, 123, scale, order=5, row_chunk=2,
    )
    assert np.max(
        np.abs(basis.conditional_mean(beta) - streamed.conditional_mean(beta))
    ) <= 1e-10


def test_t2_beta_zero_is_q0_conditional_mean(monkeypatch):
    _cpu_qmc(monkeypatch)
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=400, seed=7)
    panel = (2, 5)
    rows = sample_full(mixture, 4, seed=21)[:, list(panel)]
    basis = build_conditional_feature_basis(
        mixture, rows, panel, order=4, seed=5, scale=scale, dtype="float64",
    )
    cached = evaluate_conditional_basis(np.zeros(12), basis)
    phi = basis.phi_nodes.astype(np.float64)
    prob = np.exp(basis.component_log_posterior)
    prob = prob / prob.sum(axis=1, keepdims=True)
    q0 = np.einsum("kc,kcnr->kr", prob, phi) / phi.shape[2]
    assert np.max(np.abs(cached - q0)) <= 1e-10


def test_t4_cholesky_matches_numpy_solve():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 6))
    h = a.T @ a + 0.2 * np.eye(6)
    u = rng.normal(size=6)
    step, diag = cholesky_solve(h, u, ridge=1e-10)
    expected = np.linalg.solve(h, u)
    assert np.linalg.norm(step - expected) <= 1e-10
    assert diag["ridge"] == 0.0
    assert diag["linear_algebra"] == "cholesky"


def test_anchored_pilot_schedule_matches_guide_table():
    schedule = {
        "kind": "anchored_power",
        "anchor_budget": 8000,
        "anchor_pilot": 400,
        "exponent": 0.5,
        "max_fraction": 0.2,
    }
    expected = {2000: 200, 4000: 283, 8000: 400, 16000: 566, 32000: 800}
    for budget, target in expected.items():
        assert compute_pilot_budget(schedule, budget) == target


def test_reference_moment_cache_matches_feature_map():
    mixture = make_frozen_mixture(seed=3)
    scale = reference_scale(mixture, n=200, seed=4)
    reference = sample_full(mixture, 64, seed=5)
    cache = ReferenceMomentCache.build(reference, scale, beta_true=np.zeros(12))
    beta = np.linspace(-0.2, 0.15, 12)
    mu, fisher = cache.moments(beta)
    features = feature_map(reference, scale)
    logits = features @ beta
    w = np.exp(logits - logits.max())
    w /= w.sum()
    assert np.linalg.norm(mu - w @ features) <= 1e-12
    assert cache.kl(np.zeros(12), np.zeros(12)) == 0.0
    beta_true = np.linspace(-0.15, 0.2, 12)
    cache_true = ReferenceMomentCache.build(reference, scale, beta_true=beta_true)
    beta_hat = beta_true + 0.05
    kl = cache_true.kl(beta_true, beta_hat)
    assert kl > 0.0
    assert abs(kl - full_target_kl(beta_true, beta_hat, reference, scale)) <= 1e-12


def test_information_beta_zero_matches_unweighted_covariance(monkeypatch):
    _cpu_qmc(monkeypatch)
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=200, seed=4)
    reference = sample_full(mixture, 64, seed=5)
    panels = ((0, 1), (2, 5))
    basis = build_panel_information_basis(
        mixture, reference, panels, scale=scale, outer_rows=16, order=4, seed=9, dtype="float64",
    )
    infos = basis.information(np.zeros(12), panels)
    mu = basis.outer_phi.mean(0)
    n = len(basis.outer_phi)
    for panel in panels:
        a = basis.basis_a[panel].conditional_mean(np.zeros(12)) - mu
        b = basis.basis_b[panel].conditional_mean(np.zeros(12)) - mu
        a = a - a.mean(0)
        b = b - b.mean(0)
        expected = psd_project((a.T @ b + b.T @ a) / (2 * (n - 1)))
        assert np.max(np.abs(infos[panel] - expected)) <= 1e-10


def test_information_uses_tilt_weights_when_beta_nonzero(monkeypatch):
    _cpu_qmc(monkeypatch)
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=200, seed=4)
    reference = sample_full(mixture, 64, seed=5)
    panels = ((0, 1),)
    basis = build_panel_information_basis(
        mixture, reference, panels, scale=scale, outer_rows=16, order=4, seed=9, dtype="float64",
    )
    beta = np.linspace(-0.2, 0.15, 12)
    weighted = basis.information(beta, panels)[panels[0]]
    mu = basis.outer_phi.mean(0)
    a = basis.basis_a[panels[0]].conditional_mean(beta) - mu
    b = basis.basis_b[panels[0]].conditional_mean(beta) - mu
    a = a - a.mean(0)
    b = b - b.mean(0)
    unweighted = (a.T @ b + b.T @ a) / (2 * (len(a) - 1))
    assert np.linalg.norm(weighted - unweighted, ord="fro") > 1e-6


def test_information_memmap_storage_matches_in_memory(monkeypatch, tmp_path):
    _cpu_qmc(monkeypatch)
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=200, seed=4)
    reference = sample_full(mixture, 64, seed=5)
    panels = ((0, 1), (2, 5))
    plain = build_panel_information_basis(
        mixture, reference, panels, scale=scale, outer_rows=16, order=4, seed=9, dtype="float64",
    )
    streamed = build_panel_information_basis(
        mixture, reference, panels, scale=scale, outer_rows=16, order=4, seed=9,
        dtype="float64", storage_dir=tmp_path / "basis",
    )
    beta = np.linspace(-0.2, 0.15, 12)
    plain_info = plain.information(beta, panels)
    streamed_info = streamed.information(beta, panels)
    for panel in panels:
        assert isinstance(streamed.basis_a[panel].phi_nodes, np.memmap)
        assert np.max(np.abs(plain_info[panel] - streamed_info[panel])) <= 1e-12

    reused = build_panel_information_basis(
        mixture,
        reference,
        panels,
        scale=scale,
        outer_rows=16,
        order=4,
        seed=9,
        dtype="float64",
        storage_dir=tmp_path / "basis",
        reuse_existing=True,
    )
    reused_info = reused.information(beta, panels)
    for panel in panels:
        assert reused.basis_a[panel].phi_nodes.mode == "r"
        assert np.max(np.abs(plain_info[panel] - reused_info[panel])) <= 1e-12


def test_information_memmap_reuse_rejects_incomplete_file(monkeypatch, tmp_path):
    _cpu_qmc(monkeypatch)
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=200, seed=4)
    reference = sample_full(mixture, 64, seed=5)
    storage = tmp_path / "basis"
    build_panel_information_basis(
        mixture,
        reference,
        ((0, 1),),
        scale=scale,
        outer_rows=16,
        order=4,
        seed=9,
        dtype="float64",
        storage_dir=storage,
    )
    path = storage / "basis_a_0_1.dat"
    path.write_bytes(path.read_bytes()[:-8])
    import pytest

    with pytest.raises(ValueError, match="invalid reusable information basis size"):
        build_panel_information_basis(
            mixture,
            reference,
            ((0, 1),),
            scale=scale,
            outer_rows=16,
            order=4,
            seed=9,
            dtype="float64",
            storage_dir=storage,
            reuse_existing=True,
        )


def test_weighted_cross_information_matches_basis(monkeypatch):
    _cpu_qmc(monkeypatch)
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=200, seed=4)
    reference = sample_full(mixture, 64, seed=5)
    panels = ((0, 1), (2, 5))
    basis = build_panel_information_basis(
        mixture, reference, panels, scale=scale, outer_rows=16, order=4, seed=9, dtype="float64",
    )
    from rr_gid_cn.conditional_backend import weighted_cross_information
    beta = np.linspace(-0.25, 0.2, 12)
    weights = basis.tilt_weights(beta)
    cached = basis.information(beta, panels)
    for panel in panels:
        rebuilt = weighted_cross_information(
            basis.basis_a[panel].conditional_mean(beta),
            basis.basis_b[panel].conditional_mean(beta),
            weights,
            basis.outer_phi,
        )
        assert np.max(np.abs(cached[panel] - rebuilt)) <= 1e-10


def test_full_law_moment_cache_reweights():
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=200, seed=4)
    cache = build_full_law_moment_cache(mixture, scale, order=6, scrambles=1, seed=3)
    mu0, _ = cache.moments(np.zeros(12))
    mu1, _ = cache.moments(np.linspace(-0.2, 0.2, 12))
    assert mu0.shape == (12,)
    assert np.linalg.norm(mu0 - mu1) > 1e-6
    again, _ = cache.moments(np.zeros(12))
    assert np.linalg.norm(mu0 - again) <= 1e-12


def test_shared_score_basis_views_match_independent_bases(monkeypatch):
    _cpu_qmc(monkeypatch)
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=300, seed=7)
    panels = ((0, 1), (2, 5))
    rows_a = np.asarray([[0.1, -0.2], [0.3, 0.4], [0.5, -0.6]], dtype=float)
    rows_b = np.asarray([[0.3, 0.4], [0.7, 0.8]], dtype=float)
    grouped = {
        "a": GroupedPanelData.from_observations([
            (panels[0], rows_a[0]), (panels[0], rows_a[1]), (panels[1], rows_a[2]),
        ]),
        "b": GroupedPanelData.from_observations([
            (panels[0], rows_b[0]), (panels[0], rows_b[1]), (panels[1], rows_a[2]),
        ]),
    }
    views, _elapsed, meta = build_shared_score_bases(
        mixture, grouped, scale, order=4, seed=123, dtype="float64",
    )
    assert meta["shared"] is True
    beta = np.linspace(-0.3, 0.4, 12)
    for policy, data in grouped.items():
        for panel in data.panel_ids:
            independent = build_conditional_feature_basis(
                mixture, data.rows(panel), panel, order=4, seed=123, scale=scale,
                dtype="float64",
            ).conditional_mean(beta)
            actual = views[policy][panel].conditional_mean(beta)
            assert np.max(np.abs(actual - independent)) <= 1e-12


def test_direct_cuda_fixed_nodes_match_cached_basis():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for direct fixed-node score test")
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=300, seed=7)
    panel = (0, 1)
    rows = sample_full(mixture, 12, seed=19)[:7][:, list(panel)]
    beta = np.linspace(-0.3, 0.4, 12)
    basis = build_conditional_feature_basis(
        mixture, rows, panel, order=5, seed=123, scale=scale, dtype="float32",
    )
    direct = DirectConditionalMean(
        mixture, rows, panel, 123, scale, order=5, row_chunk=3,
    )
    expected = basis.conditional_mean(beta)
    actual = direct.conditional_mean(beta)
    assert np.max(np.abs(expected - actual)) <= 5e-6


def test_direct_information_matches_cached_information_on_cpu(monkeypatch):
    _cpu_qmc(monkeypatch)
    import rr_gid_cn.conditional_backend as backend

    monkeypatch.setattr(backend, "_cuda_for_basis", lambda: None)
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=300, seed=7)
    reference = sample_full(mixture, 64, seed=19)
    panels = ((0, 1), (2, 5))
    cached = build_panel_information_basis(
        mixture, reference, panels, scale=scale, outer_rows=16, order=4,
        seed=123, dtype="float64",
    )
    direct = build_direct_panel_information_basis(
        mixture, reference, panels, scale=scale, outer_rows=16, order=4,
        seed=123,
    )
    beta = np.linspace(-0.25, 0.2, 12)
    cached_info = cached.information(beta, panels)
    direct_info = direct.information(beta, panels)
    assert np.max(np.abs(cached.outer_phi - direct.outer_phi)) <= 1e-12
    for panel in panels:
        assert np.max(np.abs(cached_info[panel] - direct_info[panel])) <= 1e-12
