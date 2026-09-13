import numpy as np
import pytest

from rr_gid_cn.generator_validation import (
    conditional_reconstruction_report,
    information_fidelity_report,
    sampling_health_report,
    secondary_diagnostics_report,
    spearman_correlation,
    weighted_covariance,
)


def test_spearman_handles_perfect_and_reversed_rankings():
    assert spearman_correlation(np.array([1, 2, 3]), np.array([2, 4, 8])) == pytest.approx(1.0)
    assert spearman_correlation(np.array([1, 2, 3]), np.array([8, 4, 2])) == pytest.approx(-1.0)


def test_weighted_covariance_is_symmetric_and_centered():
    x = np.array([[0.0, 1.0], [2.0, 1.0], [4.0, 5.0]])
    cov = weighted_covariance(x, np.array([1.0, 2.0, 1.0]))
    assert cov == pytest.approx(cov.T)
    assert np.linalg.eigvalsh(cov).min() >= -1e-12


def test_information_fidelity_identity_passes():
    fisher = np.eye(2)
    infos = np.array([
        np.diag([2.0, 0.2]),
        np.diag([0.2, 2.0]),
        np.diag([1.0, 1.0]),
    ])
    result = information_fidelity_report(
        fisher, infos, infos, top_k=2, top_k_overlap_min=1.0, fw_max_iter=30
    )
    assert result["passed"] is True
    assert result["panel_ranking_spearman"] == pytest.approx(1.0)
    assert result["validation_design_ratio"] == pytest.approx(1.0)


def test_conditional_and_sampling_checks_fail_closed():
    reconstruction = conditional_reconstruction_report(
        np.zeros((2, 2)), np.array([[0.0, 0.0], [50.0, 0.0]]), np.ones(2)
    )
    assert reconstruction["passed"] is False
    assert "catastrophic_conditional_score_outlier" in reconstruction["failed_gates"]
    sampling = sampling_health_report(
        importance_ess_fraction=0.05,
        unconditional_acceptance=0.01,
        conditional_acceptance=1e-5,
    )
    assert sampling["passed"] is False
    assert "importance_ess" in sampling["failed_gates"]
    assert "conditional_acceptance" in sampling["failed_gates"]


def test_secondary_report_separates_completion_from_scientific_failure():
    row = {
        "campaign": "batch7",
        "method": "RR-GID",
        "heldout_mean_rmse": 0.1,
        "heldout_std_rmse": 0.1,
        "c2st_auc": 0.95,
        "importance_ess_fraction": 0.5,
        "unconditional_acceptance": 0.01,
        "conditional_acceptance": 0.01,
        "lambda_min_M": 1e-4,
        "primary_loss_abs_delta": 0.0,
        "fw_gap": 1e-5,
    }
    report = secondary_diagnostics_report([row], expected_rows=1)
    assert report["diagnostic_complete"] is True
    assert report["scientific_gate_passed"] is False
    assert report["scientific_failures"] == ["batch7:RR-GID:c2st_auc"]
