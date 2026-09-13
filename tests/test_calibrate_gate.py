import importlib.util
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_fast_backend.py"
_SPEC = importlib.util.spec_from_file_location("calibrate_fast_backend", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def test_e2e_gate_uses_signed_paired_se_and_reports_fields():
    rows = [
        {"fast": 1.1, "gold": 1.0, "gold_converged": True},
        {"fast": 1.1, "gold": 1.0, "gold_converged": True},
    ]
    result = _MODULE.evaluate_e2e_gate(rows, n_e2e=2, cal={"end_to_end_mean_abs_delta": 0.15})
    assert result["signed_mean_delta"] == pytest.approx(0.1)
    assert result["signed_paired_se"] == pytest.approx(0.0)
    assert result["mean_abs_delta"] == pytest.approx(0.1)
    assert result["failed_gates"] == ["signed_mean_delta_not_within_paired_se"]
    assert result["ok"] is False


def test_e2e_gate_two_rep_opposite_signed_errors_can_pass():
    rows = [
        {"fast": 1.03, "gold": 1.0, "gold_converged": True},
        {"fast": 0.95, "gold": 1.0, "gold_converged": True},
    ]
    result = _MODULE.evaluate_e2e_gate(rows, n_e2e=2, cal={"end_to_end_mean_abs_delta": 0.15})
    assert result["mean_abs_delta"] == pytest.approx(0.04)
    assert abs(result["signed_mean_delta"]) == pytest.approx(0.01)
    assert result["signed_paired_se"] == pytest.approx(0.04)
    assert result["failed_gates"] == []
    assert result["ok"] is True


def test_e2e_gate_requires_every_gold_route_to_converge():
    rows = [
        {"fast": 1.01, "gold": 1.0, "gold_converged": True},
        {"fast": 1.01, "gold": 1.0, "gold_converged": False},
    ]
    result = _MODULE.evaluate_e2e_gate(rows, n_e2e=2, cal={"end_to_end_mean_abs_delta": 0.15})
    assert result["gold_converged_frac"] == 0.5
    assert "gold_convergence" in result["failed_gates"]
    assert result["ok"] is False
