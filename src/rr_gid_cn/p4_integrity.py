"""Phase-1 reproducibility and validation helpers for the P4 gate."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


FORMAL_POLICIES = ("Uniform SQD", "A-OSQD", "oracle RR-GID")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_pilot_budget(schedule: dict[str, Any], budget: int) -> int:
    """Compute the pilot once in the runner from an auditable schedule."""
    kind = str(schedule.get("kind", ""))
    if kind == "anchored_power":
        required = {"anchor_budget", "anchor_pilot", "exponent", "max_fraction"}
        missing = required - set(schedule)
        if missing:
            raise ValueError(f"pilot schedule missing fields: {sorted(missing)}")
        ratio = float(budget) / float(schedule["anchor_budget"])
        raw = float(schedule["anchor_pilot"]) * (ratio ** float(schedule["exponent"]))
        count = int(math.ceil(raw))
        count = min(count, int(math.floor(float(schedule["max_fraction"]) * budget)), int(budget))
        return max(0, count)
    required = {"kind", "exponent", "multiplier", "max_fraction", "min_per_support", "rounding_rule"}
    missing = required - set(schedule)
    if missing:
        raise ValueError(f"pilot schedule missing fields: {sorted(missing)}")
    if kind != "power":
        raise ValueError(f"unsupported pilot schedule kind: {schedule['kind']}")
    raw = float(schedule["multiplier"]) * float(budget) ** float(schedule["exponent"])
    rounding = str(schedule["rounding_rule"])
    if rounding == "ceil":
        count = int(math.ceil(raw))
    elif rounding == "floor":
        count = int(math.floor(raw))
    elif rounding == "nearest":
        count = int(round(raw))
    else:
        raise ValueError(f"unsupported pilot rounding rule: {rounding}")
    count = max(count, 6 * int(schedule["min_per_support"]))
    count = min(count, int(math.floor(float(schedule["max_fraction"]) * budget)), int(budget))
    return max(0, count)


def validate_experiment_mode(cfg: dict[str, Any], *, formal: bool = True, paper: bool = False) -> None:
    """Reject ambiguous exact-score labels before any expensive work starts."""
    if paper:
        mode = str(cfg.get("experiment_mode", cfg.get("score_backend", "cached_qmc")))
        if mode in {"rejection", "finite_lu_rejection"}:
            raise ValueError("paper bulk forbids rejection")
        replications = int(cfg.get("replications", 0) or 0)
        # The PDF specifies at least 200 target replications for the main
        # synthetic result.  Keep an explicit upper guard so a malformed
        # config cannot silently request an unbounded run.
        if replications > 200:
            raise ValueError("paper experiments support at most 200 replications")
        score_backend = str(cfg.get("score_backend", "cached_qmc"))
        if score_backend not in {"cached_qmc", "fixed_qmc", "exact_adaptive"}:
            raise ValueError(
                "paper experiments require score_backend in {cached_qmc, fixed_qmc, exact_adaptive}"
            )
        if score_backend == "exact_adaptive":
            if not bool(cfg.get("exact_observed_score", False)):
                raise ValueError(
                    "exact_adaptive paper runs require exact_observed_score=true"
                )
            if int(cfg.get("adaptive_scrambles", 0) or 0) < 2:
                raise ValueError("exact_adaptive paper runs require adaptive_scrambles >= 2")
            if int(cfg.get("adaptive_max_order", 0) or 0) < int(cfg.get("score_qmc_order", 0) or 0):
                raise ValueError("adaptive_max_order must cover score_qmc_order")
        if score_backend == "fixed_qmc":
            # The streaming route is mathematically the same nested QMC
            # evaluator as cached_qmc, but it does not retain the full score
            # basis in host RAM.  It is admissible only with an explicit,
            # order-matched equivalence certificate; this prevents a silent
            # switch to an unvalidated numerical path in formal runs.
            certificate = cfg.get("fixed_qmc_equivalence_certificate")
            if not certificate:
                raise ValueError(
                    "fixed_qmc formal bulk requires fixed_qmc_equivalence_certificate"
                )
            cert_path = Path(str(certificate))
            if not cert_path.exists():
                raise ValueError(f"fixed_qmc equivalence certificate not found: {cert_path}")
            try:
                cert = json.loads(cert_path.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - defensive config gate
                raise ValueError(f"invalid fixed_qmc equivalence certificate: {cert_path}") from exc
            if not bool(cert.get("passed")):
                raise ValueError("fixed_qmc equivalence certificate is not passed")
            configured_order = int(cfg.get("score_qmc_order", -1))
            if int(cert.get("order", -2)) != configured_order:
                raise ValueError(
                    "fixed_qmc equivalence certificate order does not match score_qmc_order"
                )
        return
    if not formal:
        return
    p4 = cfg
    if "use_oracle_H" in p4:
        raise ValueError("formal config forbids ambiguous use_oracle_H; use frozen_beta_star_information")
    mode = str(p4.get("experiment_mode", ""))
    method = str(p4.get("conditional_method", ""))
    if mode in {"oracle_gold_qmc", "validated_fixed_qmc"}:
        if method not in {"qmc", "exact_adaptive"}:
            raise ValueError(f"{mode} requires a certified QMC conditional method")
        if not p4.get("qmc_error_diagnostics", False):
            raise ValueError(f"{mode} requires qmc_error_diagnostics=true")
    elif mode == "finite_lu_rejection":
        if method != "rejection":
            raise ValueError("finite_lu_rejection requires conditional_method=rejection")
        if not p4.get("sandwich_benchmark", False):
            raise ValueError("finite_lu_rejection requires a sandwich benchmark")
    else:
        raise ValueError(f"unsupported formal experiment_mode: {mode}")
    if bool(p4.get("exact_observed_score", False)) and mode == "finite_lu_rejection":
        raise ValueError("finite-LU rejection mean cannot be labelled exact_observed_score")


def load_seed_manifest(path: Path) -> tuple[dict[tuple[int, int], dict[str, int]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    keys = [(int(row["budget"]), int(row["replication"])) for row in rows]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate seed manifest entries: {duplicates[:5]}")
    manifest = {}
    required = {
        "replication_seed", "target_draw_seed", "pilot_or_design_seed",
        "score_seed_root", "information_seed_root",
    }
    for key, row in zip(keys, rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"manifest entry {key} missing seed fields: {sorted(missing)}")
        manifest[key] = {name: int(row[name]) for name in required}
    return manifest, sha256_file(path)


def require_manifest_grid(
    manifest: dict[tuple[int, int], dict[str, int]], budgets: Iterable[int], replications: Iterable[int]
) -> None:
    expected = {(int(b), int(r)) for b in budgets for r in replications}
    observed = set(manifest)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(f"seed manifest grid mismatch: missing={missing[:5]}, extra={extra[:5]}")


def validate_artifact_metadata(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"prepared artifact metadata mismatch: {mismatches}")


def validate_expected_grid(
    rows: list[dict[str, Any]],
    budgets: Iterable[int],
    replications: Iterable[int],
    policies: Iterable[str] = FORMAL_POLICIES,
) -> None:
    expected = {
        (int(budget), int(replication), str(policy))
        for budget in budgets for replication in replications for policy in policies
    }
    keys = [
        (int(row["budget"]), int(row["replication"]), str(row["policy"]))
        for row in rows
    ]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    observed = set(keys)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if duplicates or missing or extra:
        raise ValueError(
            f"result grid mismatch: missing={missing[:5]}, extra={extra[:5]}, "
            f"duplicates={duplicates[:5]}"
        )
