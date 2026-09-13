"""Frozen empirical kNN generator: a drop-in conditional-law interface for Gas.

The failed VAEAC checkpoint does not match the Gas reference law (C2ST AUC ~1,
anti-correlated panel ranking).  Figure 3 only needs a frozen sampler for
Q_0(x) and Q_0(x | x_S).  Kernel-weighted nearest neighbors on the frozen
reference-train split is that interface, and it is the same estimator already
used as the Gas RISK-2 validation reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .generator_validation import weighted_covariance
from .synthetic_oracle import feature_map


def _stable_tilt_weights(features: np.ndarray, beta: np.ndarray) -> np.ndarray:
    logits = np.asarray(features, dtype=float) @ np.asarray(beta, dtype=float)
    weights = np.exp(logits - float(np.max(logits)))
    return weights / weights.sum()


def _project_psd(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    sym = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    values, vectors = np.linalg.eigh(sym)
    return (vectors * np.maximum(values, floor)) @ vectors.T


def empirical_conditional_means(
    donor_x: np.ndarray,
    donor_phi: np.ndarray,
    query_observed: np.ndarray,
    panel: tuple[int, ...],
    *,
    donor_tilt_weights: np.ndarray,
    neighbors: int,
    query_chunk: int = 128,
) -> np.ndarray:
    columns = list(panel)
    donors = np.asarray(donor_x[:, columns], dtype=np.float64)
    queries = np.atleast_2d(np.asarray(query_observed, dtype=np.float64))
    phi = np.asarray(donor_phi, dtype=np.float64)
    tilt = np.asarray(donor_tilt_weights, dtype=np.float64)
    k = min(max(2, int(neighbors)), len(donors) - 1)
    outputs = []
    donor_norm = np.sum(donors * donors, axis=1)
    for start in range(0, len(queries), int(query_chunk)):
        query = queries[start:start + int(query_chunk)]
        d2 = np.sum(query * query, axis=1)[:, None] + donor_norm[None, :] - 2.0 * query @ donors.T
        d2 = np.maximum(d2, 0.0)
        indices = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        local_d2 = np.take_along_axis(d2, indices, axis=1)
        bandwidth2 = np.maximum(np.median(local_d2, axis=1), 1e-10)
        kernel = np.exp(-local_d2 / (2.0 * bandwidth2[:, None]))
        weights = kernel * tilt[indices]
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)
        outputs.append(np.einsum("qk,qkr->qr", weights, phi[indices]))
    return np.concatenate(outputs, axis=0)


def empirical_information_basis(
    donor_x: np.ndarray,
    donor_phi: np.ndarray,
    outer_x: np.ndarray,
    outer_phi: np.ndarray,
    panels: tuple[tuple[int, ...], ...],
    beta: np.ndarray,
    *,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    donor_weights = _stable_tilt_weights(donor_phi, beta)
    outer_weights = _stable_tilt_weights(outer_phi, beta)
    fisher = _project_psd(weighted_covariance(outer_phi, outer_weights))
    means, infos = [], []
    for panel in panels:
        conditional = empirical_conditional_means(
            donor_x,
            donor_phi,
            outer_x[:, list(panel)],
            panel,
            donor_tilt_weights=donor_weights,
            neighbors=neighbors,
        )
        means.append(conditional)
        infos.append(_project_psd(weighted_covariance(conditional, outer_weights)))
    return fisher, np.asarray(infos), np.asarray(means)


class EmpiricalKNNGenerator:
    """Bootstrap Q_0 plus kernel-weighted kNN conditionals on a frozen donor table."""

    def __init__(self, reference: np.ndarray, feature_fn=None, neighbors: int = 60, scale=None):
        self.reference = np.asarray(reference, dtype=np.float64)
        if self.reference.ndim != 2 or len(self.reference) < 8:
            raise ValueError("empirical generator requires a 2-D donor table")
        self.neighbors = int(max(2, min(int(neighbors), len(self.reference) - 1)))
        self.scale = np.ones(self.reference.shape[1], dtype=float) if scale is None else np.asarray(scale, float)
        self.feature_fn = feature_map if feature_fn is None else feature_fn
        self.device = "cpu"
        self.kind = "empirical_knn"

    def panel_information(self, beta, panels, outer_x, neighbors=None):
        """Tilt-weighted kNN information; same operator as the Gas RISK-2 reference."""
        donor_phi = self.feature_fn(self.reference)
        outer_phi = self.feature_fn(np.asarray(outer_x, dtype=float))
        return empirical_information_basis(
            self.reference,
            donor_phi,
            np.asarray(outer_x, dtype=float),
            outer_phi,
            tuple(panels),
            np.asarray(beta, dtype=float),
            neighbors=int(self.neighbors if neighbors is None else neighbors),
        )

    @property
    def dimension(self) -> int:
        return int(self.reference.shape[1])

    def sample_full(self, n, seed=0):
        rng = np.random.default_rng(seed)
        return self.reference[rng.integers(len(self.reference), size=int(n))].copy()

    def _neighbor_indices(self, queries: np.ndarray, panel: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        columns = list(panel)
        donors = self.reference[:, columns]
        query = np.atleast_2d(np.asarray(queries, dtype=np.float64))
        donor_norm = np.sum(donors * donors, axis=1)
        d2 = np.sum(query * query, axis=1)[:, None] + donor_norm[None, :] - 2.0 * query @ donors.T
        d2 = np.maximum(d2, 0.0)
        k = self.neighbors
        indices = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        local = np.take_along_axis(d2, indices, axis=1)
        bandwidth2 = np.maximum(np.median(local, axis=1), 1e-10)
        kernel = np.exp(-local / (2.0 * bandwidth2[:, None]))
        weights = kernel / np.maximum(kernel.sum(axis=1, keepdims=True), 1e-300)
        return indices, weights

    def sample_conditional(self, observed, panel, n, seed=0):
        indices, weights = self._neighbor_indices(np.asarray(observed, dtype=float)[None, :], panel)
        rng = np.random.default_rng(seed)
        pick = rng.choice(indices[0], size=int(n), p=weights[0])
        out = self.reference[pick].copy()
        out[:, list(panel)] = np.asarray(observed, dtype=float)
        return out

    def sample_conditional_batch(self, observed_batch, panel, n, seed=0):
        observed = np.atleast_2d(np.asarray(observed_batch, dtype=float))
        indices, weights = self._neighbor_indices(observed, panel)
        rng = np.random.default_rng(seed)
        rows = len(observed)
        cols = np.asarray(list(panel), dtype=int)
        out = np.empty((rows, int(n), self.dimension), dtype=np.float64)
        for i in range(rows):
            pick = rng.choice(indices[i], size=int(n), p=weights[i])
            out[i] = self.reference[pick]
        out[:, :, cols] = observed[:, None, :]
        return out

    def tilted_full_sample(self, beta, n, seed=0):
        sample, _, _ = self.tilted_full_diagnostics(beta, n, seed)
        return sample

    def tilted_full_diagnostics(self, beta, n, seed=0):
        """Exact discrete tilt of the frozen empirical measure Q0.

        Rejection sampling with a crude envelope is the wrong interface here:
        Q0 is a finite donor table, so weighted resampling is the exact law
        Q_β ∝ Q0 exp(φ(x)ᵀβ).
        """
        rng = np.random.default_rng(seed)
        logits = self.feature_fn(self.reference) @ np.asarray(beta, dtype=float)
        logits -= float(np.max(logits))
        weights = np.exp(logits)
        probabilities = weights / np.maximum(weights.sum(), 1e-300)
        idx = rng.choice(len(self.reference), size=int(n), replace=True, p=probabilities)
        out = self.reference[idx].copy()
        ess = float((weights.sum() ** 2) / np.maximum(np.sum(weights ** 2), 1e-300) / len(weights))
        return out, 1.0, ess

    def tilted_conditional_diagnostics(self, beta, observed, panel, n, seed=0):
        indices, kernel_w = self._neighbor_indices(np.asarray(observed, dtype=float)[None, :], panel)
        donor_idx = indices[0]
        logits = self.feature_fn(self.reference[donor_idx]) @ np.asarray(beta, dtype=float)
        logits -= float(np.max(logits))
        weights = kernel_w[0] * np.exp(logits)
        probabilities = weights / np.maximum(weights.sum(), 1e-300)
        rng = np.random.default_rng(seed)
        pick = rng.choice(donor_idx, size=int(n), replace=True, p=probabilities)
        out = self.reference[pick].copy()
        out[:, list(panel)] = np.asarray(observed, dtype=float)
        ess = float((weights.sum() ** 2) / np.maximum(np.sum(weights ** 2), 1e-300) / len(weights))
        return out, 1.0, ess

    def tilted_conditional_sample(self, beta, observed, panel, n, seed=0):
        sample, _, _ = self.tilted_conditional_diagnostics(beta, observed, panel, n, seed)
        return sample

    def importance_ess(self, beta, pool, pool_weights=None):
        logits = self.feature_fn(pool) @ np.asarray(beta)
        if pool_weights is not None:
            logits += np.log(np.asarray(pool_weights) + 1e-300)
        logits -= logits.max()
        weights = np.exp(logits)
        return float(weights.sum() ** 2 / np.sum(weights ** 2) / len(weights))


def write_empirical_artifact(path: Path, *, data_path: Path, neighbors: int, donor_split: str, data_sha256: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "empirical_knn",
        "neighbors": int(neighbors),
        "donor_split": str(donor_split),
        "data_path": str(data_path),
        "data_sha256": str(data_sha256),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_gas_generator(cfg: dict, data, feature_fn, device: str = "cpu"):
    """Load the frozen Gas generator named by config.

    ``generator_kind: empirical_knn`` is the paper-facing replacement for the
    failed VAEAC checkpoint.  The VAEAC path remains available for diagnostics.
    """
    kind = str(cfg.get("generator_kind") or "").strip().lower()
    checkpoint = cfg.get("generator_checkpoint") or cfg.get("checkpoint")
    if not kind and checkpoint and str(checkpoint).endswith(".json"):
        kind = "empirical_knn"
    if kind in {"empirical_knn", "empirical", "knn"}:
        neighbors = int(cfg.get("generator_neighbors", 60))
        if checkpoint and str(checkpoint).endswith(".json"):
            meta = json.loads(Path(checkpoint).read_text(encoding="utf-8"))
            neighbors = int(meta.get("neighbors", neighbors))
        return EmpiricalKNNGenerator(
            np.asarray(data["ref_train"], dtype=float),
            feature_fn=feature_fn,
            neighbors=neighbors,
        )
    from .vaeac import VAEACGenerator, load_vaeac_checkpoint

    model, payload = load_vaeac_checkpoint(str(checkpoint), device=device, expected_dim=128)
    return VAEACGenerator(
        model,
        payload.get("scale", np.ones(128)),
        alpha=float(payload.get("alpha", 0.0)),
        device=device,
        feature_fn=feature_fn,
    )
