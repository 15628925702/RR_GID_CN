"""Cached Q0 conditional bases for bulk scoring and information.

High-precision / rejection samplers stay in ``synthetic_oracle`` for
calibration. Bulk experiments reweight a frozen Q0 feature basis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
import os
from pathlib import Path
from typing import Any
from time import perf_counter

import numpy as np

from .rrgid import psd_project
from .synthetic_oracle import (
    _conditional_qmc_at_order,
    _conditional_parameters,
    _nested_qmc_normals,
    _record_workload,
    conditional_component_posterior_batch,
    feature_map,
    inverse_warp,
    log_partition,
    tilted_conditional_mean_exact,
    tilted_conditional_mean_qmc_nested,
    tilted_conditional_mean_qmc,
    warp,
)

FEATURE_DIM = 12
_CACHE_COUNTERS = {
    "feature_nodes_built": 0,
    "feature_nodes_reweighted": 0,
    "component_nodes": 0,
    "cache_hit": 0,
    "cache_miss": 0,
    "bytes_cached": 0,
    "gpu_cache_hit": 0,
    "gpu_cache_miss": 0,
}
_GPU_BASIS_CACHE: OrderedDict[int, tuple[Any, Any, int, Any]] = OrderedDict()
_GPU_BASIS_CACHE_BYTES = 0


def _cuda_for_basis():
    """Return a CUDA device for hot basis reductions when available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    return None


def reset_cache_counters() -> None:
    clear_gpu_basis_cache()
    for key in _CACHE_COUNTERS:
        _CACHE_COUNTERS[key] = 0


def cache_counters() -> dict[str, int]:
    return {key: int(value) for key, value in _CACHE_COUNTERS.items()}


def _bump(**values: int) -> None:
    payload = {key: int(value) for key, value in values.items()}
    for key, value in payload.items():
        if key in _CACHE_COUNTERS:
            _CACHE_COUNTERS[key] += value
    _record_workload(**payload)


def _gpu_cache_limit_bytes() -> int:
    value = os.environ.get("RR_GID_CN_GPU_CACHE_MB", "1024").strip()
    try:
        return max(0, int(float(value) * 1024 * 1024))
    except ValueError:
        return 1024 * 1024 * 1024


def clear_gpu_basis_cache() -> None:
    """Release panel basis tensors retained across scoring steps."""
    global _GPU_BASIS_CACHE_BYTES
    _GPU_BASIS_CACHE.clear()
    _GPU_BASIS_CACHE_BYTES = 0
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _cached_cuda_basis(basis: "ConditionalFeatureBasis", device):
    """Return a resident CUDA basis when it fits the bounded process cache."""
    global _GPU_BASIS_CACHE_BYTES
    key = id(basis)
    cached = _GPU_BASIS_CACHE.get(key)
    if cached is not None:
        if cached[3] is not basis:
            _stale = _GPU_BASIS_CACHE.pop(key, None)
            if _stale is not None:
                _GPU_BASIS_CACHE_BYTES -= int(_stale[2])
            cached = None
        else:
            _GPU_BASIS_CACHE.move_to_end(key)
            _CACHE_COUNTERS["gpu_cache_hit"] += 1
            return cached[0], cached[1]
    _CACHE_COUNTERS["gpu_cache_miss"] += 1
    if int(np.asarray(basis.phi_nodes).nbytes) > _gpu_cache_limit_bytes():
        return None
    try:
        import torch
        phi = torch.as_tensor(np.asarray(basis.phi_nodes), dtype=torch.float32, device=device)
        log_post = torch.as_tensor(
            np.asarray(basis.component_log_posterior), dtype=torch.float32, device=device,
        )[:, :, None]
        bytes_used = int(phi.numel() * phi.element_size() + log_post.numel() * log_post.element_size())
        limit = _gpu_cache_limit_bytes()
        while _GPU_BASIS_CACHE and _GPU_BASIS_CACHE_BYTES + bytes_used > limit:
            _key, (_old_phi, _old_log_post, old_bytes, _old_basis) = _GPU_BASIS_CACHE.popitem(last=False)
            _GPU_BASIS_CACHE_BYTES -= int(old_bytes)
        if _GPU_BASIS_CACHE_BYTES + bytes_used > limit:
            del phi, log_post
            return None
        # Retain the Python basis object alongside its id so an id cannot be
        # recycled while a stale CUDA tensor remains in the cache.
        _GPU_BASIS_CACHE[key] = (phi, log_post, bytes_used, basis)
        _GPU_BASIS_CACHE_BYTES += bytes_used
        return phi, log_post
    except Exception as exc:
        if "out of memory" in str(exc).lower():
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            return None
        raise


def _stable_logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    shift = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(shift, axis=axis) + np.log(np.exp(values - shift).sum(axis=axis))


def _hot_dtype(name: str) -> np.dtype:
    if name == "float16":
        # Storage-only compression.  Conditional reductions immediately cast
        # nodes to float64, so this does not change the accumulation dtype.
        return np.float16
    if name == "float32":
        return np.float32
    if name == "float64":
        return np.float64
    raise ValueError(f"unsupported hot dtype: {name}")


@dataclass
class ReferenceMomentCache:
    phi_reference: np.ndarray
    phi_reference_eval: np.ndarray | None = None
    A_beta_true: float | None = None
    mu_beta_true: np.ndarray | None = None
    F_beta_true: np.ndarray | None = None

    @classmethod
    def build(
        cls,
        reference: np.ndarray,
        scale: np.ndarray,
        *,
        beta_true: np.ndarray | None = None,
        feature_fn=None,
        dtype: str = "float64",
    ) -> "ReferenceMomentCache":
        fn = feature_fn or (lambda x: feature_map(x, scale))
        phi = np.asarray(fn(np.asarray(reference)), dtype=np.float64)
        cache = cls(phi_reference=phi, phi_reference_eval=phi)
        if beta_true is not None:
            cache.mu_beta_true, cache.F_beta_true = cache.moments(np.asarray(beta_true, dtype=float))
            logits = phi @ np.asarray(beta_true, dtype=float)
            cache.A_beta_true = float(_stable_logsumexp(logits, axis=0) - np.log(len(logits)))
        _bump(bytes_cached=int(phi.nbytes), cache_miss=1)
        return cache

    def moments(self, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        beta = np.asarray(beta, dtype=float)
        logits = self.phi_reference @ beta
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        mean = weights @ self.phi_reference
        centered = self.phi_reference - mean
        fisher = (centered * weights[:, None]).T @ centered
        return mean, fisher

    def log_partition(self, beta: np.ndarray) -> float:
        logits = self.phi_reference @ np.asarray(beta, dtype=float)
        return float(_stable_logsumexp(logits, axis=0) - np.log(len(logits)))

    def kl(self, beta_true: np.ndarray, beta_hat: np.ndarray) -> float:
        if self.A_beta_true is None or self.mu_beta_true is None:
            mu_true, _ = self.moments(beta_true)
            a_true = self.log_partition(beta_true)
        else:
            mu_true = self.mu_beta_true
            a_true = self.A_beta_true
        # KL(Q_{β*} || Q_{β̂}) = A(β̂) - A(β*) - μ* · (β̂ - β*)
        return float(
            self.log_partition(beta_hat) - a_true
            - mu_true @ (np.asarray(beta_hat) - np.asarray(beta_true))
        )


def build_full_law_moment_cache(
    mixture,
    scale,
    *,
    order: int = 10,
    scrambles: int = 2,
    seed: int = 0,
    feature_fn=None,
) -> ReferenceMomentCache:
    """Q0 feature nodes for GMM μ(β) via nested Sobol, then IS reweight.

    Score QMC approximates the GMM conditional. Centering those scores with IS
    on a frozen Monte Carlo reference leaves a constant U/B bias. This cache
    is the bulk stand-in for G3 ``oracle.mu``: build once, reweight every step.
    """
    from scipy.special import ndtri
    from scipy.stats import qmc

    if order < 6 or order > 16:
        raise ValueError("full-law QMC order must be in [6, 16]")
    if scrambles < 1:
        raise ValueError("full-law QMC needs at least one scramble")
    fn = feature_fn or (lambda x: feature_map(x, scale))
    chunks = []
    for scramble in range(int(scrambles)):
        for k, (mean, covariance) in enumerate(zip(mixture.means, mixture.covariances)):
            engine = qmc.Sobol(
                d=int(mixture.dimension),
                scramble=True,
                seed=int(seed) + 1_000_003 * scramble + 104729 * k,
            )
            uniform = np.clip(engine.random_base2(int(order)), 1e-12, 1.0 - 1e-12)
            latent = ndtri(uniform) @ np.linalg.cholesky(covariance).T + mean
            chunks.append(warp(latent, mixture.alpha))
    return ReferenceMomentCache.build(np.concatenate(chunks, axis=0), scale, feature_fn=fn)


@dataclass
class GroupedPanelData:
    panel_ids: tuple[tuple[int, ...], ...]
    offsets: np.ndarray
    observed_values: np.ndarray
    counts: np.ndarray

    @classmethod
    def from_observations(
        cls,
        observations: list[tuple[tuple[int, ...], np.ndarray]],
        *,
        width: int | None = None,
    ) -> "GroupedPanelData":
        grouped: dict[tuple[int, ...], list[np.ndarray]] = {}
        for panel, row in observations:
            grouped.setdefault(tuple(panel), []).append(np.asarray(row, dtype=float))
        panel_ids = tuple(grouped)
        blocks = [np.atleast_2d(np.asarray(grouped[panel], dtype=float)) for panel in panel_ids]
        if not blocks:
            return cls((), np.zeros(1, dtype=int), np.zeros((0, int(width or 0))), np.zeros(0, dtype=int))
        counts = np.asarray([len(block) for block in blocks], dtype=int)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        return cls(panel_ids, offsets, np.concatenate(blocks, axis=0), counts)

    def rows(self, panel: tuple[int, ...]) -> np.ndarray:
        index = self.panel_ids.index(tuple(panel))
        return self.observed_values[self.offsets[index] : self.offsets[index + 1]]

    def as_groups(self) -> dict[tuple[int, ...], np.ndarray]:
        return {panel: self.rows(panel) for panel in self.panel_ids}


@dataclass
class ConditionalFeatureBasis:
    panel: tuple[int, ...]
    observed_rows: np.ndarray
    component_log_posterior: np.ndarray
    phi_nodes: np.ndarray
    qmc_order: int
    seed: int
    dtype: str
    n_nodes: int = 0

    def __post_init__(self) -> None:
        self.n_nodes = int(self.phi_nodes.shape[2])

    def conditional_mean(self, beta: np.ndarray) -> np.ndarray:
        mean, _cov = self._reduce(beta, want_cov=False)
        return mean

    def conditional_mean_subset(self, beta: np.ndarray, row_indices: np.ndarray) -> np.ndarray:
        """Evaluate only a deterministic subset of observed rows.

        Shared score-basis runs build one basis for the union of rows used by
        all policies, then use this method to reduce only the rows belonging to
        the current policy.  The node construction and weighting formula are
        unchanged; this is an indexing/performance optimisation only.
        """
        mean, _cov = self._reduce(beta, want_cov=False, row_indices=row_indices)
        return mean

    def conditional_mean_and_cov(self, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._reduce(beta, want_cov=True)

    def _reduce(
        self,
        beta: np.ndarray,
        *,
        want_cov: bool,
        row_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        # Keep the hot reweighting path in the on-disk node dtype.  The
        # stored nodes are already calibrated at float32/float16 precision;
        # promoting every panel block to float64 triples memory traffic and
        # made a four-step B=2000 run take tens of minutes.  Accumulators and
        # the final conditional means remain float64, so only the bounded
        # exponential-weight contraction uses the hot dtype.
        compute_dtype = np.float64 if str(self.dtype) == "float64" else np.float32
        beta = np.asarray(beta, dtype=compute_dtype)
        device = _cuda_for_basis()
        if device is not None and compute_dtype == np.float32:
            reduced = self._reduce_cuda(
                beta, want_cov=want_cov, device=device, row_indices=row_indices,
            )
            if reduced is not None:
                return reduced
        if isinstance(self.phi_nodes, np.memmap):
            return self._reduce_streaming(
                beta, want_cov=want_cov, row_indices=row_indices,
            )
        compute_dtype = np.float64 if str(self.dtype) == "float64" else np.float32
        if row_indices is None:
            phi = self.phi_nodes.astype(compute_dtype, copy=False)
            log_posterior = self.component_log_posterior.astype(compute_dtype, copy=False)
        else:
            indices = np.asarray(row_indices, dtype=np.int64)
            phi = np.asarray(self.phi_nodes[indices], dtype=compute_dtype)
            log_posterior = np.asarray(self.component_log_posterior[indices], dtype=compute_dtype)
        n_rows, n_comp, n_nodes, rank = phi.shape
        logits = np.einsum("kcnr,r->kcn", phi, beta, optimize=True)
        logw = logits + log_posterior[:, :, None]
        logw = logw.reshape(n_rows, n_comp * n_nodes)
        shift = logw.max(axis=1, keepdims=True)
        weights = np.exp(logw - shift)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)
        flat = phi.reshape(n_rows, n_comp * n_nodes, rank)
        mean = np.einsum("kn,knr->kr", weights, flat, dtype=np.float64, optimize=True)
        _bump(feature_nodes_reweighted=n_rows * n_comp * n_nodes)
        if not want_cov:
            return mean.astype(np.float64, copy=False), None
        centered = flat - mean[:, None, :]
        cov = np.einsum("kn,knr,kns->krs", weights, centered, centered, dtype=np.float64, optimize=True)
        return mean.astype(np.float64, copy=False), cov.astype(np.float64, copy=False)

    def _reduce_cuda(
        self,
        beta: np.ndarray,
        *,
        want_cov: bool,
        device,
        row_indices: np.ndarray | None = None,
    ):
        """Reduce one in-memory basis on CUDA without changing the estimator.

        Bases are transferred panel-by-panel, so peak device memory is bounded
        by the current panel's observation block rather than the full budget.
        The information and score callers both consume float64 NumPy means;
        only the bounded node/log-weight contraction runs in float32.
        """
        try:
            import torch
            with torch.inference_mode():
                beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
                if row_indices is None:
                    indices = np.arange(len(self.component_log_posterior), dtype=np.int64)
                else:
                    indices = np.asarray(row_indices, dtype=np.int64)
                if indices.size == 0:
                    rank = int(self.phi_nodes.shape[-1])
                    empty = np.zeros((0, rank), dtype=np.float64)
                    return (empty, np.zeros((0, rank, rank), dtype=np.float64)) if want_cov else (empty, None)
                n_comp = int(self.phi_nodes.shape[1])
                n_nodes = int(self.phi_nodes.shape[2])
                rank = int(self.phi_nodes.shape[3])
                # Keep transfers bounded on the 6 GB laptop GPU.  The chunk
                # limit is intentionally conservative and works for both
                # ndarray and memmap-backed bases.
                max_rows = max(1, int(8_000_000 // max(1, n_comp * n_nodes)))
                means = []
                covariances = [] if want_cov else None
                for start in range(0, len(indices), max_rows):
                    chunk_indices = indices[start:start + max_rows]
                    phi = None
                    log_post = None
                    cached = _cached_cuda_basis(self, device)
                    if cached is not None:
                        cached_phi, cached_log_post = cached
                        if row_indices is None:
                            phi, log_post = cached_phi, cached_log_post
                        else:
                            idx_t = torch.as_tensor(chunk_indices, dtype=torch.long, device=device)
                            phi = cached_phi.index_select(0, idx_t)
                            log_post = cached_log_post.index_select(0, idx_t)
                    else:
                        phi = torch.as_tensor(
                            np.asarray(self.phi_nodes[chunk_indices]),
                            dtype=torch.float32, device=device,
                        )
                        log_post = torch.as_tensor(
                            np.asarray(self.component_log_posterior[chunk_indices]),
                            dtype=torch.float32, device=device,
                        )[:, :, None]
                    width = int(len(chunk_indices))
                    logits = torch.matmul(phi.reshape(-1, rank), beta_t).reshape(
                        width, n_comp, n_nodes,
                    ) + log_post
                    flat = phi.reshape(width, n_comp * n_nodes, rank)
                    logw = logits.reshape(width, n_comp * n_nodes)
                    weights = torch.softmax(logw, dim=1)
                    mean = torch.matmul(weights.unsqueeze(1), flat).squeeze(1)
                    means.append(mean.to(dtype=torch.float64).cpu().numpy())
                    _bump(feature_nodes_reweighted=width * n_comp * n_nodes)
                    if want_cov:
                        centered = flat - mean.unsqueeze(1)
                        cov = torch.einsum("kn,knr,kns->krs", weights, centered, centered)
                        assert covariances is not None
                        covariances.append(cov.to(dtype=torch.float64).cpu().numpy())
                    del phi, log_post, logits, flat, logw, weights, mean
                mean_np = np.concatenate(means, axis=0)
                if not want_cov:
                    return mean_np, None
                assert covariances is not None
                return mean_np, np.concatenate(covariances, axis=0)
        except Exception as exc:
            # A panel-sized OOM should fall back to the tested CPU path; do
            # not make a numerical run fail solely because another process
            # temporarily occupies GPU memory.
            if "out of memory" in str(exc).lower():
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return None
            raise

    def _reduce_streaming(
        self,
        beta: np.ndarray,
        *,
        want_cov: bool,
        row_chunk: int = 128,
        node_chunk: int = 4096,
        row_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Reduce a disk-backed basis without materialising it as float64.

        The online log-sum-exp update is algebraically identical to the
        in-memory reduction, while bounding the working set by one row/node
        block.  This is needed for formal-budget score bases whose stored
        float32 representation is larger than host RAM at calibrated orders.
        """
        if row_indices is None:
            selected = np.arange(self.phi_nodes.shape[0], dtype=np.int64)
        else:
            selected = np.asarray(row_indices, dtype=np.int64)
        n_rows = int(len(selected))
        n_comp = int(self.phi_nodes.shape[1])
        n_nodes = int(self.phi_nodes.shape[2])
        rank = int(self.phi_nodes.shape[3])
        compute_dtype = np.float64 if str(self.dtype) == "float64" else np.float32
        means = np.empty((n_rows, rank), dtype=np.float64)
        covariances = np.empty((n_rows, rank, rank), dtype=np.float64) if want_cov else None
        for row_start in range(0, n_rows, max(1, int(row_chunk))):
            row_stop = min(row_start + max(1, int(row_chunk)), n_rows)
            width = row_stop - row_start
            running_max = np.full(width, -np.inf, dtype=compute_dtype)
            denominator = np.zeros(width, dtype=np.float64)
            numerator = np.zeros((width, rank), dtype=np.float64)
            second = np.zeros((width, rank, rank), dtype=np.float64) if want_cov else None
            selected_rows = selected[row_start:row_stop]
            log_posterior = self.component_log_posterior[selected_rows, :, None]
            for node_start in range(0, n_nodes, max(1, int(node_chunk))):
                node_stop = min(node_start + max(1, int(node_chunk)), n_nodes)
                phi = np.asarray(
                    self.phi_nodes[selected_rows, :, node_start:node_stop, :],
                    dtype=compute_dtype,
                )
                # Flatten the feature coordinate before the matrix-vector
                # product.  ``einsum`` here creates a large contraction plan
                # for every row/node block and becomes the dominant cost when
                # all 120 panels are reduced from disk-backed bases.  The
                # equivalent GEMM path is both numerically identical (up to
                # normal BLAS rounding) and substantially faster.
                logits = (
                    phi.reshape(-1, rank) @ beta
                ).reshape(width, n_comp, node_stop - node_start) + log_posterior
                block_max = logits.max(axis=(1, 2))
                new_max = np.maximum(running_max, block_max)
                old_scale = np.zeros(width, dtype=compute_dtype)
                finite = np.isfinite(running_max)
                old_scale[finite] = np.exp(running_max[finite] - new_max[finite])
                weights = np.exp(logits - new_max[:, None, None])
                denominator = denominator * old_scale + weights.sum(axis=(1, 2))
                flat_phi = phi.reshape(width, -1, rank)
                flat_weights = weights.reshape(width, -1)
                block_numerator = np.matmul(
                    flat_weights[:, None, :], flat_phi,
                )[:, 0, :]
                numerator = numerator * old_scale[:, None] + block_numerator
                if want_cov:
                    assert second is not None
                    block_second = np.matmul(
                        (flat_phi * flat_weights[:, :, None]).transpose(0, 2, 1),
                        flat_phi,
                    )
                    second = second * old_scale[:, None, None] + block_second
                running_max = new_max
            safe = np.maximum(denominator, 1e-300)
            mean = numerator / safe[:, None]
            means[row_start:row_stop] = mean
            if want_cov:
                assert second is not None and covariances is not None
                covariance = second / safe[:, None, None] - np.einsum(
                    "kr,ks->krs", mean, mean, optimize=True,
                )
                covariances[row_start:row_stop] = 0.5 * (
                    covariance + np.swapaxes(covariance, 1, 2)
                )
        _bump(feature_nodes_reweighted=n_rows * n_comp * n_nodes)
        return means, covariances


def build_conditional_feature_basis(
    mixture,
    rows: np.ndarray,
    panel: tuple[int, ...],
    *,
    order: int,
    seed: int,
    scale: np.ndarray,
    feature_fn=None,
    dtype: str = "float32",
    feature_coords: int = FEATURE_DIM,
    node_chunk: int = 4096,
    row_chunk: int = 128,
    storage_path: str | Path | None = None,
    build_on_cuda: bool = False,
) -> ConditionalFeatureBasis:
    """Build one Q0 conditional feature basis. Does not call Gaussian sampling later."""
    if order < 4 or order > 18:
        raise ValueError("QMC order must be in [4, 18]")
    panel = tuple(int(i) for i in panel)
    rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    n_nodes = 1 << int(order)
    complement = tuple(i for i in range(mixture.dimension) if i not in panel)
    z_s = inverse_warp(rows, mixture.alpha)
    log_post = _component_log_posterior(mixture, rows, panel)
    posterior = np.exp(log_post - log_post.max(axis=1, keepdims=True))
    posterior /= posterior.sum(axis=1, keepdims=True)
    parameters = _conditional_parameters(mixture, panel)[1]
    normals = _nested_qmc_normals(n_nodes, len(complement), seed=int(seed))
    hot = _hot_dtype(dtype)
    n_comp = len(parameters)
    fn = feature_fn
    mapped = None
    shape = (len(rows), n_comp, n_nodes, int(feature_coords))
    if storage_path is not None:
        path = Path(storage_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mapped = np.memmap(
            path,
            mode="w+",
            dtype=hot,
            shape=shape,
        )
        phi_nodes = mapped
    else:
        phi_nodes = np.empty(shape, dtype=hot)
    # The conditional node construction is independent across panels and
    # components.  When the built-in synthetic feature map is active, keep
    # the fixed Sobol normals and conditional Gaussian transform identical but
    # execute the warp/feature tensor on CUDA.  This removes the dominant CPU
    # cost of constructing 120 panels at order 12/14 on the laptop GPU.
    use_cuda = bool(build_on_cuda) and fn is None and _cuda_for_basis() is not None and hot == np.float32
    cuda_device = _cuda_for_basis() if use_cuda else None
    torch = None
    if use_cuda:
        import torch as _torch
        torch = _torch
    observed = warp(z_s[:, None, :], mixture.alpha)
    row_step = max(1, int(row_chunk))
    node_step = max(1, int(node_chunk))
    if use_cuda:
        with torch.inference_mode():
            normals_t = torch.as_tensor(normals, dtype=torch.float32, device=cuda_device)
            scale_t = torch.as_tensor(
                np.asarray(scale)[: int(feature_coords)], dtype=torch.float32, device=cuda_device,
            )
            panel_feature = [int(i) for i in panel if int(i) < int(feature_coords)]
            panel_feature_pos = [int(i) for i, coord in enumerate(panel) if int(coord) < int(feature_coords)]
            target_missing_feature = [int(i) for i in range(int(feature_coords)) if i not in set(panel_feature)]
            target_missing_idx_t = [int(complement.index(i)) for i in target_missing_feature]
            for k, (mean_c, mean_s, _inv_ss, gain, _cond_cov, chol, _logdet) in enumerate(parameters):
                mean_c_t = torch.as_tensor(mean_c, dtype=torch.float32, device=cuda_device)
                mean_s_t = torch.as_tensor(mean_s, dtype=torch.float32, device=cuda_device)
                gain_t = torch.as_tensor(gain, dtype=torch.float32, device=cuda_device)
                chol_t = torch.as_tensor(chol, dtype=torch.float32, device=cuda_device)
                for row_start in range(0, len(rows), row_step):
                    row_stop = min(row_start + row_step, len(rows))
                    z_s_t = torch.as_tensor(
                        z_s[row_start:row_stop], dtype=torch.float32, device=cuda_device,
                    )
                    cond_mean_t = mean_c_t[None, :] + (z_s_t - mean_s_t) @ gain_t.T
                    for start in range(0, n_nodes, node_step):
                        stop = min(start + node_step, n_nodes)
                        node_normals = normals_t[start:stop]
                        z_missing_t = cond_mean_t[:, None, :] + node_normals[None, :, :] @ chol_t.T
                        z_feat = torch.zeros(
                            (row_stop - row_start, stop - start, int(feature_coords)),
                            dtype=torch.float32, device=cuda_device,
                        )
                        for pos, coord in zip(panel_feature_pos, panel_feature):
                            z_feat[:, :, coord] = z_s_t[:, pos].unsqueeze(1)
                        z_feat[:, :, target_missing_feature] = z_missing_t[:, :, target_missing_idx_t]
                        x_feat = (
                            torch.sinh(float(mixture.alpha) * z_feat) / float(mixture.alpha)
                            if mixture.alpha > 0 else z_feat
                        )
                        xt = x_feat / scale_t
                        phi_chunk = torch.cat(
                            (torch.tanh(xt[:, :, :6]), torch.tanh(xt[:, :, :6] * xt[:, :, 6:12])),
                            dim=-1,
                        )
                        phi_nodes[row_start:row_stop, k, start:stop, :] = (
                            phi_chunk.detach().cpu().numpy().astype(hot, copy=False)
                        )
                        del node_normals, z_missing_t, z_feat, x_feat, xt, phi_chunk
                    del z_s_t, cond_mean_t
                del mean_c_t, mean_s_t, gain_t, chol_t
            del normals_t, scale_t
    else:
        for k, (mean_c, mean_s, _inv_ss, gain, _cond_cov, chol, _logdet) in enumerate(parameters):
            # Process QMC nodes in bounded chunks.  Order-12 bases over all
            # panels otherwise create a large temporary tensor at once.
            cond_mean = mean_c[None, :] + (z_s - mean_s) @ gain.T
            for row_start in range(0, len(rows), row_step):
                row_stop = min(row_start + row_step, len(rows))
                for start in range(0, n_nodes, node_step):
                    stop = min(start + node_step, n_nodes)
                    x = np.zeros(
                        (row_stop - row_start, stop - start, mixture.dimension),
                        dtype=np.float64,
                    )
                    x[:, :, list(panel)] = observed[row_start:row_stop]
                    z_missing = (
                        cond_mean[row_start:row_stop, None, :]
                        + normals[None, start:stop, :] @ chol.T
                    )
                    x[:, :, list(complement)] = warp(z_missing, mixture.alpha)
                    if fn is None:
                        phi_chunk = feature_map(x.reshape(-1, mixture.dimension), scale)
                    else:
                        phi_chunk = np.asarray(fn(x.reshape(-1, mixture.dimension)), dtype=np.float64)
                    phi_chunk = phi_chunk.reshape(row_stop - row_start, stop - start, -1)
                    if phi_chunk.shape[-1] != int(feature_coords):
                        raise ValueError(
                            f"feature dimension {phi_chunk.shape[-1]} does not match feature_coords={feature_coords}"
                        )
                    phi_nodes[row_start:row_stop, k, start:stop, :] = phi_chunk.astype(
                        hot, copy=False,
                    )
    if mapped is not None:
        mapped.flush()
    _bump(
        feature_nodes_built=int(phi_nodes.size // max(phi_nodes.shape[-1], 1)),
        component_nodes=int(n_comp * n_nodes),
        cache_miss=1,
        bytes_cached=int(phi_nodes.nbytes + log_post.nbytes),
    )
    return ConditionalFeatureBasis(
        panel=panel,
        observed_rows=rows.astype(np.float64, copy=False),
        component_log_posterior=np.log(np.maximum(posterior, 1e-300)),
        phi_nodes=phi_nodes,
        qmc_order=int(order),
        seed=int(seed),
        dtype=str(dtype),
    )


def evaluate_conditional_basis(beta: np.ndarray, basis: ConditionalFeatureBasis) -> np.ndarray:
    return basis.conditional_mean(beta)


def _component_log_posterior(mixture, rows: np.ndarray, panel: tuple[int, ...]) -> np.ndarray:
    posterior = conditional_component_posterior_batch(mixture, rows, panel)
    return np.log(np.maximum(posterior, 1e-300))


@dataclass
class PanelInformationBasis:
    outer_phi: np.ndarray
    outer_panel_values: dict[tuple[int, ...], np.ndarray]
    basis_a: dict[tuple[int, ...], ConditionalFeatureBasis]
    basis_b: dict[tuple[int, ...], ConditionalFeatureBasis]
    outer_rows: int
    qmc_order: int
    seed: int

    def tilt_weights(self, beta: np.ndarray) -> np.ndarray:
        logits = self.outer_phi @ np.asarray(beta, dtype=np.float64)
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        return weights

    def information(self, beta: np.ndarray, panels: tuple[tuple[int, ...], ...] | None = None) -> dict[tuple[int, ...], np.ndarray]:
        """I_S(β) from frozen Q0 outer rows: IS-weighted cross-completion.

        The outer sample is Q0. Weights ``w_i ∝ exp(β·φ(x_i))`` turn the
        unweighted Q0 covariance into the Q_β covariance required by Fisher
        scoring. Uniform (unweighted) ``a.T @ b / (n-1)`` would estimate
        ``Cov_{Q0}(E_{Qβ}[φ|X_S])`` instead of ``I_S(Q_β)``.
        """
        weights = self.tilt_weights(beta)
        active = panels if panels is not None else tuple(self.basis_a)
        return {
            panel: weighted_cross_information(
                self.basis_a[panel].conditional_mean(beta),
                self.basis_b[panel].conditional_mean(beta),
                weights,
                self.outer_phi,
            )
            for panel in active
        }


@dataclass
class DirectPanelInformationBasis:
    """Cross-completion information using fixed QMC nodes on demand.

    This has the same outer rows, component posterior, two independent Sobol
    seeds, and ``weighted_cross_information`` contraction as
    ``PanelInformationBasis``.  It avoids storing 2 x 120 large tensors when
    the CUDA evaluator can reduce each panel directly in bounded chunks.
    """

    mixture: Any
    outer_phi: np.ndarray
    outer_panel_values: dict[tuple[int, ...], np.ndarray]
    panels: tuple[tuple[int, ...], ...]
    scale: np.ndarray
    outer_rows: int
    qmc_order: int
    seed: int
    feature_fn: Any = None
    row_chunk: int = 128

    def tilt_weights(self, beta: np.ndarray) -> np.ndarray:
        logits = self.outer_phi @ np.asarray(beta, dtype=np.float64)
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        return weights

    def _conditional_mean(self, beta: np.ndarray, panel: tuple[int, ...], seed: int) -> np.ndarray:
        rows = np.asarray(self.outer_panel_values[tuple(panel)], dtype=float)
        if len(rows) == 0:
            return np.zeros((0, len(np.asarray(beta))))
        chunks = []
        step = max(1, int(self.row_chunk))
        for start in range(0, len(rows), step):
            stop = min(start + step, len(rows))
            if _cuda_for_basis() is not None and self.feature_fn is None:
                estimate = tilted_conditional_mean_qmc_nested(
                    self.mixture,
                    beta,
                    rows[start:stop],
                    tuple(panel),
                    int(self.qmc_order),
                    seed=int(seed),
                    scale=self.scale,
                    feature_fn=None,
                    return_orders=(int(self.qmc_order),),
                )[int(self.qmc_order)]
            else:
                estimate = tilted_conditional_mean_qmc(
                    self.mixture,
                    beta,
                    rows[start:stop],
                    tuple(panel),
                    int(self.qmc_order),
                    seed=int(seed),
                    scale=self.scale,
                    feature_fn=self.feature_fn,
                )
            chunks.append(np.asarray(estimate, dtype=np.float64))
        return np.concatenate(chunks, axis=0)

    def information(
        self,
        beta: np.ndarray,
        panels: tuple[tuple[int, ...], ...] | None = None,
    ) -> dict[tuple[int, ...], np.ndarray]:
        weights = self.tilt_weights(beta)
        active = panels if panels is not None else self.panels
        out = {}
        for panel in active:
            panel = tuple(panel)
            a = self._conditional_mean(beta, panel, int(self.seed) + 1)
            b = self._conditional_mean(beta, panel, int(self.seed) + 2)
            out[panel] = weighted_cross_information(a, b, weights, self.outer_phi)
        return out


def build_direct_panel_information_basis(
    mixture,
    reference: np.ndarray,
    panels: tuple[tuple[int, ...], ...],
    *,
    scale: np.ndarray,
    outer_rows: int,
    order: int,
    seed: int,
    feature_fn=None,
    row_chunk: int = 128,
) -> DirectPanelInformationBasis:
    fn = feature_fn or (lambda x: feature_map(x, scale))
    rng = np.random.default_rng(int(seed))
    index = rng.choice(len(reference), size=int(outer_rows), replace=False)
    outer = np.asarray(reference)[index]
    outer_phi = np.asarray(fn(outer), dtype=np.float64)
    outer_panel_values = {tuple(panel): outer[:, list(panel)] for panel in panels}
    return DirectPanelInformationBasis(
        mixture=mixture,
        outer_phi=outer_phi,
        outer_panel_values=outer_panel_values,
        panels=tuple(tuple(panel) for panel in panels),
        scale=np.asarray(scale),
        outer_rows=int(outer_rows),
        qmc_order=int(order),
        seed=int(seed),
        feature_fn=feature_fn,
        row_chunk=max(1, int(row_chunk)),
    )


def build_panel_information_basis(
    mixture,
    reference: np.ndarray,
    panels: tuple[tuple[int, ...], ...],
    *,
    scale: np.ndarray,
    outer_rows: int,
    order: int,
    seed: int,
    feature_fn=None,
    dtype: str = "float32",
    scrambles: int = 2,
    storage_dir: str | Path | None = None,
    reuse_existing: bool = False,
) -> PanelInformationBasis:
    if int(scrambles) < 2:
        raise ValueError("cross-completion requires at least two independent bases")
    fn = feature_fn or (lambda x: feature_map(x, scale))
    rng = np.random.default_rng(int(seed))
    index = rng.choice(len(reference), size=int(outer_rows), replace=False)
    outer = np.asarray(reference)[index]
    outer_phi = np.asarray(fn(outer), dtype=np.float64)
    outer_panel_values = {tuple(panel): outer[:, list(panel)] for panel in panels}
    basis_a: dict[tuple[int, ...], ConditionalFeatureBasis] = {}
    basis_b: dict[tuple[int, ...], ConditionalFeatureBasis] = {}
    spill_root = Path(storage_dir) if storage_dir is not None else None
    if spill_root is not None:
        spill_root.mkdir(parents=True, exist_ok=True)

    shape = (int(outer_rows), len(mixture.weights), 1 << int(order), int(outer_phi.shape[1]))
    hot = _hot_dtype(dtype)
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * int(np.dtype(hot).itemsize)

    def _load_spill(panel: tuple[int, ...], label: str, basis_seed: int):
        if spill_root is None or not reuse_existing:
            return None
        token = "_".join(str(int(i)) for i in panel)
        path = spill_root / f"basis_{label}_{token}.dat"
        if not path.exists():
            raise FileNotFoundError(f"missing reusable information basis: {path}")
        actual_bytes = int(path.stat().st_size)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"invalid reusable information basis size for {path}: "
                f"expected {expected_bytes}, found {actual_bytes}"
            )
        mapped = np.memmap(path, mode="r", dtype=hot, shape=shape)
        observed = outer_panel_values[tuple(panel)]
        return ConditionalFeatureBasis(
            panel=tuple(panel),
            observed_rows=np.asarray(observed, dtype=np.float64),
            component_log_posterior=_component_log_posterior(mixture, observed, tuple(panel)),
            phi_nodes=mapped,
            qmc_order=int(order),
            seed=int(basis_seed),
            dtype=str(dtype),
        )

    def _spill(basis: ConditionalFeatureBasis, label: str) -> ConditionalFeatureBasis:
        if spill_root is None:
            return basis
        # Keep only one panel's construction temporaries resident.  The memmap
        # has the same dtype/shape, so downstream reductions are unchanged.
        token = "_".join(str(int(i)) for i in basis.panel)
        path = spill_root / f"basis_{label}_{token}.dat"
        mapped = np.memmap(path, mode="w+", dtype=basis.phi_nodes.dtype, shape=basis.phi_nodes.shape)
        mapped[...] = basis.phi_nodes
        mapped.flush()
        basis.phi_nodes = mapped
        return basis

    for panel in panels:
        observed = outer_panel_values[tuple(panel)]
        reused_a = _load_spill(tuple(panel), "a", int(seed) + 1)
        reused_b = _load_spill(tuple(panel), "b", int(seed) + 2)
        basis_a[tuple(panel)] = reused_a if reused_a is not None else _spill(
            build_conditional_feature_basis(
                mixture, observed, tuple(panel), order=order, seed=int(seed) + 1,
                scale=scale, feature_fn=feature_fn, dtype=dtype,
            ),
            "a",
        )
        basis_b[tuple(panel)] = reused_b if reused_b is not None else _spill(
            build_conditional_feature_basis(
                mixture, observed, tuple(panel), order=order, seed=int(seed) + 2,
                scale=scale, feature_fn=feature_fn, dtype=dtype,
            ),
            "b",
        )
    return PanelInformationBasis(
        outer_phi=outer_phi,
        outer_panel_values=outer_panel_values,
        basis_a=basis_a,
        basis_b=basis_b,
        outer_rows=int(outer_rows),
        qmc_order=int(order),
        seed=int(seed),
    )


def cholesky_solve(matrix: np.ndarray, rhs: np.ndarray, *, ridge: float = 1e-8) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve H x = u with Cholesky. Auditable ridge if the minimum eigenvalue is too small."""
    matrix = 0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T)
    rhs = np.asarray(rhs, dtype=np.float64)
    values = np.linalg.eigvalsh(matrix)
    lambda_min = float(values.min())
    used_ridge = 0.0
    work = matrix
    if lambda_min < float(ridge):
        used_ridge = float(ridge) - min(lambda_min, 0.0)
        work = matrix + used_ridge * np.eye(len(matrix))
    try:
        step = np.linalg.solve(work, rhs)
        factor = "cholesky_via_solve"
        np.linalg.cholesky(work)
        factor = "cholesky"
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError("H is not usable after auditable ridge") from exc
    diagnostics = {
        "lambda_min_H": lambda_min,
        "lambda_max_H": float(values.max()),
        "ridge": used_ridge,
        "linear_algebra": factor,
    }
    return step, diagnostics


def uncached_qmc_mean(mixture, beta, rows, panel, order, seed, scale, feature_fn=None) -> np.ndarray:
    """Reference path used by T1: existing QMC integrand, no cache."""
    return tilted_conditional_mean_qmc(
        mixture, beta, rows, panel, int(order), seed=int(seed), scale=scale, feature_fn=feature_fn,
    )


@dataclass
class FixedConditionalMean:
    """Fixed-order conditional score evaluated in bounded row blocks.

    This diagnostic path uses the same nested QMC nodes and seed as a cached
    basis, but avoids retaining the complete ``(rows, components, nodes,
    features)`` tensor. It separates QMC accuracy from basis storage cost.
    """

    mixture: Any
    rows: np.ndarray
    panel: tuple[int, ...]
    seed: int
    scale: np.ndarray
    feature_fn: Any = None
    order: int = 12
    row_chunk: int = 64
    scrambles: int = 1
    storage_path: str | Path | None = None
    # Direct callers expect the class to match a float64 reference basis. The
    # paper bulk path passes its explicit float32 hot dtype when memory matters.
    cache_dtype: str = "float64"
    build_on_cuda: bool = False
    last_converged: bool = True
    _basis: ConditionalFeatureBasis | None = field(default=None, init=False, repr=False)
    last_diagnostics: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def conditional_mean(self, beta: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(self.rows, dtype=float))
        if len(rows) == 0:
            return np.zeros((0, len(np.asarray(beta))))
        # Build the fixed QMC feature nodes once per observed row/panel and
        # only reweight them for subsequent Fisher-scoring steps.  The old
        # implementation regenerated the full conditional Sobol tensor on
        # every step, which made a four-step bulk run roughly four times
        # slower while evaluating the same mathematical approximation.
        if self._basis is None:
            self._basis = build_conditional_feature_basis(
                self.mixture,
                rows,
                self.panel,
                order=int(self.order),
                seed=int(self.seed),
                scale=self.scale,
                feature_fn=self.feature_fn,
                dtype=str(self.cache_dtype),
                storage_path=self.storage_path,
                row_chunk=max(1, int(self.row_chunk)),
                build_on_cuda=bool(self.build_on_cuda),
            )
        self.last_diagnostics = [{
            "cached_basis": True,
            "n_rows": int(len(rows)),
            "order": int(self.order),
            "scrambles": int(max(1, int(self.scrambles))),
        }]
        return self._basis.conditional_mean(beta)


@dataclass
class DirectConditionalMean:
    """Fixed-node CUDA score evaluation without materialising a row basis.

    The QMC node set and seed are identical to ``FixedConditionalMean``.  On
    CUDA this evaluates the same conditional ratio in bounded row chunks,
    avoiding a multi-gigabyte ``(rows, components, nodes, features)`` tensor.
    CPU/custom-feature callers fall back to the cached basis implementation.
    """

    mixture: Any
    rows: np.ndarray
    panel: tuple[int, ...]
    seed: int
    scale: np.ndarray
    feature_fn: Any = None
    order: int = 12
    row_chunk: int = 128
    cache_dtype: str = "float32"
    last_converged: bool = True
    _fallback: FixedConditionalMean | None = field(default=None, init=False, repr=False)
    last_diagnostics: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def conditional_mean(self, beta: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(self.rows, dtype=float))
        if len(rows) == 0:
            return np.zeros((0, len(np.asarray(beta))))
        if self.feature_fn is not None or _cuda_for_basis() is None:
            if self._fallback is None:
                self._fallback = FixedConditionalMean(
                    self.mixture, rows, self.panel, int(self.seed), self.scale,
                    self.feature_fn, order=int(self.order), row_chunk=int(self.row_chunk),
                    cache_dtype=str(self.cache_dtype),
                )
            value = self._fallback.conditional_mean(beta)
            self.last_diagnostics = [{"direct_cuda": False, **self._fallback.last_diagnostics[0]}]
            return value
        pieces = []
        step = max(1, int(self.row_chunk))
        for start in range(0, len(rows), step):
            stop = min(start + step, len(rows))
            result = tilted_conditional_mean_qmc_nested(
                self.mixture,
                np.asarray(beta, dtype=np.float64),
                rows[start:stop],
                tuple(self.panel),
                int(self.order),
                seed=int(self.seed),
                scale=self.scale,
                feature_fn=None,
                return_orders=(int(self.order),),
            )
            pieces.append(np.asarray(result[int(self.order)], dtype=np.float64))
        self.last_diagnostics = [{
            "direct_cuda": True,
            "n_rows": int(len(rows)),
            "row_chunk": int(step),
            "order": int(self.order),
        }]
        return np.concatenate(pieces, axis=0)


@dataclass
class IndexedConditionalMean:
    """View of a shared conditional basis for one policy's row ordering."""

    basis: ConditionalFeatureBasis
    row_indices: np.ndarray
    last_converged: bool = True
    last_diagnostics: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def conditional_mean(self, beta: np.ndarray) -> np.ndarray:
        indices = np.asarray(self.row_indices, dtype=np.int64)
        value = self.basis.conditional_mean_subset(beta, indices)
        self.last_diagnostics = [{
            "shared_basis": True,
            "n_rows": int(len(indices)),
            "basis_rows": int(self.basis.phi_nodes.shape[0]),
            "qmc_order": int(self.basis.qmc_order),
        }]
        return value


def build_shared_score_bases(
    mixture,
    grouped_by_policy: dict[str, GroupedPanelData],
    scale,
    *,
    order: int,
    seed: int,
    dtype: str = "float32",
    feature_fn=None,
    storage_dir: str | Path | None = None,
    build_on_cuda: bool = False,
    row_chunk: int = 128,
) -> tuple[dict[str, dict[tuple[int, ...], IndexedConditionalMean]], float, dict[str, Any]]:
    """Build one deduplicated basis per panel and expose policy-specific views.

    Policies share pilot observations, and occasionally share exact target
    rows.  Constructing those rows once avoids repeated conditional Gaussian
    transforms while preserving each policy's original row order through an
    integer index view.  The returned timing and counts are provenance fields,
    not part of the estimator.
    """
    t0 = perf_counter()
    union_rows: dict[tuple[int, ...], list[np.ndarray]] = {}
    union_keys: dict[tuple[int, ...], dict[bytes, int]] = {}
    policy_indices: dict[str, dict[tuple[int, ...], np.ndarray]] = {}
    for policy, grouped in grouped_by_policy.items():
        per_policy: dict[tuple[int, ...], np.ndarray] = {}
        for panel in grouped.panel_ids:
            panel = tuple(panel)
            bucket = union_rows.setdefault(panel, [])
            keys = union_keys.setdefault(panel, {})
            local: list[int] = []
            for row in grouped.rows(panel):
                row_arr = np.ascontiguousarray(np.asarray(row, dtype=np.float64))
                key = row_arr.tobytes()
                index = keys.get(key)
                if index is None:
                    index = len(bucket)
                    keys[key] = index
                    bucket.append(row_arr)
                local.append(int(index))
            per_policy[panel] = np.asarray(local, dtype=np.int64)
        policy_indices[str(policy)] = per_policy

    spill_root = Path(storage_dir) if storage_dir is not None else None
    if spill_root is not None:
        spill_root.mkdir(parents=True, exist_ok=True)
    basis_by_panel: dict[tuple[int, ...], ConditionalFeatureBasis] = {}
    for panel, rows in union_rows.items():
        token = "_".join(str(int(i)) for i in panel)
        basis_by_panel[panel] = build_conditional_feature_basis(
            mixture,
            np.asarray(rows, dtype=np.float64),
            panel,
            order=int(order),
            seed=int(seed),
            scale=scale,
            feature_fn=feature_fn,
            dtype=str(dtype),
            row_chunk=max(1, int(row_chunk)),
            storage_path=(spill_root / f"score_{token}.dat") if spill_root is not None else None,
            build_on_cuda=bool(build_on_cuda),
        )
    views: dict[str, dict[tuple[int, ...], IndexedConditionalMean]] = {}
    for policy, mapping in policy_indices.items():
        views[policy] = {
            panel: IndexedConditionalMean(basis_by_panel[panel], indices)
            for panel, indices in mapping.items()
        }
    metadata = {
        "shared": True,
        "policies": list(views),
        "union_rows_by_panel": {
            "_".join(str(int(i)) for i in panel): int(len(rows))
            for panel, rows in union_rows.items()
        },
        "union_rows_total": int(sum(len(rows) for rows in union_rows.values())),
    }
    return views, perf_counter() - t0, metadata


def build_shared_adaptive_score_bases(
    mixture,
    grouped_by_policy: dict[str, GroupedPanelData],
    scale,
    *,
    seed: int,
    feature_fn=None,
    start_order: int = 8,
    max_order: int = 16,
    atol: float = 2e-6,
    rtol: float = 2e-5,
    scrambles: int = 4,
    chunk_rows: int = 32,
) -> tuple[dict[str, dict[tuple[int, ...], IndexedAdaptiveConditionalMean]], float, dict[str, Any]]:
    """Build shared adaptive conditional queries for several allocations.

    The runner constructs policy observations from common target draws, so a
    panel's rows are usually prefixes shared by all policies.  Deduplicating
    those rows preserves the exact adaptive-QMC estimator while avoiding a
    full conditional integration pass for every policy.  ``AdaptiveConditionalMean``
    also caches each beta value, which makes the common pilot beta a single
    numerical evaluation across the entire policy set.
    """
    t0 = perf_counter()
    union_rows: dict[tuple[int, ...], list[np.ndarray]] = {}
    union_keys: dict[tuple[int, ...], dict[bytes, int]] = {}
    policy_indices: dict[str, dict[tuple[int, ...], np.ndarray]] = {}
    for policy, grouped in grouped_by_policy.items():
        per_policy: dict[tuple[int, ...], np.ndarray] = {}
        for panel in grouped.panel_ids:
            panel = tuple(panel)
            bucket = union_rows.setdefault(panel, [])
            keys = union_keys.setdefault(panel, {})
            local: list[int] = []
            for row in grouped.rows(panel):
                row_arr = np.ascontiguousarray(np.asarray(row, dtype=np.float64))
                key = row_arr.tobytes()
                index = keys.get(key)
                if index is None:
                    index = len(bucket)
                    keys[key] = index
                    bucket.append(row_arr)
                local.append(int(index))
            per_policy[panel] = np.asarray(local, dtype=np.int64)
        policy_indices[str(policy)] = per_policy

    basis_by_panel: dict[tuple[int, ...], AdaptiveConditionalMean] = {}
    for panel, rows in union_rows.items():
        basis_by_panel[panel] = AdaptiveConditionalMean(
            mixture,
            np.asarray(rows, dtype=np.float64),
            panel,
            int(seed),
            scale,
            feature_fn,
            start_order=int(start_order),
            max_order=int(max_order),
            atol=float(atol),
            rtol=float(rtol),
            scrambles=int(scrambles),
            chunk_rows=int(chunk_rows),
        )
    views: dict[str, dict[tuple[int, ...], IndexedAdaptiveConditionalMean]] = {}
    for policy, mapping in policy_indices.items():
        views[policy] = {
            panel: IndexedAdaptiveConditionalMean(basis_by_panel[panel], indices)
            for panel, indices in mapping.items()
        }
    metadata = {
        "shared": True,
        "backend": "exact_adaptive",
        "policies": list(views),
        "union_rows_by_panel": {
            "_".join(str(int(i)) for i in panel): int(len(rows))
            for panel, rows in union_rows.items()
        },
        "union_rows_total": int(sum(len(rows) for rows in union_rows.values())),
    }
    return views, perf_counter() - t0, metadata


def weighted_cross_information(mean_a: np.ndarray, mean_b: np.ndarray, weights: np.ndarray, outer_phi: np.ndarray) -> np.ndarray:
    """IS-weighted cross-completion I_S from two conditional-mean replicates."""
    weights = np.asarray(weights, dtype=np.float64)
    mu = weights @ np.asarray(outer_phi, dtype=np.float64)
    a = np.asarray(mean_a, dtype=np.float64) - mu
    b = np.asarray(mean_b, dtype=np.float64) - mu
    a = a - weights @ a
    b = b - weights @ b
    mass = 1.0 - float(np.dot(weights, weights))
    info = (a.T @ (weights[:, None] * b) + b.T @ (weights[:, None] * a)) / max(2.0 * mass, 1e-12)
    return psd_project(info)


@dataclass
class AdaptiveConditionalMean:
    """Calibration gold: adaptive QMC conditional mean with the same ``conditional_mean`` API."""

    mixture: Any
    rows: np.ndarray
    panel: tuple[int, ...]
    seed: int
    scale: np.ndarray
    feature_fn: Any = None
    start_order: int = 8
    max_order: int = 16
    atol: float = 2e-6
    rtol: float = 2e-5
    scrambles: int = 4
    chunk_rows: int = 32
    last_converged: bool = True
    last_diagnostics: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _result_cache: OrderedDict[bytes, tuple[np.ndarray, bool, list[dict[str, Any]]]] = field(
        default_factory=OrderedDict, init=False, repr=False,
    )

    def _cache_key(self, beta: np.ndarray) -> bytes:
        # The beta vectors are produced by deterministic floating-point
        # operations within one replication.  Using their contiguous bytes
        # keeps the cache exact and avoids an arbitrary rounding tolerance.
        return np.ascontiguousarray(np.asarray(beta, dtype=np.float64)).tobytes()

    def conditional_mean(self, beta: np.ndarray) -> np.ndarray:
        key = self._cache_key(beta)
        cached = self._result_cache.get(key)
        if cached is not None:
            value, converged, diagnostics = cached
            self._result_cache.move_to_end(key)
            self.last_converged = bool(converged)
            self.last_diagnostics = [dict(item) for item in diagnostics]
            return value.copy()
        rows = np.atleast_2d(np.asarray(self.rows, dtype=float))
        if len(rows) == 0:
            rank = int(np.asarray(beta).size)
            value = np.zeros((0, rank), dtype=np.float64)
            self.last_converged = True
            self.last_diagnostics = []
            self._result_cache[key] = (value, True, [])
            return value.copy()
        chunks = []
        converged = True
        summaries: list[dict[str, Any]] = []
        for start in range(0, len(rows), int(self.chunk_rows)):
            stop = min(start + int(self.chunk_rows), len(rows))
            estimate, diagnostics = tilted_conditional_mean_exact(
                self.mixture, beta, rows[start:stop], self.panel,
                seed=int(self.seed), scale=self.scale, feature_fn=self.feature_fn,
                start_order=int(self.start_order), max_order=int(self.max_order),
                atol=float(self.atol), rtol=float(self.rtol), scrambles=int(self.scrambles),
                return_diagnostics=True,
            )
            chunks.append(estimate)
            converged = converged and bool(diagnostics.get("converged"))
            row_orders = np.asarray(diagnostics.get("row_final_order", []), dtype=int)
            unique_orders, order_counts = (
                np.unique(row_orders, return_counts=True)
                if row_orders.size
                else (np.asarray([], dtype=int), np.asarray([], dtype=int))
            )
            summaries.append({
                "chunk_start": int(start),
                "chunk_stop": int(stop),
                "n_rows": int(stop - start),
                "converged": bool(diagnostics.get("converged")),
                "n_unconverged": int(diagnostics.get("n_unconverged", 0)),
                "final_order": int(diagnostics.get("final_order", self.max_order)),
                "row_final_order_histogram": {
                    str(int(order)): int(count)
                    for order, count in zip(unique_orders, order_counts)
                },
                "max_abs_delta": diagnostics.get("max_abs_delta"),
                "max_rel_delta": diagnostics.get("max_rel_delta"),
                "scramble_se": diagnostics.get("scramble_se"),
                "n_active_by_order": diagnostics.get("n_active_by_order", {}),
            })
        self.last_converged = bool(converged)
        self.last_diagnostics = summaries
        value = np.concatenate(chunks, axis=0).astype(np.float64, copy=False)
        # A replication evaluates only a handful of beta values (one per
        # scoring step and policy).  Retaining these compact means avoids
        # recomputing the shared panel rows when policies use the same beta.
        self._result_cache[key] = (value.copy(), bool(converged), [dict(item) for item in summaries])
        self._result_cache.move_to_end(key)
        while len(self._result_cache) > 8:
            self._result_cache.popitem(last=False)
        return value


@dataclass
class IndexedAdaptiveConditionalMean:
    """Policy-specific indexed view over shared adaptive conditional rows."""

    basis: AdaptiveConditionalMean
    row_indices: np.ndarray
    last_converged: bool = True
    last_diagnostics: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def conditional_mean(self, beta: np.ndarray) -> np.ndarray:
        indices = np.asarray(self.row_indices, dtype=np.int64)
        value = self.basis.conditional_mean(beta)
        self.last_converged = bool(self.basis.last_converged)
        self.last_diagnostics = [{
            "shared_adaptive_basis": True,
            "n_rows": int(len(indices)),
            "basis_rows": int(len(self.basis.rows)),
            "start_order": int(self.basis.start_order),
            "max_order": int(self.basis.max_order),
            "basis_diagnostics": [dict(item) for item in self.basis.last_diagnostics],
        }]
        return value[indices]


def gold_information_from_basis(
    basis: PanelInformationBasis,
    mixture,
    beta: np.ndarray,
    scale: np.ndarray,
    panels: tuple[tuple[int, ...], ...] | None = None,
    *,
    feature_fn=None,
    start_order: int = 8,
    max_order: int = 16,
    atol: float = 2e-6,
    rtol: float = 2e-5,
    scrambles: int = 4,
    chunk_rows: int = 32,
    adaptive_cache: dict[tuple[tuple[int, ...], str], AdaptiveConditionalMean] | None = None,
    return_diagnostics: bool = False,
):
    """Adaptive inner means on the *same* frozen outer rows as ``basis``."""
    weights = basis.tilt_weights(beta)
    active = panels if panels is not None else tuple(basis.basis_a)
    out: dict[tuple[int, ...], np.ndarray] = {}
    converged = True
    panel_diagnostics: list[dict[str, Any]] = []
    for panel in active:
        observed = basis.outer_panel_values[tuple(panel)]
        if adaptive_cache is None:
            adaptive_cache = {}
        key_a = (tuple(panel), "a")
        key_b = (tuple(panel), "b")
        mean_a = adaptive_cache.get(key_a)
        if mean_a is None:
            mean_a = AdaptiveConditionalMean(
                mixture, observed, tuple(panel), int(basis.basis_a[panel].seed), scale, feature_fn,
                start_order=start_order, max_order=max_order, atol=atol, rtol=rtol,
                scrambles=scrambles, chunk_rows=chunk_rows,
            )
            adaptive_cache[key_a] = mean_a
        mean_b = adaptive_cache.get(key_b)
        if mean_b is None:
            mean_b = AdaptiveConditionalMean(
                mixture, observed, tuple(panel), int(basis.basis_b[panel].seed), scale, feature_fn,
                start_order=start_order, max_order=max_order, atol=atol, rtol=rtol,
                scrambles=scrambles, chunk_rows=chunk_rows,
            )
            adaptive_cache[key_b] = mean_b
        a = mean_a.conditional_mean(beta)
        b = mean_b.conditional_mean(beta)
        panel_ok = bool(mean_a.last_converged and mean_b.last_converged)
        converged = converged and panel_ok
        panel_diagnostics.append({
            "panel": [int(v) for v in panel],
            "converged": panel_ok,
            "basis_a": mean_a.last_diagnostics,
            "basis_b": mean_b.last_diagnostics,
        })
        out[tuple(panel)] = weighted_cross_information(a, b, weights, basis.outer_phi)
    if return_diagnostics:
        return out, bool(converged), panel_diagnostics
    return out, bool(converged)
