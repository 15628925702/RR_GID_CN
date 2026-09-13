"""Exact synthetic reference oracle from the frozen RR-GID_CN specification."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from collections import OrderedDict
import gc
import math
import os

import numpy as np
try:
    from scipy.special import ndtri
    from scipy.stats import qmc
except Exception:  # pragma: no cover - fallback is retained for minimal installs
    ndtri = None
    qmc = None
try:
    import torch
except Exception:  # pragma: no cover
    torch = None

_CONDITIONAL_PARAMETER_CACHE = {}
# Per-process CUDA cache for panel-specific conditional-Gaussian tensors.  The
# frozen mixture has only 120 two-coordinate panels, so this cache is bounded
# by the panel library and avoids repeated host-to-device conversion during
# active-panel Fisher updates.  It deliberately excludes beta, observations,
# and QMC nodes, which remain call-specific and seed-controlled.
_TORCH_PANEL_TENSOR_CACHE = {}
_TORCH_REJECTION_PANEL_CACHE = {}
_TORCH_QMC_NORMAL_CACHE = OrderedDict()
_TORCH_QMC_NORMAL_CACHE_LIMIT = 16
# Bound rows * 2^order so a (rows, n, 12) float64 tensor stays near 0.4 GiB.
# Concatenating row chunks is algebraically identical to one unchunked integral.
_CUDA_QMC_ROW_NODE_BUDGET = 4_000_000


def _release_cuda_workspace(*, drop_qmc_cache: bool = False) -> None:
    gc.collect()
    if drop_qmc_cache:
        _TORCH_QMC_NORMAL_CACHE.clear()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cuda_qmc_max_rows(order: int) -> int:
    return max(1, _CUDA_QMC_ROW_NODE_BUDGET // (1 << int(order)))


_WORKLOAD_COUNTERS = {
    "full_rejection_calls": 0,
    "full_proposals": 0,
    "full_accepted_raw": 0,
    "full_requested": 0,
    "conditional_rejection_calls": 0,
    "conditional_proposals": 0,
    "conditional_accepted_raw": 0,
    "conditional_requested": 0,
    "qmc_calls": 0,
    "qmc_nodes": 0,
    "qmc_scrambles": 0,
    "feature_scans": 0,
    "feature_nodes_built": 0,
    "feature_nodes_reweighted": 0,
    "component_nodes": 0,
    "cache_hit": 0,
    "cache_miss": 0,
    "bytes_cached": 0,
}
_CONDITIONAL_ACCEPT_RATES: list[float] = []
_QMC_ORDERS: list[int] = []


def reset_workload_counters() -> None:
    """Reset process-local sampler work counters at a replication boundary."""
    for key in _WORKLOAD_COUNTERS:
        _WORKLOAD_COUNTERS[key] = 0
    _CONDITIONAL_ACCEPT_RATES.clear()
    _QMC_ORDERS.clear()


def workload_counters() -> dict[str, int]:
    """Return an immutable snapshot of process-local sampler work."""
    return {key: int(value) for key, value in _WORKLOAD_COUNTERS.items()}


def workload_rate_summary() -> dict[str, float | int | None]:
    """Per-call conditional rejection acceptance rates for a replication."""
    rates = list(_CONDITIONAL_ACCEPT_RATES)
    if not rates:
        return {"min": None, "median": None, "max": None, "n": 0}
    ordered = sorted(rates)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return {
        "min": float(ordered[0]),
        "median": float(median),
        "max": float(ordered[-1]),
        "n": n,
    }


def workload_qmc_orders() -> list[int]:
    return list(_QMC_ORDERS)


def _record_workload(**values: int) -> None:
    for key, value in values.items():
        _WORKLOAD_COUNTERS[key] += int(value)
    proposed = int(values.get("conditional_proposals", 0))
    accepted = int(values.get("conditional_accepted_raw", 0))
    if proposed > 0:
        _CONDITIONAL_ACCEPT_RATES.append(accepted / proposed)


def _rejection_round_budget(
    *,
    round_index: int,
    remaining_samples: int,
    proposals_total: int,
    accepted_raw_total: int,
    min_rounds: int = 20000,
    safety: float = 8.0,
    hard_cap: int = 200_000,
) -> int:
    """Extend the 20000-round floor using observed envelope acceptance.

    Far-pilot ``||beta||`` makes bounded-tilt acceptance ~1e-6.  The rejection
    law is unchanged: only the implementation stop is allowed to grow with the
    remaining work.  A hard cap still fails loudly rather than hang for days.
    """
    if remaining_samples <= 0:
        return int(round_index)
    p = float(accepted_raw_total) / max(int(proposals_total), 1)
    avg = float(proposals_total) / max(int(round_index), 1)
    if p <= 0.0 or avg <= 0.0:
        return int(min_rounds if round_index < min_rounds else round_index)
    expected = float(remaining_samples) / (p * avg)
    return int(min(hard_cap, max(min_rounds, round_index + math.ceil(safety * expected))))


def _cuda_device():
    """Return the explicitly assigned CUDA device for reproducible sharding."""
    if torch is None or not torch.cuda.is_available():
        return None
    value = os.environ.get("RR_GID_CN_CUDA_DEVICE", "0").strip()
    if value.startswith("cuda:"):
        return torch.device(value)
    if not value.isdigit():
        raise ValueError("RR_GID_CN_CUDA_DEVICE must be an integer or cuda:<integer>")
    index = int(value)
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {index} unavailable (count={torch.cuda.device_count()})")
    return torch.device(f"cuda:{index}")


def _conditional_parameters(mixture: FrozenMixture, panel: tuple[int, ...]):
    key = (id(mixture), tuple(panel))
    cached = _CONDITIONAL_PARAMETER_CACHE.get(key)
    if cached is not None:
        return cached
    complement = tuple(i for i in range(mixture.dimension) if i not in panel)
    params = []
    for mean, cov in zip(mixture.means, mixture.covariances):
        sigma_ss = cov[np.ix_(panel, panel)]
        inv_ss = np.linalg.inv(sigma_ss)
        gain = cov[np.ix_(complement, panel)] @ inv_ss
        cond_cov = cov[np.ix_(complement, complement)] - gain @ cov[np.ix_(panel, complement)]
        sign, logdet = np.linalg.slogdet(sigma_ss)
        if sign <= 0:
            raise ValueError("mixture covariance is not positive definite")
        chol = np.linalg.cholesky(cond_cov)
        params.append((mean[list(complement)], mean[list(panel)], inv_ss, gain, cond_cov, chol, logdet))
    _CONDITIONAL_PARAMETER_CACHE[key] = (complement, tuple(params))
    return _CONDITIONAL_PARAMETER_CACHE[key]


@dataclass(frozen=True)
class FrozenMixture:
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    alpha: float

    @property
    def dimension(self) -> int:
        return int(self.means.shape[1])


def make_frozen_mixture(seed: int = 2026, dimension: int = 16, components: int = 4, alpha: float = 1.0) -> FrozenMixture:
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, 1.5, size=(components, dimension))
    covariances = np.empty((components, dimension, dimension))
    for k in range(components):
        L = rng.normal(0.0, 0.25, size=(dimension, 4))
        D = np.diag(rng.uniform(0.4, 0.8, size=dimension))
        covariances[k] = D + L @ L.T
    return FrozenMixture(np.full(components, 1.0 / components), means, covariances, float(alpha))


def warp(z: np.ndarray, alpha: float) -> np.ndarray:
    z = np.asarray(z)
    return np.sinh(alpha * z) / alpha if alpha > 0 else z.copy()


def inverse_warp(x: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(x)
    return np.arcsinh(alpha * x) / alpha if alpha > 0 else x.copy()


def all_pairs(dimension: int = 16) -> tuple[tuple[int, int], ...]:
    return tuple(combinations(range(dimension), 2))


def sample_full(mixture: FrozenMixture, n: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    components = rng.choice(len(mixture.weights), size=n, p=mixture.weights)
    z = np.empty((n, mixture.dimension))
    for k in range(len(mixture.weights)):
        idx = np.flatnonzero(components == k)
        if idx.size:
            z[idx] = rng.multivariate_normal(mixture.means[k], mixture.covariances[k], size=idx.size)
    return warp(z, mixture.alpha)


def conditional_component_posterior(mixture: FrozenMixture, x_s: np.ndarray, panel: tuple[int, ...]) -> np.ndarray:
    """Posterior component probabilities given warped observations on ``panel``."""
    panel = tuple(panel)
    z_s = inverse_warp(np.asarray(x_s), mixture.alpha)
    logs = []
    for weight, mean, cov in zip(mixture.weights, mixture.means, mixture.covariances):
        mu = mean[list(panel)]
        sigma = cov[np.ix_(panel, panel)]
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            raise ValueError("mixture covariance is not positive definite")
        delta = z_s - mu
        logs.append(np.log(weight) - 0.5 * (len(panel) * np.log(2 * np.pi) + logdet + delta @ np.linalg.solve(sigma, delta)))
    logs = np.asarray(logs)
    probs = np.exp(logs - logs.max())
    return probs / probs.sum()


def conditional_component_posterior_batch(
    mixture: FrozenMixture, x_s: np.ndarray, panel: tuple[int, ...]
) -> np.ndarray:
    """Vectorized component posterior for many observed panel rows.

    This is algebraically identical to ``conditional_component_posterior``;
    batching only removes the Python loop over rows in the CUDA conditional
    samplers.  Keeping the calculation in float64 preserves the formal
    reference implementation's numerical path.
    """
    panel = tuple(panel)
    observed = np.atleast_2d(np.asarray(x_s, dtype=float))
    if observed.shape[1] != len(panel):
        raise ValueError("observed batch has incompatible panel width")
    z_s = inverse_warp(observed, mixture.alpha)
    _, parameters = _conditional_parameters(mixture, panel)
    log_probs = np.empty((len(observed), len(mixture.weights)), dtype=float)
    for k, (mean_c, mean_s, inv_ss, _gain, _cond_cov, _chol, logdet) in enumerate(parameters):
        delta = z_s - mean_s
        log_probs[:, k] = np.log(mixture.weights[k]) - 0.5 * (
            len(panel) * np.log(2.0 * np.pi)
            + logdet
            + np.einsum("ni,ij,nj->n", delta, inv_ss, delta)
        )
    log_probs -= log_probs.max(axis=1, keepdims=True)
    probs = np.exp(log_probs)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def sample_conditional(mixture: FrozenMixture, x_s: np.ndarray, panel: tuple[int, ...], n: int, seed: int | None = None) -> np.ndarray:
    panel = tuple(panel)
    if len(panel) == mixture.dimension:
        return np.repeat(np.asarray(x_s)[None, :], n, axis=0)
    rng = np.random.default_rng(seed)
    complement = tuple(i for i in range(mixture.dimension) if i not in panel)
    z_s = inverse_warp(np.asarray(x_s), mixture.alpha)
    posterior = conditional_component_posterior(mixture, x_s, panel)
    components = rng.choice(len(posterior), size=n, p=posterior)
    z = np.empty((n, mixture.dimension))
    z[:, list(panel)] = z_s
    for k in range(len(posterior)):
        idx = np.flatnonzero(components == k)
        if not idx.size:
            continue
        cov = mixture.covariances[k]
        ss = np.ix_(panel, panel)
        cc = np.ix_(complement, complement)
        cs = np.ix_(complement, panel)
        cond_mean = mixture.means[k][list(complement)] + cov[cs] @ np.linalg.solve(cov[ss], z_s - mixture.means[k][list(panel)])
        cond_cov = cov[cc] - cov[cs] @ np.linalg.solve(cov[ss], cov[np.ix_(panel, complement)])
        z[np.ix_(idx, complement)] = rng.multivariate_normal(cond_mean, cond_cov, size=idx.size)
    return warp(z, mixture.alpha)


def sample_conditional_batch(
    mixture: FrozenMixture, x_s: np.ndarray, panel: tuple[int, ...], n: int, seed: int | None = None
    , _parameters=None
) -> np.ndarray:
    """Independent exact GMM conditional completions for a batch of panels.

    The returned array has shape ``(n_rows, n, dimension)``.  This is a
    vectorized implementation of :func:`sample_conditional`, used only to make
    the formal-budget S1 estimator computationally feasible.
    """
    panel = tuple(panel)
    observed = np.atleast_2d(np.asarray(x_s, dtype=float))
    if observed.shape[1] != len(panel):
        raise ValueError("observed batch has incompatible panel width")
    rng = np.random.default_rng(seed)
    rows = observed.shape[0]
    complement = tuple(i for i in range(mixture.dimension) if i not in panel)
    z_s = inverse_warp(observed, mixture.alpha)
    log_probs = np.empty((rows, len(mixture.weights)))
    if _parameters is None:
        complement, parameters = _conditional_parameters(mixture, panel)
    else:
        parameters = _parameters
    for k, (weight, mean, cov) in enumerate(zip(mixture.weights, mixture.means, mixture.covariances)):
        _mean_c, mean_s, inv_ss, _gain, _cond_cov, _chol, logdet = parameters[k]
        delta = z_s - mean[list(panel)]
        log_probs[:, k] = np.log(weight) - 0.5 * (
            len(panel) * np.log(2 * np.pi) + logdet + np.einsum("ni,ij,nj->n", delta, inv_ss, delta)
        )
    probs = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    uniforms = rng.random((rows, n))
    components = (uniforms[:, :, None] > np.cumsum(probs, axis=1)[:, None, :]).sum(axis=2)
    z = np.empty((rows, n, mixture.dimension))
    z[:, :, list(panel)] = z_s[:, None, :]
    for k, (mean_c, mean_s, _inv_ss, gain, _cond_cov, chol, _logdet) in enumerate(parameters):
        row_idx, sample_idx = np.nonzero(components == k)
        if row_idx.size:
            conditional_mean = mean_c + (z_s[row_idx] - mean_s) @ gain.T
            draws = rng.standard_normal((row_idx.size, len(complement))) @ chol.T + conditional_mean
            for j, coordinate in enumerate(complement):
                z[row_idx, sample_idx, coordinate] = draws[:, j]
    return warp(z, mixture.alpha)


def tilted_conditional_sample(
    mixture: FrozenMixture,
    beta: np.ndarray,
    x_s: np.ndarray,
    panel: tuple[int, ...],
    n: int,
    seed: int | None = None,
    scale: np.ndarray | None = None,
    feature_fn=None,
) -> np.ndarray:
    """Exact rejection samples from the relative tilt conditional law.

    Since every feature coordinate is bounded by one, ``sum(abs(beta))`` is a
    valid log-tilt envelope. Rejection from the exact GMM conditional therefore
    has no self-normalized importance bias.
    """
    rng = np.random.default_rng(seed)
    fn = (lambda x: feature_map(x, scale)) if feature_fn is None else feature_fn
    beta = np.asarray(beta, dtype=float)
    panel_set = set(panel)
    fixed = np.zeros(beta.shape[0], dtype=bool)
    fixed[:6] = [i in panel_set for i in range(6)]
    fixed[6:] = [i in panel_set and i + 6 in panel_set for i in range(6)]
    observed_full = np.zeros(mixture.dimension)
    observed_full[list(panel)] = np.asarray(x_s)
    observed_features = fn(observed_full[None, :])[0]
    envelope = float(np.dot(beta[fixed], observed_features[fixed]) + np.sum(np.abs(beta[~fixed])))
    accepted = []
    while len(accepted) < n:
        proposal = sample_conditional(mixture, x_s, panel, max(32, 2 * (n - len(accepted))), int(rng.integers(2**31 - 1)))
        logits = fn(proposal) @ beta
        keep = rng.random(len(proposal)) < np.exp(logits - envelope)
        accepted.extend(proposal[keep])
    return np.asarray(accepted[:n])


def tilted_conditional_batch(
    mixture: FrozenMixture,
    beta: np.ndarray,
    x_s: np.ndarray,
    panel: tuple[int, ...],
    n: int,
    seed: int | None = None,
    scale: np.ndarray | None = None,
    feature_fn=None,
) -> np.ndarray:
    if _cuda_device() is not None and feature_fn is None:
        return _tilted_conditional_batch_torch(
            mixture, beta, x_s, panel, n, seed, scale,
        )
    rng = np.random.default_rng(seed)
    fn = (lambda x: feature_map(x, scale)) if feature_fn is None else feature_fn
    rows = np.atleast_2d(np.asarray(x_s))
    out = np.empty((len(rows), n, mixture.dimension))
    panel_set = set(panel)
    fixed = np.zeros(beta.shape[0], dtype=bool)
    fixed[:6] = [i in panel_set for i in range(6)]
    fixed[6:] = [i in panel_set and i + 6 in panel_set for i in range(6)]
    full = np.zeros((len(rows), mixture.dimension)); full[:, list(panel)] = rows
    fixed_features = fn(full)[:, fixed] @ beta[fixed]
    envelope = fixed_features + np.sum(np.abs(beta[~fixed]))
    filled = np.zeros(len(rows), dtype=int)
    rounds = 0
    attempted = 0
    accepted_raw = 0
    batch_floor = int(max(8, 2 * n))
    # Bound temporary proposal memory when many rows are active.  The cap is
    # on the whole call (roughly 32 MiB at float64, dimension 16), not per row.
    # Large formal budgets have hundreds of observed rows per panel.  A
    # 4,194,304-proposal cap (~512 MiB at float64, d=16) keeps exact
    # rejection vectorized even when acceptance is near the declared
    # low-overlap boundary, without changing the proposal law.
    max_batch_total = 4_194_304
    round_budget = 20000
    while np.any(filled < n):
        rounds += 1
        if rounds > round_budget:
            remaining_samples = int(np.maximum(n - filled, 0).sum())
            round_budget = _rejection_round_budget(
                round_index=rounds - 1,
                remaining_samples=remaining_samples,
                proposals_total=attempted,
                accepted_raw_total=accepted_raw,
            )
            if rounds > round_budget:
                acceptance = filled / max(attempted, 1)
                raise RuntimeError(
                    "conditional exact tilt did not reach requested samples; "
                    f"panel={panel}, rounds={rounds - 1}, acceptance={acceptance.tolist()}"
                )
        need = np.maximum(n - filled, 0)
        active = np.flatnonzero(need)
        # Generate a batch directly for each active observation.  The previous
        # implementation repeated every row proposal_n times and then called
        # sample_conditional_batch(..., n=1), rebuilding the same conditional
        # Gaussian parameters for every proposal.  Sampling n=proposal_n in one
        # call is mathematically identical (independent Q0 proposals) and keeps
        # exact rejection sampling while avoiding that O(rows * proposal_n)
        # parameter/setup overhead.  A modest batch size is sufficient because
        # the envelope acceptance is typically 0.2--0.4; rows are retried until
        # their requested count is filled, so this does not alter the law.
        per_row_cap = max(8, max_batch_total // max(len(active), 1))
        target_batch = max(batch_floor, 2 * int(need.max()))
        proposal_n = int(max(8, min(per_row_cap, 65536, target_batch)))
        proposals = sample_conditional_batch(
            mixture, rows[active], panel, proposal_n, int(rng.integers(2**31 - 1))
        )
        attempted += int(len(active) * proposal_n)
        feats = fn(proposals.reshape(-1, mixture.dimension)).reshape(len(active), proposal_n, -1)
        logits = np.einsum("abr,r->ab", feats, beta)
        keep = rng.random((len(active), proposal_n)) < np.exp(logits - envelope[active, None])
        accepted_round = int(keep.sum())
        accepted_raw += accepted_round
        # Low-overlap rows need fewer Python/NumPy setup passes, while ordinary
        # rows should retain small allocations.  Only the proposal batch size
        # changes; every proposal is still drawn from Q0 and accepted with the
        # exact bounded likelihood ratio.
        if accepted_round == 0 or accepted_round / max(len(active) * proposal_n, 1) < 0.01:
            batch_floor = min(65536, batch_floor * 2)
        elif accepted_round / max(len(active) * proposal_n, 1) > 0.5:
            batch_floor = max(8, batch_floor // 2)
        for local, row_id in enumerate(active):
            selected = proposals[local][keep[local]]
            take = min(len(selected), n - filled[row_id])
            if take:
                out[row_id, filled[row_id]:filled[row_id] + take] = selected[:take]
                filled[row_id] += take
    _record_workload(
        conditional_rejection_calls=1,
        conditional_proposals=attempted,
        conditional_accepted_raw=accepted_raw,
        conditional_requested=len(rows) * n,
    )
    return out


def _tilted_conditional_batch_torch(mixture, beta, x_s, panel, n, seed, scale, return_feature_mean=False):
    """Exact rejection TiltCond on CUDA, with the same proposal law as NumPy."""
    device = _cuda_device()
    dtype = torch.float64
    rows = np.atleast_2d(np.asarray(x_s, dtype=float))
    panel = tuple(panel)
    complement = tuple(i for i in range(mixture.dimension) if i not in panel)
    panel_index = torch.as_tensor(panel, dtype=torch.long, device=device)
    complement_index = torch.as_tensor(complement, dtype=torch.long, device=device)
    rows_t = torch.as_tensor(rows, dtype=dtype, device=device)
    z_s = torch.as_tensor(inverse_warp(rows, mixture.alpha), dtype=dtype, device=device)
    beta_t = torch.as_tensor(np.asarray(beta), dtype=dtype, device=device)
    scale_t = torch.as_tensor(np.asarray(scale), dtype=dtype, device=device)
    cache_key = (id(mixture), panel, str(device), str(dtype))
    static = _TORCH_REJECTION_PANEL_CACHE.get(cache_key)
    if static is None:
        static = tuple(
            (
                torch.as_tensor(mean_c, dtype=dtype, device=device),
                torch.as_tensor(mean_s, dtype=dtype, device=device),
                torch.as_tensor(gain, dtype=dtype, device=device),
                torch.as_tensor(chol, dtype=dtype, device=device),
            )
            for mean_c, mean_s, _inv_ss, gain, _cond_cov, chol, _logdet
            in _conditional_parameters(mixture, panel)[1]
        )
        _TORCH_REJECTION_PANEL_CACHE[cache_key] = static
    panel_set = set(panel)
    fixed = np.zeros(beta.shape[0], dtype=bool)
    fixed[:6] = [i in panel_set for i in range(6)]
    fixed[6:] = [i in panel_set and i + 6 in panel_set for i in range(6)]
    full = torch.zeros((len(rows), mixture.dimension), dtype=dtype, device=device)
    full[:, panel_index] = rows_t
    xt_fixed = full / scale_t
    phi_fixed = torch.cat(
        (torch.tanh(xt_fixed[:, :6]), torch.tanh(xt_fixed[:, :6] * xt_fixed[:, 6:12])),
        dim=1,
    )
    fixed_t = torch.as_tensor(fixed, dtype=torch.bool, device=device)
    envelope = phi_fixed[:, fixed_t] @ beta_t[fixed_t] + torch.as_tensor(
        np.abs(np.asarray(beta))[~fixed], dtype=dtype, device=device,
    ).sum()
    posterior_all = torch.as_tensor(
        conditional_component_posterior_batch(mixture, rows, panel),
        dtype=dtype, device=device,
    )
    posterior_cdf = posterior_all.cumsum(dim=1)
    accumulate_mean = bool(return_feature_mean)
    feature_dim = int(beta_t.numel())
    out = None if accumulate_mean else torch.empty(
        (len(rows), n, mixture.dimension), dtype=dtype, device=device
    )
    feat_sum = (
        torch.zeros((len(rows), feature_dim), dtype=dtype, device=device)
        if accumulate_mean else None
    )
    filled = torch.zeros(len(rows), dtype=torch.int64, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed or 0) % (2**63 - 1))
    # Bound a proposal wave to the actual device class.  The original
    # 8,000,000-pair A800 setting can exhaust a 6 GiB workstation GPU because
    # ``z``, component transforms, features, logits and masks coexist.  This
    # changes only batching, not proposal generation or the rejection law.
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    if total_memory <= 8 * 1024**3:
        max_pairs = 250_000
    elif total_memory <= 16 * 1024**3:
        max_pairs = 1_000_000
    else:
        max_pairs = 8_000_000
    batch_floor = max(2048, 16 * int(n))
    proposals_total = 0
    accepted_raw_total = 0
    round_budget = 20000
    round_index = 0
    oom_error = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    while True:
        active = torch.nonzero(filled < n, as_tuple=False).flatten()
        if active.numel() == 0:
            break
        round_index += 1
        if round_index > round_budget:
            remaining_samples = int((n - filled).clamp(min=0).sum().item())
            round_budget = _rejection_round_budget(
                round_index=round_index - 1,
                remaining_samples=remaining_samples,
                proposals_total=proposals_total,
                accepted_raw_total=accepted_raw_total,
            )
            if round_index > round_budget:
                n_filled = filled.detach().cpu().numpy()
                raise RuntimeError(
                    "CUDA exact conditional rejection did not converge; "
                    f"panel={panel}, rounds={round_index - 1}, n={int(n)}, rows={len(rows)}, "
                    f"filled_min={int(n_filled.min())}, filled_max={int(n_filled.max())}, "
                    f"proposals={proposals_total}, accepted_raw={accepted_raw_total}"
                )
        a = int(active.numel())
        remaining = n - filled[active]
        proposal_n = min(65536, max(batch_floor, 8 * int(remaining.max().item())))
        proposal_n = max(8, min(proposal_n, max_pairs // max(a, 1)))
        accepted = 0
        attempted = max(a * proposal_n, 1)
        while proposal_n >= 8:
            try:
                uniforms = torch.rand((a, proposal_n), dtype=dtype, device=device, generator=generator)
                components = (uniforms[:, :, None] > posterior_cdf[active][:, None, :]).sum(dim=2)
                normals = torch.randn((a, proposal_n, len(complement)), dtype=dtype, device=device, generator=generator)
                z = torch.empty((a, proposal_n, mixture.dimension), dtype=dtype, device=device)
                z[:, :, panel_index] = z_s[active].unsqueeze(1)
                z_comp = torch.zeros((a, proposal_n, len(complement)), dtype=dtype, device=device)
                for k, (mean_c, mean_s, gain, chol) in enumerate(static):
                    cond_mean = mean_c[None, :] + (z_s[active] - mean_s) @ gain.T
                    z_k = cond_mean[:, None, :] + normals @ chol.T
                    mask = components == k
                    z_comp = torch.where(mask.unsqueeze(-1), z_k, z_comp)
                z[:, :, complement_index] = z_comp
                x = torch.sinh(float(mixture.alpha) * z) / float(mixture.alpha) if mixture.alpha > 0 else z
                xt = x / scale_t
                phi = torch.cat((torch.tanh(xt[..., :6]), torch.tanh(xt[..., :6] * xt[..., 6:12])), dim=-1)
                logits = torch.einsum("abr,r->ab", phi, beta_t)
                keep = torch.rand((a, proposal_n), dtype=dtype, device=device, generator=generator) < torch.exp(
                    logits - envelope[active].unsqueeze(1)
                )
                rank = keep.cumsum(dim=1)
                take_mask = keep & (rank <= remaining.unsqueeze(1))
                if accumulate_mean:
                    feat_sum[active] += (phi * take_mask.unsqueeze(-1)).sum(dim=1)
                else:
                    dest = filled[active].unsqueeze(1) + rank - 1
                    row_ids = active.unsqueeze(1).expand_as(keep)
                    out[row_ids[take_mask], dest[take_mask]] = x[take_mask]
                filled[active] += take_mask.sum(dim=1)
                accepted = int(keep.sum().item())
                attempted = max(a * proposal_n, 1)
                break
            except Exception as exc:
                message = str(exc).lower()
                if not (isinstance(exc, oom_error) or "out of memory" in message):
                    raise
                torch.cuda.empty_cache()
                next_n = max(8, proposal_n // 2)
                if next_n == proposal_n:
                    raise
                proposal_n = next_n
        proposals_total += attempted
        accepted_raw_total += accepted
        if accepted / attempted < 0.01:
            batch_floor = min(65536, batch_floor * 2)
        elif accepted / attempted > 0.5:
            batch_floor = max(2048, batch_floor // 2)
    _record_workload(
        conditional_rejection_calls=1,
        conditional_proposals=proposals_total,
        conditional_accepted_raw=accepted_raw_total,
        conditional_requested=len(rows) * n,
    )
    if accumulate_mean:
        return (feat_sum / float(n)).cpu().numpy()
    return out.cpu().numpy()


def tilted_conditional_feature_mean_batch(
    mixture, beta, x_s, panel, n, seed, scale, feature_fn=None,
) -> np.ndarray:
    """Mean feature map of exact TiltCond completions, kept on device when possible."""
    if _cuda_device() is not None and feature_fn is None:
        return _tilted_conditional_batch_torch(
            mixture, beta, x_s, panel, n, seed, scale, return_feature_mean=True,
        )
    samples = tilted_conditional_batch(mixture, beta, x_s, panel, n, seed, scale, feature_fn=feature_fn)
    fn = (lambda x: feature_map(x, scale)) if feature_fn is None else feature_fn
    return fn(samples).mean(axis=1)


_HALTON_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)


def _halton_normals(n: int, dimension: int, skip: int = 0) -> np.ndarray:
    """Deterministic nested low-discrepancy standard normals (no SciPy)."""
    if dimension == 0:
        return np.empty((n, 0))
    # Box-Muller consumes pairs of radical-inverse coordinates.  The extra
    # coordinate is harmless for odd complement dimensions and preserves the
    # same prefix when n doubles, which is required for order diagnostics.
    need = 2 * ((dimension + 1) // 2)
    u = np.empty((n, need), dtype=float)
    idx = np.arange(skip + 1, skip + n + 1, dtype=np.int64)
    for j in range(need):
        base = _HALTON_PRIMES[j]
        value = idx.copy()
        radical = np.zeros(n, dtype=float)
        inv = 1.0 / base
        while np.any(value):
            value, digit = divmod(value, base)
            radical += digit * inv
            inv /= base
        u[:, j] = np.clip(radical, 1e-12, 1.0 - 1e-12)
    z = np.empty((n, dimension), dtype=float)
    for j in range(0, dimension, 2):
        radius = np.sqrt(-2.0 * np.log(u[:, j]))
        z[:, j] = radius * np.cos(2.0 * np.pi * u[:, j + 1])
        if j + 1 < dimension:
            z[:, j + 1] = radius * np.sin(2.0 * np.pi * u[:, j + 1])
    return z


def _nested_qmc_normals(n: int, dimension: int, seed: int = 0) -> np.ndarray:
    """Sobol normal net with a deterministic Halton fallback."""
    try:
        return _nested_qmc_normals_impl(n, dimension, seed)
    except MemoryError:
        _release_cuda_workspace(drop_qmc_cache=True)
        return _nested_qmc_normals_impl(n, dimension, seed)


def _nested_qmc_normals_impl(n: int, dimension: int, seed: int = 0) -> np.ndarray:
    if qmc is None or ndtri is None:
        return _halton_normals(n, dimension, skip=seed)
    if dimension == 0:
        return np.empty((n, 0))
    if n & (n - 1):
        raise ValueError("nested Sobol sample size must be a power of two")
    engine = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
    uniforms = np.clip(engine.random_base2(int(np.log2(n))), 1e-12, 1.0 - 1e-12)
    return ndtri(uniforms)


def tilted_conditional_mean_qmc(
    mixture: FrozenMixture,
    beta: np.ndarray,
    x_s: np.ndarray,
    panel: tuple[int, ...],
    order: int,
    seed: int = 0,
    scale: np.ndarray | None = None,
    feature_fn=None,
) -> np.ndarray:
    """Nested QMC approximation to E_{Q_beta}[phi(X)|X_S=x_s].

    The complete tilted integrand is evaluated at every conditional Gaussian
    point: all unary and pairwise features remain coupled through
    ``exp(beta @ phi)``.  This is a diagnostic alternative to rejection LU;
    order ``k`` uses ``2**k`` points and shares its prefix with order ``k+1``.
    """
    if order < 4 or order > 18:
        raise ValueError("QMC order must be in [4, 18]")
    fn = (lambda x: feature_map(x, scale)) if feature_fn is None else feature_fn
    panel = tuple(panel)
    rows = np.atleast_2d(np.asarray(x_s, dtype=float))
    complement = tuple(i for i in range(mixture.dimension) if i not in panel)
    n = 1 << int(order)
    _QMC_ORDERS.append(int(order))
    _record_workload(qmc_calls=1, qmc_nodes=len(rows) * n, qmc_scrambles=1)
    z_s = inverse_warp(rows, mixture.alpha)
    posterior = conditional_component_posterior_batch(mixture, rows, panel)
    parameters = _conditional_parameters(mixture, panel)[1]
    if _cuda_device() is not None and feature_fn is None:
        return _tilted_conditional_mean_qmc_torch(
            mixture, beta, rows, panel, order, seed, scale, posterior, parameters,
        )
    # Vectorize all observed rows for one component.  This changes only the
    # evaluation order, not the QMC nodes or the mixture-weighted ratio, and
    # avoids a Python loop over every acquired observation at formal budgets.
    normals = _nested_qmc_normals(n, len(complement), seed=int(seed))
    all_phi, all_logw = [], []
    for mean_c, mean_s, _inv_ss, gain, _cond_cov, chol, _logdet in parameters:
        z = np.empty((len(rows), n, mixture.dimension), dtype=float)
        z[:, :, list(panel)] = z_s[:, None, :]
        cond_mean = mean_c[None, :] + (z_s - mean_s) @ gain.T
        z[:, :, list(complement)] = cond_mean[:, None, :] + normals[None, :, :] @ chol.T
        phi = fn(warp(z.reshape(-1, mixture.dimension), mixture.alpha)).reshape(len(rows), n, -1)
        all_phi.append(phi)
        all_logw.append(np.einsum("nkr,r->nk", phi, np.asarray(beta)))
    shift = np.maximum.reduce([np.max(v, axis=1) for v in all_logw])
    numer = np.zeros((len(rows), beta.shape[0]), dtype=float)
    denom = np.zeros(len(rows), dtype=float)
    for phi, logw, prob in zip(all_phi, all_logw, posterior.T):
        w = prob[:, None] * np.exp(logw - shift[:, None])
        denom += w.mean(axis=1)
        numer += np.einsum("nk,nkr->nr", w, phi) / n
    out = numer / np.maximum(denom[:, None], 1e-300)
    return out


def _conditional_qmc_at_order(
    mixture, beta, rows, panel, order, seed, scale, feature_fn
):
    """Evaluate one Sobol order for a row subset, without filling max_order."""
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    beta = np.asarray(beta, dtype=float)
    if len(rows) == 0:
        return np.zeros((0, beta.shape[0]))
    use_cuda = _cuda_device() is not None and feature_fn is None
    max_rows = _cuda_qmc_max_rows(order) if use_cuda else len(rows)
    if len(rows) > max_rows:
        parts = []
        for start in range(0, len(rows), max_rows):
            parts.append(
                _conditional_qmc_at_order(
                    mixture, beta, rows[start:min(start + max_rows, len(rows))],
                    panel, order, seed, scale, feature_fn,
                )
            )
            _release_cuda_workspace()
        return np.concatenate(parts, axis=0)
    if use_cuda:
        estimates = tilted_conditional_mean_qmc_nested(
            mixture, beta, rows, panel, int(order), seed=int(seed),
            scale=scale, feature_fn=feature_fn, return_orders=(int(order),),
        )
        return estimates[int(order)]
    return tilted_conditional_mean_qmc(
        mixture, beta, rows, panel, int(order), seed=int(seed),
        scale=scale, feature_fn=feature_fn,
    )


def _conditional_qmc_ladder(
    mixture, beta, rows, panel, start_order, end_order, seed, scale, feature_fn,
):
    """Evaluate a nested order ladder while constructing the integrand once.

    The CUDA evaluator already exposes all nested prefixes from one highest
    order call.  Adaptive gold used to call it separately for every order,
    rebuilding the same conditional Gaussian/feature tensor repeatedly.  A
    short ladder keeps the row-wise stopping rule unchanged while removing
    that redundant work.  CPU/custom-feature paths retain the original
    per-order implementation.
    """
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    requested = tuple(range(int(start_order), int(end_order) + 1))
    if len(rows) == 0:
        return {order: np.zeros((0, np.asarray(beta).size), dtype=float) for order in requested}
    use_cuda = _cuda_device() is not None and feature_fn is None
    max_rows = _cuda_qmc_max_rows(int(end_order)) if use_cuda else len(rows)
    if len(rows) > max_rows:
        pieces = {order: [] for order in requested}
        for start in range(0, len(rows), max_rows):
            chunk = _conditional_qmc_ladder(
                mixture, beta, rows[start:min(start + max_rows, len(rows))], panel,
                start_order, end_order, seed, scale, feature_fn,
            )
            for order in requested:
                pieces[order].append(chunk[order])
            _release_cuda_workspace()
        return {order: np.concatenate(pieces[order], axis=0) for order in requested}
    if use_cuda:
        return {
            int(order): np.asarray(value, dtype=np.float64)
            for order, value in tilted_conditional_mean_qmc_nested(
                mixture, beta, rows, tuple(panel), int(end_order), seed=int(seed),
                scale=scale, feature_fn=feature_fn, return_orders=requested,
            ).items()
        }
    return {
        int(order): tilted_conditional_mean_qmc(
            mixture, beta, rows, tuple(panel), int(order), seed=int(seed),
            scale=scale, feature_fn=feature_fn,
        )
        for order in requested
    }


def tilted_conditional_mean_exact(
    mixture: FrozenMixture,
    beta: np.ndarray,
    x_s: np.ndarray,
    panel: tuple[int, ...],
    seed: int = 0,
    scale: np.ndarray | None = None,
    feature_fn=None,
    start_order: int = 8,
    max_order: int = 16,
    atol: float = 2e-6,
    rtol: float = 2e-5,
    scrambles: int = 4,
    scramble_se_atol: float | None = None,
    scramble_se_rtol: float | None = None,
    return_diagnostics: bool = False,
):
    """Deterministic nested conditional-score oracle with an error certificate.

    The Gaussian-mixture conditional law is exact; only its bounded nonlinear
    expectation needs numerical integration.  Each query row doubles its own
    nested Sobol prefix from ``start_order`` until successive estimates agree
    componentwise; converged rows stop and do not pay later orders.  Calls that
    do not meet the certificate are explicit failures rather than silently
    being treated as exact scores.
    """
    if start_order < 4 or max_order < start_order:
        raise ValueError("invalid adaptive conditional integration orders")
    if int(scrambles) < 2:
        raise ValueError("adaptive conditional integration requires at least two independent scrambles")
    scramble_se_atol = float(atol if scramble_se_atol is None else scramble_se_atol)
    scramble_se_rtol = float(rtol if scramble_se_rtol is None else scramble_se_rtol)
    rows = np.atleast_2d(np.asarray(x_s, dtype=float))
    beta = np.asarray(beta, dtype=float)
    n_rows = len(rows)
    estimate = np.zeros((n_rows, beta.shape[0]))
    previous = None
    active = np.ones(n_rows, dtype=bool)
    row_final_order = np.full(n_rows, int(max_order), dtype=int)
    row_abs_delta = np.full(n_rows, np.inf)
    row_scramble_se = np.full(n_rows, np.inf)
    n_active_by_order = {}
    diagnostics = {
        "method": "multi_scramble_sobol_adaptive",
        "orders": [],
        "converged": False,
        "max_abs_delta": None,
        "max_rel_delta": None,
        "scramble_se": None,
        "scrambles": int(scrambles),
        "start_order": int(start_order),
        "max_order": int(max_order),
        "n_active_by_order": n_active_by_order,
    }
    # A block of nested prefixes is substantially cheaper than reconstructing
    # the same feature tensor for every order.  Keep the first block modest so
    # rows that converge early do not pay for the full max order; only the
    # residual active rows enter later blocks.
    block_width = 6
    block_start = int(start_order)
    while block_start <= int(max_order) and np.any(active):
        block_end = min(int(max_order), block_start + block_width)
        block_index = np.flatnonzero(active)
        replicates_by_order = {order: [] for order in range(block_start, block_end + 1)}
        for scramble in range(int(scrambles)):
            scramble_seed = int(seed) + 1_000_003 * scramble
            ladder = _conditional_qmc_ladder(
                mixture, beta, rows[block_index], panel,
                block_start, block_end, scramble_seed, scale, feature_fn,
            )
            for order in replicates_by_order:
                replicates_by_order[order].append(ladder[order])
        for order in replicates_by_order:
            active_index = np.flatnonzero(active)
            n_active_by_order[str(order)] = int(len(active_index))
            diagnostics["orders"].append(order)
            # ``block_index`` is the active set at block entry.  Rows that
            # stopped at an earlier prefix are ignored for this order; their
            # values remain certified in ``estimate``.
            local = np.flatnonzero(np.isin(block_index, active_index))
            if local.size == 0:
                continue
            replicate_values = np.stack(replicates_by_order[order], axis=0)[:, local]
            mean_active = replicate_values.mean(axis=0)
            se_active = replicate_values.std(axis=0, ddof=1) / np.sqrt(int(scrambles))
            current_index = block_index[local]
            estimate[current_index] = mean_active
            row_se = np.max(se_active, axis=1)
            row_scramble_se[current_index] = row_se
            diagnostics["scramble_se"] = float(np.max(row_scramble_se[np.isfinite(row_scramble_se)]))
            if previous is not None:
                delta = np.abs(mean_active - previous[current_index])
                row_delta = np.max(delta, axis=1)
                row_abs_delta[current_index] = row_delta
                scale_norm = np.maximum(
                    np.max(np.abs(mean_active), axis=1),
                    np.max(np.abs(previous[current_index]), axis=1),
                )
                scale_norm = np.maximum(scale_norm, 1.0)
                newly = (
                    (row_delta <= float(atol) + float(rtol) * scale_norm)
                    & (row_se <= scramble_se_atol + scramble_se_rtol * scale_norm)
                )
                stopped = current_index[newly]
                row_final_order[stopped] = order
                active[stopped] = False
                diagnostics["max_abs_delta"] = float(np.max(row_abs_delta[np.isfinite(row_abs_delta)]))
                max_scale = float(np.max(np.abs(estimate))) if n_rows else 1.0
                diagnostics["max_rel_delta"] = float(
                    diagnostics["max_abs_delta"] / max(max_scale, 1.0)
                )
                if not np.any(active):
                    diagnostics["converged"] = True
                    diagnostics["n_unconverged"] = 0
                    diagnostics["final_order"] = int(row_final_order.max())
                    diagnostics["row_final_order"] = row_final_order
                    _release_cuda_workspace()
                    return (estimate, diagnostics) if return_diagnostics else estimate
            previous = estimate.copy()
        block_start = block_end + 1
        _release_cuda_workspace()
    diagnostics["final_order"] = int(max_order)
    diagnostics["n_unconverged"] = int(np.count_nonzero(active))
    diagnostics["row_final_order"] = row_final_order
    diagnostics["max_abs_delta"] = float(np.max(row_abs_delta[np.isfinite(row_abs_delta)])) if n_rows else 0.0
    max_scale = float(np.max(np.abs(estimate))) if n_rows else 1.0
    diagnostics["max_rel_delta"] = float(diagnostics["max_abs_delta"] / max(max_scale, 1.0))
    diagnostics["scramble_se"] = float(np.max(row_scramble_se[np.isfinite(row_scramble_se)])) if n_rows else 0.0
    if return_diagnostics:
        return estimate, diagnostics
    raise RuntimeError(
        "adaptive exact conditional integral did not meet tolerance: "
        f"abs={diagnostics['max_abs_delta']}, rel={diagnostics['max_rel_delta']}, "
        f"scramble_se={diagnostics['scramble_se']}"
    )


def tilted_conditional_mean_qmc_nested(
    mixture, beta, x_s, panel, order, seed=0, scale=None, feature_fn=None,
    return_orders=(None,),
):
    """Evaluate one highest-order CUDA QMC grid and reuse nested prefixes."""
    rows = np.atleast_2d(np.asarray(x_s, dtype=float))
    _QMC_ORDERS.append(int(order))
    _record_workload(qmc_calls=1, qmc_nodes=len(rows) * (1 << int(order)), qmc_scrambles=1)
    complement = tuple(i for i in range(mixture.dimension) if i not in tuple(panel))
    z_s = inverse_warp(rows, mixture.alpha)
    posterior = conditional_component_posterior_batch(mixture, rows, tuple(panel))
    parameters = _conditional_parameters(mixture, tuple(panel))[1]
    return _tilted_conditional_mean_qmc_torch(
        mixture, beta, rows, tuple(panel), int(order), seed, scale, posterior,
        parameters, return_orders=tuple(int(x) for x in return_orders if x is not None),
    )


def _tilted_conditional_mean_qmc_torch(mixture, beta, rows, panel, order, seed, scale, posterior, parameters, return_orders=None):
    """Inference-only wrapper for the CUDA conditional integral."""
    with torch.inference_mode():
        return _tilted_conditional_mean_qmc_torch_impl(
            mixture, beta, rows, panel, order, seed, scale, posterior,
            parameters, return_orders=return_orders,
        )


def _tilted_conditional_mean_qmc_torch_impl(mixture, beta, rows, panel, order, seed, scale, posterior, parameters, return_orders=None):
    """CUDA implementation of the same complete tilted QMC integrand."""
    device = _cuda_device()
    # The bulk fixed-QMC path is calibrated against the float64 evaluator,
    # but doing the full conditional tensor in float64 on a laptop RTX 4050
    # makes each scoring step unnecessarily slow.  Keep the node/warp/logit
    # hot path in float32 and return NumPy float64 reductions to the solver.
    # The final beta/H/KL operations remain float64 in ``paper_run``.
    dtype = torch.float32
    n = 1 << int(order)
    complement = tuple(i for i in range(mixture.dimension) if i not in panel)
    feature_dimension = 12
    target_missing = tuple(i for i in range(feature_dimension) if i not in panel)
    target_missing_idx = tuple(complement.index(i) for i in target_missing)
    z_s = torch.as_tensor(inverse_warp(rows, mixture.alpha), dtype=dtype, device=device)
    # Keep the original full-complement Sobol dimensionality so fixed seeds
    # remain paired with the pre-optimization implementation.  Only the
    # dimensions needed by the feature map are gathered for the reduced draw.
    normal_key = (str(device), int(n), len(complement), int(seed))
    normals = _TORCH_QMC_NORMAL_CACHE.get(normal_key)
    if normals is None:
        normals = torch.as_tensor(
            _nested_qmc_normals(n, len(complement), seed=int(seed)),
            dtype=dtype, device=device,
        )
        _TORCH_QMC_NORMAL_CACHE[normal_key] = normals
        _TORCH_QMC_NORMAL_CACHE.move_to_end(normal_key)
        while len(_TORCH_QMC_NORMAL_CACHE) > _TORCH_QMC_NORMAL_CACHE_LIMIT:
            _TORCH_QMC_NORMAL_CACHE.popitem(last=False)
    else:
        _TORCH_QMC_NORMAL_CACHE.move_to_end(normal_key)
    scale_t = torch.as_tensor(np.asarray(scale), dtype=dtype, device=device)
    beta_t = torch.as_tensor(np.asarray(beta), dtype=dtype, device=device)
    cache_key = (id(mixture), tuple(panel), str(device), str(dtype))
    static = _TORCH_PANEL_TENSOR_CACHE.get(cache_key)
    if static is None:
        static = tuple(
            (
                torch.as_tensor(mean_c[list(target_missing_idx)], dtype=dtype, device=device),
                torch.as_tensor(mean_s, dtype=dtype, device=device),
                torch.as_tensor(gain[list(target_missing_idx), :], dtype=dtype, device=device),
                torch.as_tensor(
                    np.linalg.cholesky(_cond_cov[np.ix_(target_missing_idx, target_missing_idx)]),
                    dtype=dtype, device=device,
                ),
            )
            for mean_c, mean_s, _inv_ss, gain, _cond_cov, chol, _logdet in parameters
        )
        _TORCH_PANEL_TENSOR_CACHE[cache_key] = static
    # The frozen compact parameter set has ||beta||_1 <= 5 and every feature
    # coordinate lies in [-1, 1].  A deterministic global envelope is thus a
    # finite, row-independent log-weight shift.  It avoids a preliminary pass
    # over every component while preserving the normalized integral exactly.
    shift = torch.full((len(rows),), float(np.abs(np.asarray(beta)).sum()), dtype=dtype, device=device)
    posterior_t = torch.as_tensor(posterior, dtype=dtype, device=device)
    requested = None if return_orders is None else tuple(sorted(set(int(v) for v in return_orders)))
    sizes = (n,) if not requested else tuple(1 << q for q in requested)
    numer = {size: torch.zeros((len(rows), beta_t.numel()), dtype=dtype, device=device) for size in sizes}
    denom = {size: torch.zeros(len(rows), dtype=dtype, device=device) for size in sizes}
    for k, (mean_c_t, mean_s_t, gain_t, chol_t) in enumerate(static):
        z = torch.empty((len(rows), n, feature_dimension), dtype=dtype, device=device)
        observed_feature = [j for j, coordinate in enumerate(panel) if coordinate < feature_dimension]
        if observed_feature:
            z[:, :, [panel[j] for j in observed_feature]] = z_s[:, None, observed_feature]
        cond_mean = mean_c_t[None, :] + (z_s - mean_s_t) @ gain_t.T
        target_normals = normals[:, list(target_missing_idx)]
        z[:, :, list(target_missing)] = cond_mean[:, None, :] + target_normals[None, :, :] @ chol_t.T
        x = torch.sinh(float(mixture.alpha) * z) / float(mixture.alpha) if mixture.alpha > 0 else z
        xt = x / scale_t[:feature_dimension]
        phi = torch.cat((torch.tanh(xt[..., :6]), torch.tanh(xt[..., :6] * xt[..., 6:12])), dim=-1)
        logw = torch.einsum("nkr,r->nk", phi, beta_t)
        w = posterior_t[:, k, None] * torch.exp(logw - shift[:, None])
        for size in sizes:
            wp = w[:, :size]
            numer[size] += torch.einsum("nk,nkr->nr", wp, phi[:, :size]) / size
            denom[size] += wp.mean(dim=1)
        del z, x, xt, phi, logw, w
    estimates = {int(np.log2(size)): (numer[size] / denom[size][:, None].clamp_min(1e-300)).cpu().numpy() for size in sizes}
    if requested:
        return estimates
    return estimates[int(order)]


def conditional_moments(mixture: FrozenMixture, x_s: np.ndarray, panel: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Analytic conditional mean/covariance in warped coordinates' latent space.

    The exact Gaussian-mixture conditional is used as the sampling oracle check;
    moments are returned for the latent coordinates because the nonlinear warp
    has no closed-form Gaussian moments.
    """
    panel = tuple(panel)
    complement = tuple(i for i in range(mixture.dimension) if i not in panel)
    z_s = inverse_warp(np.asarray(x_s), mixture.alpha)
    posterior = conditional_component_posterior(mixture, x_s, panel)
    means = []
    covs = []
    for k in range(len(posterior)):
        cov = mixture.covariances[k]
        ss = np.ix_(panel, panel)
        cs = np.ix_(complement, panel)
        cond_mean = mixture.means[k][list(complement)] + cov[cs] @ np.linalg.solve(cov[ss], z_s - mixture.means[k][list(panel)])
        cond_cov = cov[np.ix_(complement, complement)] - cov[cs] @ np.linalg.solve(cov[ss], cov[np.ix_(panel, complement)])
        means.append(cond_mean)
        covs.append(cond_cov)
    means = np.asarray(means)
    mean = posterior @ means
    covariance = sum(p * (cov + np.outer(mu - mean, mu - mean)) for p, mu, cov in zip(posterior, means, covs))
    return mean, covariance


def feature_map(x: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 0:
        rows = 0
    elif x.ndim == 1:
        rows = 1
    else:
        rows = int(x.reshape(-1, x.shape[-1]).shape[0])
    _record_workload(feature_scans=rows)
    if x.shape[-1] != 16:
        raise ValueError("synthetic feature map expects 16 coordinates")
    scale = np.ones(16) if scale is None else np.asarray(scale)
    if scale.shape != (16,) or np.any(scale <= 0):
        raise ValueError("scale must be a positive length-16 vector")
    x_tilde = x / scale
    unary = np.tanh(x_tilde[..., :6])
    pair = np.tanh(x_tilde[..., :6] * x_tilde[..., 6:12])
    return np.concatenate([unary, pair], axis=-1)


# ---------------------------------------------------------------------------
# P7 S2 reuse: bounded unary/pairwise feature dictionary.
#
# PDF 7.3 freezes the single-task feature map ``phi`` above but requires that a
# generator trained once can be *reused* across campaigns that redraw their own
# 12-feature map from a pre-defined bounded dictionary, redraw ``beta*``, and
# redraw a subset of candidate pair panels.  ``feature_fn_from_dictionary``
# materializes any such draw as an opaque ``x -> R^r`` callable so the whole
# downstream tilt / information machinery can accept ``feature_fn`` uniformly.
# ---------------------------------------------------------------------------

UNARY_FEATURE = "unary"
PAIR_FEATURE = "pair"


def make_feature_dictionary(dimension: int = 16) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Bounded unary/pairwise dictionary (PDF 7.3) over all coordinate pairs."""
    dim = int(dimension)
    unary = [(UNARY_FEATURE, (j,)) for j in range(dim)]
    pairs = [(PAIR_FEATURE, (i, j)) for i in range(dim) for j in range(i + 1, dim)]
    return tuple(unary + pairs)


def sample_feature_draw(dictionary, seed: int, r: int = 12):
    """Draw ``r`` features without replacement from the dictionary."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dictionary), size=r, replace=False)
    return [dictionary[i] for i in idx]


def feature_fn_from_dictionary(features, scale: np.ndarray | None = None):
    """Build an ``x -> R^r`` callable for a list of (kind, coords) features."""
    scale = np.ones(16) if scale is None else np.asarray(scale, dtype=float)

    def fn(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x_tilde = x / scale
        cols = []
        for kind, coords in features:
            if kind == UNARY_FEATURE:
                cols.append(np.tanh(x_tilde[..., coords[0]]))
            else:  # PAIR_FEATURE
                i, j = coords
                cols.append(np.tanh(x_tilde[..., i] * x_tilde[..., j]))
        return np.stack(cols, axis=-1)

    return fn


def reference_scale(mixture: FrozenMixture, n: int = 6000, seed: int = 2026) -> np.ndarray:
    """Estimate the frozen per-coordinate Q0 reference scale from an independent pool."""
    if n <= 1:
        raise ValueError("reference scale pool size must exceed one")
    samples = sample_full(mixture, n, seed=seed)
    scale = samples.std(axis=0, ddof=1)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("reference scale must be finite and positive")
    return scale


def log_partition(beta: np.ndarray, reference_samples: np.ndarray, scale: np.ndarray | None = None,
                  feature_fn=None) -> float:
    from numpy import logaddexp
    beta = np.asarray(beta)
    fn = (lambda x: feature_map(x, scale)) if feature_fn is None else feature_fn
    values = fn(reference_samples) @ beta
    return float(logaddexp.reduce(values) - np.log(values.size))


def tilted_moments(beta: np.ndarray, reference_samples: np.ndarray, scale: np.ndarray | None = None,
                   feature_fn=None) -> tuple[np.ndarray, np.ndarray]:
    fn = (lambda x: feature_map(x, scale)) if feature_fn is None else feature_fn
    features = fn(reference_samples)
    logits = features @ np.asarray(beta)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    mean = weights @ features
    centered = features - mean
    fisher = (centered * weights[:, None]).T @ centered
    return mean, fisher


def tilted_sample_from_reference(beta: np.ndarray, reference_samples: np.ndarray, n: int, seed: int = 0,
                                 scale: np.ndarray | None = None, feature_fn=None) -> np.ndarray:
    """Sample Q_beta by exact importance resampling from an independent Q0 pool."""
    rng = np.random.default_rng(seed)
    fn = (lambda x: feature_map(x, scale)) if feature_fn is None else feature_fn
    features = fn(reference_samples)
    logits = features @ np.asarray(beta)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    return np.asarray(reference_samples)[rng.choice(len(reference_samples), size=n, replace=True, p=weights)]


def tilted_full_sample(mixture: FrozenMixture, beta: np.ndarray, n: int, seed: int = 0,
                       scale: np.ndarray | None = None, feature_fn=None) -> np.ndarray:
    """Exact accept-reject samples from Q_beta relative to Q0."""
    rng = np.random.default_rng(seed)
    fn = (lambda x: feature_map(x, scale)) if feature_fn is None else feature_fn
    envelope = float(np.sum(np.abs(np.asarray(beta))))
    accepted = []
    proposals_total = 0
    accepted_raw = 0
    while len(accepted) < n:
        proposal = sample_full(mixture, max(256, min(4096, 4 * (n - len(accepted)))), int(rng.integers(2**31 - 1)))
        logits = fn(proposal) @ beta
        keep = rng.random(len(proposal)) < np.exp(logits - envelope)
        proposals_total += len(proposal)
        accepted_raw += int(keep.sum())
        accepted.extend(proposal[keep])
    _record_workload(
        full_rejection_calls=1,
        full_proposals=proposals_total,
        full_accepted_raw=accepted_raw,
        full_requested=n,
    )
    return np.asarray(accepted[:n])


def beta_direction_and_scale(reference_samples: np.ndarray, seed: int = 2026, target_ess_fraction: float = 0.5,
                             scale: np.ndarray | None = None, feature_fn=None) -> np.ndarray:
    """Freeze a nonzero direction and choose its magnitude by ESS/N bisection."""
    rng = np.random.default_rng(seed)
    fn = (lambda x: feature_map(x, scale)) if feature_fn is None else feature_fn
    phi = fn(reference_samples)
    r = phi.shape[-1]
    direction = rng.normal(size=r)
    direction /= np.linalg.norm(direction)
    target = target_ess_fraction * len(phi)
    lo, hi = 0.0, 8.0
    for _ in range(60):
        mag = (lo + hi) / 2
        logits = phi @ (mag * direction)
        weights = np.exp(logits - logits.max())
        ess = weights.sum() ** 2 / np.sum(weights ** 2)
        if ess > target:
            lo = mag
        else:
            hi = mag
    return ((lo + hi) / 2) * direction


def full_target_kl(beta_true: np.ndarray, beta_est: np.ndarray, reference_samples: np.ndarray,
                   scale: np.ndarray | None = None, feature_fn=None) -> float:
    mean_true, _ = tilted_moments(beta_true, reference_samples, scale, feature_fn=feature_fn)
    return float((np.asarray(beta_true) - np.asarray(beta_est)) @ mean_true
                 - log_partition(beta_true, reference_samples, scale, feature_fn=feature_fn)
                 + log_partition(beta_est, reference_samples, scale, feature_fn=feature_fn))
