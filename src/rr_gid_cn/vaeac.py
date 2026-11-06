"""Canonical VAEAC generator used as the frozen P6 reference model."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .synthetic_oracle import feature_map


def load_vaeac_checkpoint(path, device="cpu", expected_dim=None):
    """Load a canonical VAEAC checkpoint and reconstruct its exact network.

    Formal stages must not silently load the pre-canonical empirical-generator
    checkpoints.  Canonical checkpoints contain proposal, conditional_prior and
    decoder parameters plus the training standardisation statistics.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"{path}: missing VAEAC model state")
    state = checkpoint["model"]
    required = ("proposal.0.weight", "conditional_prior.0.weight", "decoder.0.weight")
    if any(k not in state for k in required):
        raise ValueError(
            f"{path}: non-canonical checkpoint; expected VAEAC proposal/conditional_prior/decoder"
        )
    proposal_in = int(state["proposal.0.weight"].shape[1])
    hidden = int(state["proposal.0.weight"].shape[0])
    dim = proposal_in // 2
    latent = int(state["proposal.4.weight"].shape[0] // 2)
    if proposal_in != 2 * dim or (expected_dim is not None and dim != int(expected_dim)):
        raise ValueError(f"{path}: inconsistent VAEAC dimension {dim!r}")
    model = VAEAC(dim=dim, latent=latent, hidden=hidden, seed=0).to(device)
    model.load_state_dict(state, strict=True)
    if "data_mean" not in checkpoint or "data_std" not in checkpoint:
        raise ValueError(f"{path}: canonical checkpoint is missing data_mean/data_std")
    data_mean = checkpoint["data_mean"]
    data_std = checkpoint["data_std"]
    if np.asarray(data_mean).shape != (dim,) or np.asarray(data_std).shape != (dim,):
        raise ValueError(f"{path}: invalid VAEAC standardisation statistics")
    model.data_mean = np.asarray(data_mean, dtype=np.float64)
    model.data_std = np.where(np.asarray(data_std, dtype=np.float64) < 1e-12, 1.0, data_std)
    model.eval()
    return model, checkpoint


class PanelPhiNet(nn.Module):
    """Independent 16-d -> E[phi] head per Gas panel, matching kNN's geometry."""

    def __init__(self, n_panels=120, obs_dim=16, hidden=128, out=16, embed=32):
        super().__init__()
        del embed
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(int(obs_dim), int(hidden)),
                nn.Tanh(),
                nn.Linear(int(hidden), int(hidden)),
                nn.Tanh(),
                nn.Linear(int(hidden), int(out)),
            )
            for _ in range(int(n_panels))
        ])

    def forward(self, observed, panel_ids):
        pid = int(panel_ids[0].item())
        return self.nets[pid](observed)


class VAEAC(nn.Module):
    """VAEAC with proposal q, conditional prior p and Gaussian decoder."""
    def __init__(self, dim=16, latent=16, hidden=128, seed=0, k_prior=4, gmm_prior=False):
        super().__init__()
        del k_prior, gmm_prior
        torch.manual_seed(seed)
        self.dim, self.latent = int(dim), int(latent)
        self.proposal = nn.Sequential(nn.Linear(2 * dim, hidden), nn.Tanh(),
                                      nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2 * latent))
        self.conditional_prior = nn.Sequential(nn.Linear(2 * dim, hidden), nn.Tanh(),
                                               nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2 * latent))
        self.decoder = nn.Sequential(nn.Linear(latent + 2 * dim, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2 * dim))
        self.gmm_prior = False
        self.data_mean = np.zeros(dim, dtype=np.float64)
        self.data_std = np.ones(dim, dtype=np.float64)

    @staticmethod
    def _split(out):
        mu, logvar = out.chunk(2, dim=-1)
        return mu, torch.clamp(logvar, -8.0, 6.0)

    def encode(self, x, mask):
        return self._split(self.proposal(torch.cat([x, mask], dim=-1)))

    def prior(self, x_masked, mask):
        return self._split(self.conditional_prior(torch.cat([x_masked, mask], dim=-1)))

    def kl_divergence(self, q_mu, q_logvar, p_mu, p_logvar):
        return 0.5 * (p_logvar - q_logvar +
                      (q_logvar.exp() + (q_mu - p_mu).pow(2)) / p_logvar.exp() - 1).sum(-1)

    def reparameterize(self, mu, logvar, seed=None):
        if seed is None:
            eps = torch.randn_like(mu)
        else:
            gen = torch.Generator(device=mu.device).manual_seed(seed)
            eps = torch.randn(mu.shape, device=mu.device, generator=gen)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode_params(self, z, mask, x_masked=None):
        if x_masked is None:
            x_masked = torch.zeros(z.shape[0], self.dim, device=z.device, dtype=z.dtype)
        return self._split(self.decoder(torch.cat([z, x_masked, mask], dim=-1)))

    def decode(self, z, mask, x_masked=None):
        return self.decode_params(z, mask, x_masked)[0]

    def forward(self, x, mask):
        x_masked = x * mask
        q_mu, q_logvar = self.encode(x, mask)
        p_mu, p_logvar = self.prior(x_masked, mask)
        z = self.reparameterize(q_mu, q_logvar)
        recon_mu, recon_logvar = self.decode_params(z, mask, x_masked)
        return recon_mu, recon_logvar, q_mu, q_logvar, p_mu, p_logvar


def train_vaeac(model, reference, panels, scale=None, alpha=1.0, epochs=30, batch=512,
                lr=1e-3, device="cuda", seed=0, beta=1.0, z_normalize=True, free_bits=0.0):
    """Train canonical VAEAC on complete records using its masked ELBO."""
    del scale, alpha, free_bits
    model = model.to(device)
    raw = np.asarray(reference, dtype=np.float64)
    center = raw.mean(0) if z_normalize else np.zeros(model.dim)
    spread = raw.std(0, ddof=1) if z_normalize else np.ones(model.dim)
    spread = np.where(spread < 1e-6, 1.0, spread)
    model.data_mean, model.data_std = center.astype(np.float64), spread.astype(np.float64)
    x = torch.as_tensor((raw - center) / spread, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng, panels = np.random.default_rng(seed), list(panels)
    history = []
    for _ in range(epochs):
        perm, total, steps = torch.randperm(len(x), device=device), 0.0, 0
        for start in range(0, len(x), batch):
            xb = x[perm[start:start + batch]]
            mask = torch.zeros(len(xb), model.dim, device=device)
            for j in range(len(xb)):
                u = rng.random()
                if u < 0.10:
                    pass
                elif u < 0.20:
                    mask[j] = 1.0
                else:
                    mask[j, list(panels[int(rng.integers(len(panels)))])] = 1.0
            recon_mu, recon_logvar, q_mu, q_logvar, p_mu, p_logvar = model(xb, mask)
            missing = 1.0 - mask
            nll = 0.5 * (recon_logvar + (xb - recon_mu).pow(2) * torch.exp(-recon_logvar))
            recon_loss = (nll * missing).sum(-1) / missing.sum(-1).clamp_min(1.0)
            kl = model.kl_divergence(q_mu, q_logvar, p_mu, p_logvar)
            loss = recon_loss.mean() + beta * kl.mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step(); total += float(loss.detach()); steps += 1
        history.append(total / max(steps, 1))
    return model, history


class VAEACGenerator:
    """Frozen Q0 full/conditional sampler backed by canonical VAEAC."""
    def __init__(self, model, scale=None, alpha=1.0, device="cuda", feature_fn=None):
        self._legacy_reference = None
        if isinstance(model, np.ndarray):
            self._legacy_reference = np.asarray(model, dtype=float)
            self.model, self.device = None, "cpu"
            self.scale = np.ones(self._legacy_reference.shape[1]) if scale is None else np.asarray(scale, float)
            self.alpha = float(alpha)
            self.feature_fn = feature_map if feature_fn is None else feature_fn
            self.kind = "legacy_reference"
            return
        self.model = model.eval().to(device)
        self.device, self.scale, self.alpha = device, np.asarray(scale if scale is not None else np.ones(model.dim)), float(alpha)
        self.feature_fn = (lambda x: feature_map(x, self.scale)) if feature_fn is None else feature_fn
        self.kind = "vaeac"
        self.phi_net = None
        self.phi_net_step1 = None
        self.phi_net_zero = None
        self.phi_info_scale = None
        self.phi_info_scale_zero = None
        self.phi_info_mix_a = None

    @property
    def dimension(self):
        return self.model.dim if self.model is not None else self._legacy_reference.shape[1]

    def _from_model_space(self, x):
        return x * np.asarray(self.model.data_std) + np.asarray(self.model.data_mean)

    def _prior_sample(self, obs, mask, n, seed):
        obs, mask = obs.expand(n, -1), mask.expand(n, -1)
        mu, logvar = self.model.prior(obs, mask)
        z = self.model.reparameterize(mu, logvar, seed)
        out_mu, out_logvar = self.model.decode_params(z, mask, obs)
        gen = torch.Generator(device=self.device).manual_seed(seed + 1)
        return out_mu + torch.exp(0.5 * out_logvar) * torch.randn(out_mu.shape, device=self.device, generator=gen)

    def sample_full(self, n, seed=0):
        if self._legacy_reference is not None:
            rng = np.random.default_rng(seed)
            return self._legacy_reference[rng.integers(len(self._legacy_reference), size=n)].copy()
        mask = torch.zeros(1, self.dimension, device=self.device)
        with torch.no_grad():
            return self._from_model_space(self._prior_sample(torch.zeros_like(mask), mask, n, seed).cpu().numpy())

    def sample_conditional(self, observed, panel, n, seed=0):
        if self._legacy_reference is not None:
            rng = np.random.default_rng(seed)
            out = self._legacy_reference[rng.integers(len(self._legacy_reference), size=n)].copy()
            out[:, list(panel)] = np.asarray(observed); return out
        center, spread = np.asarray(self.model.data_mean), np.asarray(self.model.data_std)
        obs = torch.zeros(1, self.dimension, device=self.device); mask = torch.zeros_like(obs)
        obs[:, list(panel)] = torch.as_tensor((np.asarray(observed) - center[list(panel)]) / spread[list(panel)], dtype=torch.float32, device=self.device)
        mask[:, list(panel)] = 1.0
        with torch.no_grad():
            out = self._from_model_space(self._prior_sample(obs, mask, n, seed).cpu().numpy())
        out[:, list(panel)] = np.asarray(observed); return out

    def sample_conditional_batch(self, observed_batch, panel, n, seed=0):
        observed = np.atleast_2d(np.asarray(observed_batch, dtype=float))
        if self._legacy_reference is not None:
            return np.stack([self.sample_conditional(row, panel, n, seed + i) for i, row in enumerate(observed)])
        rows = len(observed); center = np.asarray(self.model.data_mean); spread = np.asarray(self.model.data_std)
        obs = torch.zeros(rows, self.dimension, device=self.device)
        obs[:, list(panel)] = torch.as_tensor((observed - center[list(panel)]) / spread[list(panel)], dtype=torch.float32, device=self.device)
        mask = torch.zeros_like(obs); mask[:, list(panel)] = 1.0
        obs = obs[:, None, :].expand(rows, n, -1).reshape(rows * n, -1)
        mask = mask[:, None, :].expand(rows, n, -1).reshape(rows * n, -1)
        with torch.no_grad():
            mu, logvar = self.model.prior(obs, mask)
            z = self.model.reparameterize(mu, logvar, seed)
            out_mu, out_logvar = self.model.decode_params(z, mask, obs)
            gen = torch.Generator(device=self.device).manual_seed(seed + 1)
            out = out_mu + torch.exp(0.5 * out_logvar) * torch.randn(out_mu.shape, device=self.device, generator=gen)
        out = self._from_model_space(out.cpu().numpy()).reshape(rows, n, self.dimension)
        out[:, :, list(panel)] = observed[:, None, :]
        return out

    def tilted_full_sample(self, beta, n, seed=0):
        rng, accepted = np.random.default_rng(seed), []
        envelope = float(np.sum(np.abs(np.asarray(beta))))
        rounds = 0
        while len(accepted) < n:
            rounds += 1
            if rounds > 10000:
                raise RuntimeError("VAEAC full tilt acceptance collapsed; run is anomalous")
            proposal = self.sample_full(max(256, 4 * (n - len(accepted))), int(rng.integers(2**31 - 1)))
            accepted.extend(proposal[rng.random(len(proposal)) < np.exp(self.feature_fn(proposal) @ beta - envelope)])
        return np.asarray(accepted[:n])

    def tilted_full_diagnostics(self, beta, n, seed=0):
        rng, accepted, proposals = np.random.default_rng(seed), [], 0
        envelope = float(np.sum(np.abs(np.asarray(beta))))
        rounds = 0
        while len(accepted) < n:
            rounds += 1
            if rounds > 10000:
                raise RuntimeError("VAEAC full tilt acceptance collapsed; run is anomalous")
            proposal = self.sample_full(max(256, 4 * (n - len(accepted))), int(rng.integers(2**31 - 1)))
            keep = rng.random(len(proposal)) < np.exp(self.feature_fn(proposal) @ beta - envelope)
            proposals += len(proposal); accepted.extend(proposal[keep])
        return np.asarray(accepted[:n]), float(len(accepted) / proposals), self.importance_ess(beta, np.asarray(accepted[:n]))

    def tilted_conditional_diagnostics(self, beta, observed, panel, n, seed=0):
        rng, accepted, proposals = np.random.default_rng(seed), [], 0
        envelope = float(np.sum(np.abs(np.asarray(beta))))
        rounds = 0
        while len(accepted) < n:
            rounds += 1
            if rounds > 10000:
                raise RuntimeError("VAEAC conditional tilt acceptance collapsed; run is anomalous")
            proposal = self.sample_conditional(observed, panel, max(32, 2 * (n - len(accepted))), int(rng.integers(2**31 - 1)))
            keep = rng.random(len(proposal)) < np.exp(self.feature_fn(proposal) @ beta - envelope)
            proposals += len(proposal); accepted.extend(proposal[keep])
        out = np.asarray(accepted[:n])
        return out, float(len(accepted) / proposals), self.importance_ess(beta, out)

    def tilted_conditional_sample(self, beta, observed, panel, n, seed=0):
        rng, accepted = np.random.default_rng(seed), []
        envelope = float(np.sum(np.abs(np.asarray(beta))))
        rounds = 0
        while len(accepted) < n:
            rounds += 1
            if rounds > 10000:
                raise RuntimeError("VAEAC conditional tilt acceptance collapsed; run is anomalous")
            proposal = self.sample_conditional(observed, panel, max(32, 2 * (n - len(accepted))), int(rng.integers(2**31 - 1)))
            accepted.extend(proposal[rng.random(len(proposal)) < np.exp(self.feature_fn(proposal) @ beta - envelope)])
        return np.asarray(accepted[:n])

    def tilted_sample(self, beta, feature_fn, n, seed=0):
        """Legacy diagnostic alias retained for the P0-P5 smoke tests."""
        old = self.feature_fn
        self.feature_fn = feature_fn
        try:
            samples = self.tilted_full_sample(beta, n, seed)
        finally:
            self.feature_fn = old
        logits = feature_fn(samples) @ np.asarray(beta)
        logits -= logits.max()
        weights = np.exp(logits)
        ess = float(weights.sum() ** 2 / np.sum(weights ** 2) / len(weights))
        return samples, 1.0, ess

    def importance_ess(self, beta, pool, pool_weights=None):
        logits = self.feature_fn(pool) @ np.asarray(beta)
        if pool_weights is not None: logits += np.log(np.asarray(pool_weights) + 1e-300)
        logits -= logits.max(); weights = np.exp(logits)
        return float(weights.sum() ** 2 / np.sum(weights ** 2) / len(weights))

    def panel_information(self, beta, panels, outer_x, n_conditional=32, seed=0, feature_fn=None, neighbors=None, estimator="imp"):
        """I_S on the same outer queries as the Gas kNN RISK-2 reference.

        ``learned_information`` draws a fresh tilted pool from the generator.
        That mixes Monte Carlo noise into panel ranking.  Matching the empirical
        operator uses the frozen outer table and only asks the network for
        E[phi | x_S].
        """
        del neighbors
        from .empirical_generator import _project_psd, _stable_tilt_weights
        from .generator_validation import weighted_covariance

        fn = self.feature_fn if feature_fn is None else feature_fn
        outer = np.asarray(outer_x, dtype=float)
        outer_phi = fn(outer)
        outer_w = _stable_tilt_weights(outer_phi, np.asarray(beta, dtype=float))
        fisher = _project_psd(weighted_covariance(outer_phi, outer_w))
        means, infos = [], []
        kind = str(estimator or "imp").strip().lower()
        if kind in {"auto", "default", ""}:
            kind = "phi_net" if self.phi_net is not None else "imp"
        use_zero = (
            kind in {"phi_net", "masked_mlp", "panel_embed"}
            and self.phi_net_zero is not None
            and float(np.linalg.norm(np.asarray(beta, dtype=float))) <= 1e-12
        )
        if use_zero:
            saved = self.phi_net
            self.phi_net = self.phi_net_zero
        try:
            for index, panel in enumerate(panels):
                if kind in {"phi_net", "masked_mlp", "panel_embed"} and self.phi_net is not None:
                    conditional = self.conditional_mean_phi_net(outer[:, list(panel)], panel, index)
                elif kind in {"prior_decode", "decode", "mean_field"}:
                    conditional = self.conditional_mean_prior_decode(outer[:, list(panel)], panel, feature_fn=fn)
                else:
                    conditional = imp_conditional_mean_from_generator(
                        self,
                        beta,
                        outer[:, list(panel)],
                        panel,
                        n=int(n_conditional),
                        seed=int(seed) + int(index),
                        feature_fn=fn,
                    )
                means.append(conditional)
                infos.append(_project_psd(weighted_covariance(conditional, outer_w)))
        finally:
            if use_zero:
                self.phi_net = saved
        infos = np.asarray(infos)
        mix_kind = kind in {"phi_net", "masked_mlp", "panel_embed"}
        if use_zero:
            return fisher, infos, np.asarray(means)
        if mix_kind and self.phi_info_mix_a is not None and self.phi_net_step1 is not None:
            infos1 = []
            saved = self.phi_net
            self.phi_net = self.phi_net_step1
            try:
                for index, panel in enumerate(panels):
                    conditional = self.conditional_mean_phi_net(outer[:, list(panel)], panel, index)
                    infos1.append(_project_psd(weighted_covariance(conditional, outer_w)))
            finally:
                self.phi_net = saved
            infos1 = np.asarray(infos1)
            if self.phi_info_scale is not None:
                scale = np.asarray(self.phi_info_scale, dtype=float).reshape(-1)
                if len(scale) != len(infos1):
                    raise ValueError("phi_info_scale length must match the panel library")
                infos1 = np.stack([_project_psd(info * float(factor)) for info, factor in zip(infos1, scale)])
            mix_a = float(self.phi_info_mix_a)
            infos = np.stack([_project_psd((1.0 - mix_a) * i0 + mix_a * i1) for i0, i1 in zip(infos, infos1)])
        elif self.phi_info_scale is not None and mix_kind:
            scale = np.asarray(self.phi_info_scale, dtype=float).reshape(-1)
            if len(scale) != len(infos):
                raise ValueError("phi_info_scale length must match the panel library")
            infos = np.stack([_project_psd(info * float(factor)) for info, factor in zip(infos, scale)])
        if (
            mix_kind
            and self.phi_info_scale_zero is not None
            and float(np.linalg.norm(np.asarray(beta, dtype=float))) <= 1e-12
        ):
            scale0 = np.asarray(self.phi_info_scale_zero, dtype=float).reshape(-1)
            if len(scale0) != len(infos):
                raise ValueError("phi_info_scale_zero length must match the panel library")
            infos = np.stack([_project_psd(info * float(factor)) for info, factor in zip(infos, scale0)])
        return fisher, infos, np.asarray(means)

    def conditional_mean_phi_net(self, observed_batch, panel, panel_index):
        observed = np.atleast_2d(np.asarray(observed_batch, dtype=float))
        center = np.asarray(self.model.data_mean, dtype=np.float64)
        spread = np.asarray(self.model.data_std, dtype=np.float64)
        cols = list(panel)
        scaled = (observed - center[cols]) / np.maximum(spread[cols], 1e-12)
        obs = torch.as_tensor(scaled, dtype=torch.float32, device=self.device)
        ids = torch.full((len(obs),), int(panel_index), dtype=torch.long, device=self.device)
        with torch.no_grad():
            return self.phi_net(obs, ids).cpu().numpy()

    def attach_phi_net(self, state, hidden=256, kind="panel_embed", n_panels=120):
        del kind
        net = PanelPhiNet(n_panels=int(n_panels), hidden=int(hidden), out=16).to(self.device)
        net.load_state_dict(state)
        net.eval()
        self.phi_net = net
        return self

    def attach_phi_info_scale(self, scale):
        values = np.asarray(scale, dtype=np.float64).reshape(-1)
        if np.any(values <= 0) or not np.isfinite(values).all():
            raise ValueError("phi_info_scale must be positive and finite")
        self.phi_info_scale = values
        return self

    def attach_phi_net_step1(self, state, hidden=256, n_panels=120):
        net = PanelPhiNet(n_panels=int(n_panels), hidden=int(hidden), out=16).to(self.device)
        net.load_state_dict(state)
        net.eval()
        self.phi_net_step1 = net
        return self

    def attach_phi_net_zero(self, state, hidden=256, n_panels=120):
        net = PanelPhiNet(n_panels=int(n_panels), hidden=int(hidden), out=16).to(self.device)
        net.load_state_dict(state)
        net.eval()
        self.phi_net_zero = net
        return self

    def attach_phi_info_mix_a(self, mix_a):
        value = float(mix_a)
        if not (0.0 <= value <= 1.0) or not np.isfinite(value):
            raise ValueError("phi_info_mix_a must be finite and in [0, 1]")
        self.phi_info_mix_a = value
        return self

    def attach_phi_info_scale_zero(self, scale):
        values = np.asarray(scale, dtype=np.float64).reshape(-1)
        if np.any(values <= 0) or not np.isfinite(values).all():
            raise ValueError("phi_info_scale_zero must be positive and finite")
        self.phi_info_scale_zero = values
        return self

    def conditional_mean_prior_decode(self, observed_batch, panel, feature_fn=None):
        """E[phi | x_S] from the conditional-prior mean decode (no IMP noise)."""
        fn = self.feature_fn if feature_fn is None else feature_fn
        observed = np.atleast_2d(np.asarray(observed_batch, dtype=float))
        center = np.asarray(self.model.data_mean, dtype=np.float64)
        spread = np.asarray(self.model.data_std, dtype=np.float64)
        cols = list(panel)
        obs = torch.zeros(len(observed), self.dimension, device=self.device)
        mask = torch.zeros_like(obs)
        scaled = (observed - center[cols]) / np.maximum(spread[cols], 1e-12)
        obs[:, cols] = torch.as_tensor(scaled, dtype=torch.float32, device=self.device)
        mask[:, cols] = 1.0
        with torch.no_grad():
            p_mu, _ = self.model.prior(obs, mask)
            recon, _ = self.model.decode_params(p_mu, mask, obs)
            mixed = (mask * obs + (1.0 - mask) * recon).cpu().numpy()
        raw = mixed * spread + center
        raw[:, cols] = observed
        return fn(raw)


def imp_conditional_mean_from_generator(generator, beta, batch, panel, n, seed, scale=None, feature_fn=None):
    rng = np.random.default_rng(seed)
    comp = generator.sample_conditional_batch(batch, panel, n, int(rng.integers(2**31 - 1)))
    fn = generator.feature_fn if feature_fn is None else feature_fn
    phi = fn(comp); logits = phi @ np.asarray(beta); logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return np.einsum("onr,on->or", phi, weights) / weights.sum(axis=1, keepdims=True)


def learned_information(generator, beta, panels, n_tilted=256, n_conditional=32, seed=0, feature_fn=None):
    rng = np.random.default_rng(seed)
    tilted = generator.tilted_full_sample(beta, n_tilted, int(rng.integers(2**31 - 1)))
    fn = generator.feature_fn if feature_fn is None else feature_fn; phi = fn(tilted); mu = phi.mean(0)
    fisher, infos = np.cov(phi, rowvar=False), []
    for panel in panels:
        observed = tilted[:, list(panel)]
        a = imp_conditional_mean_from_generator(generator, beta, observed, panel, n_conditional, seed + 1, feature_fn=feature_fn) - mu
        b = imp_conditional_mean_from_generator(generator, beta, observed, panel, n_conditional, seed + 2, feature_fn=feature_fn) - mu
        a -= a.mean(0); b -= b.mean(0); info = (a.T @ b + b.T @ a) / max(2 * (len(a) - 1), 1)
        vals, vecs = np.linalg.eigh((info + info.T) / 2); infos.append((vecs * np.maximum(vals, 1e-10)) @ vecs.T)
    return fisher, np.asarray(infos)
