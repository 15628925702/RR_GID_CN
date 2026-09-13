from pathlib import Path

import numpy as np
import yaml

from scripts.p10_r2_formal import (bregman_projection_loss, heldout_function_values,
                                   solve_acquisition_beta)


def test_p10_config_exists():
    path = Path("configs/paper/gas_natural.yaml")
    assert path.exists()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["experiment_name"] == "gas_natural"
    assert config["experiment_mode"] == "bulk"
    assert config["campaigns"] == ["batch7", "batches8_9", "batch10"]
    assert config["score_backend"] == "cached_qmc"
    assert config["information_backend"] == "cached_qmc_cross"
    assert config["information_scrambles"] >= 2


def test_projection_loss_is_bregman_and_nonnegative():
    rng = np.random.default_rng(4)
    phi = rng.normal(size=(400, 16))
    beta_dag = rng.normal(size=16) * 0.1
    beta_hat = beta_dag + rng.normal(size=16) * 0.2
    loss = bregman_projection_loss(phi, beta_hat, beta_dag)
    assert loss >= 0.0
    assert bregman_projection_loss(phi, beta_dag, beta_dag) == 0.0


def test_heldout_function_dictionary_has_fixed_32_functions():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(9, 128))
    mean, std = np.zeros(128), np.ones(128)
    pcs2 = np.zeros((16, 8)); pcs2[:, 1] = 1.0
    values = heldout_function_values(x, mean, std, pcs2)
    assert values.shape == (9, 32)
    assert np.all(np.abs(values) <= 1.0)


def test_acquisition_beta_uses_only_campaign_pool_observations():
    rng = np.random.default_rng(6)
    pool_x = rng.normal(size=(80, 4))
    pool_phi = np.tanh(pool_x)
    beta = solve_acquisition_beta(pool_x, pool_phi, [(tuple([0, 1]), pool_x[0, [0, 1]])])
    assert beta.shape == (4,)
    assert np.all(np.isfinite(beta))
