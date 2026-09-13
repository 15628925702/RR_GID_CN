from pathlib import Path

import yaml


def test_p9_config_exists():
    path = Path("configs/paper/gas_semisynthetic.yaml")
    assert path.exists()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["experiment_name"] == "gas_semisynthetic"
    assert config["experiment_mode"] == "bulk"
    assert config["budgets"] == [400, 800, 1600, 3200]
    assert config["score_backend"] == "cached_qmc"
    assert config["information_backend"] == "cached_qmc_cross"
