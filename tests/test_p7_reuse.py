from pathlib import Path

import yaml


def test_reuse_manifest_schema():
    path = Path("configs/paper/reuse.yaml")
    assert path.exists()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["experiment_name"] == "reuse"
    assert config["experiment_mode"] == "bulk"
    assert config["methods"] == ["RR-GID", "Discriminative Score OED"]
    assert config["campaigns"] == [1, 5, 10, 20]
    assert config["score_backend"] == "cached_qmc"
    assert config["information_backend"] == "cached_qmc_cross"
