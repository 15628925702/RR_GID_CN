"""Register the already-trained canonical Synthetic P6 VAEAC checkpoint."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rr_gid_cn.vaeac import load_vaeac_checkpoint


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    checkpoint = Path(r"H:\RR_GID_CN_data\reusable\checkpoints\vaeac_reference.pt")
    output = Path(r"H:\RR_GID_CN_data\active\paper_runs\P6_MANIFEST_v1.json")
    train_log = Path(r"H:\RR_GID_CN_data\reusable\logs\vaeac_reference_train.json")
    model, payload = load_vaeac_checkpoint(str(checkpoint), device="cpu", expected_dim=16)
    finite_state = all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())
    finite_stats = bool(np.isfinite(model.data_mean).all() and np.isfinite(model.data_std).all())
    if not finite_state or not finite_stats:
        raise RuntimeError("P6 checkpoint contains non-finite parameters or standardisation statistics")
    result = {
        "schema_version": "stage-manifest-v1",
        "stage": "P6",
        "status": "completed",
        "evidence_level": "formal_asset",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_keys": sorted(payload.keys()),
        "dimension": int(model.dim),
        "latent": int(model.latent),
        "state_finite": finite_state,
        "standardisation_stats_finite": finite_stats,
        "load_device": "cpu",
        "training_log": str(train_log),
        "training_log_sha256": sha256(train_log) if train_log.exists() else None,
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quality_warning": "Training diagnostic reports large raw sample reconstruction errors; this manifest records only the requested load/dimension/finite-value asset gate.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
