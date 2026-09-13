"""Stage paper figure inputs from the audited H-drive runs.

The old ``results/paper`` tree predates the numerical calibration and is not
admissible evidence.  This script assembles a fresh staging tree from the
stages that carry an independent PASS audit, records the SHA-256 of every
source file, and refuses to include a stage whose audit is missing or not
``PASS``.

Nothing on H: is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

# Audit documents are prose: they mention superseded rejected versions, so the
# verdict has to be read off the bolded verdict marker, not from a keyword scan.
_PASS_RE = re.compile(
    r"(?:\*\*\s*(?:PASS|ACCEPTED|PASS；))"
    r"|(?:(?:verdict|result|decision)\s*:?\s*(?:\*\*\s*)?(?:PASS|ACCEPTED))",
    re.IGNORECASE,
)
_REJECT_RE = re.compile(
    r"(?:\*\*\s*(?:REJECT|REJECTED|REJECT；))"
    r"|(?:(?:verdict|result|decision)\s*:?\s*(?:\*\*\s*)?(?:REJECT|REJECTED))",
    re.IGNORECASE,
)


RUNS = Path("H:/RR_GID_CN_data/active/paper_runs")

# stage name -> (source rows path, audit file to require PASS in)
SOURCES = {
    "synthetic_main": [
        (RUNS / "synthetic_main_b2000_v2/rows.jsonl", RUNS / "synthetic_main_b2000_v2/AUDIT_B2000_v2.md"),
        (RUNS / "synthetic_main_b4000_v1/rows.jsonl", RUNS / "synthetic_main_b4000_v1/AUDIT_B4000_v1.md"),
        (RUNS / "synthetic_main_b8000_v1/rows.jsonl", RUNS / "synthetic_main_b8000_v1/AUDIT_B8000_v1_REAUDIT_20260910.md"),
        (RUNS / "synthetic_main_b16000_v2/rows.jsonl", RUNS / "synthetic_main_b16000_v2/AUDIT_B16000_v2.md"),
        (RUNS / "synthetic_main_b32000_v1/rows.jsonl", RUNS / "synthetic_main_b32000_v1/AUDIT_B32000_v1.md"),
    ],
    "nonlinearity": [
        (RUNS / "nonlinearity_v1/rows.jsonl", RUNS / "nonlinearity_v1/AUDIT_nonlinearity_v1.md"),
    ],
    "reuse": [
        (RUNS / "reuse_v2/rows.jsonl", RUNS / "reuse_v2/AUDIT_reuse_v2.md"),
    ],
    "gas_semisynthetic": [
        (RUNS / "gas_semisynthetic_B400_v3/results/rows.jsonl", RUNS / "gas_semisynthetic_B400_v3/AUDIT_B400_v3.md"),
        (RUNS / "gas_semisynthetic_B800_v1/results/rows.jsonl", RUNS / "gas_semisynthetic_B800_v1/AUDIT_B800_v1.md"),
        (RUNS / "gas_semisynthetic_B1600_v1/results/rows.jsonl", RUNS / "gas_semisynthetic_B1600_v1/AUDIT_B1600_v1.md"),
        (RUNS / "gas_semisynthetic_B3200_v1/results/rows.jsonl", RUNS / "gas_semisynthetic_B3200_v1/AUDIT_B3200_v1.md"),
    ],
    "gas_natural": [
        (RUNS / "gas_natural_v1_clean/rows.jsonl", RUNS / "gas_natural_v1_clean/AUDIT_P10_v1_clean.md"),
    ],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/paper_audited_20260910"))
    args = ap.parse_args()
    out_root: Path = args.out

    manifest = {"schema_version": "paper-staging-20260910-v1", "stages": {}}
    for stage, entries in SOURCES.items():
        rows: list[str] = []
        provenance = []
        for rows_path, audit_path in entries:
            if not rows_path.exists():
                raise FileNotFoundError(f"missing rows: {rows_path}")
            if not audit_path.exists():
                raise FileNotFoundError(f"missing audit: {audit_path}")
            audit_text = audit_path.read_text(encoding="utf-8")
            accepted = _PASS_RE.search(audit_text)
            rejected = _REJECT_RE.search(audit_text)
            if not accepted or rejected:
                raise RuntimeError(f"audit is not a clean PASS: {audit_path}")
            text = rows_path.read_text(encoding="utf-8")
            kept = [line for line in text.splitlines() if line.strip()]
            rows.extend(kept)
            provenance.append({
                "rows": str(rows_path),
                "rows_sha256": _sha256(rows_path),
                "audit": str(audit_path),
                "rows_in_file": len(kept),
            })
        target = out_root / stage
        target.mkdir(parents=True, exist_ok=True)
        payload = "".join(line + "\n" for line in rows)
        (target / "rows.jsonl").write_text(payload, encoding="utf-8")
        manifest["stages"][stage] = {
            "row_count": len(rows),
            "rows_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "sources": provenance,
        }
        print(f"{stage}: {len(rows)} rows from {len(entries)} audited file(s)")

    (out_root / "staging_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out_root / 'staging_manifest.json'}")


if __name__ == "__main__":
    main()
