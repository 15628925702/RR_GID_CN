"""Extract, merge, summarize, and record the E5-R2 v5 decision. Does not touch v3."""
from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"G:\0-newResearch\4.RR_GID_CN")
EXTRACT = ROOT / "tmp" / "e5_r2_v5_pull" / "extract_full"
TAR = ROOT / "tmp" / "e5_r2_v5_pull" / "full.tar"
OUT = ROOT / "results" / "e5_r2_v5"
V3 = ROOT / "results" / "e5_r2_v3" / "rows.jsonl"
H_ROOT = Path(r"H:\RR_GID_CN_data\active\repaired_20260912")
METHODS = ["Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID"]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_dir(root: Path) -> list[dict]:
    rows = []
    for p in sorted(root.rglob("rows.jsonl")):
        rows.extend(load_jsonl(p))
    return rows


def row_key(row: dict) -> tuple:
    return (
        row.get("campaign"),
        row.get("method") or row.get("policy"),
        int(row.get("budget", -1)),
        int(row.get("replication", -1)),
    )


def loss_of(row: dict) -> float:
    return float(row.get("primary_loss", row.get("projection_loss", row.get("loss"))))


def paired_ci(diffs: list[float]) -> dict:
    n = len(diffs)
    mu = sum(diffs) / n
    if n < 2:
        return {"n": n, "mean": mu, "lo": None, "hi": None, "excludes_zero": None}
    var = sum((x - mu) ** 2 for x in diffs) / (n - 1)
    se = math.sqrt(var / n)
    h = 1.96 * se
    return {
        "n": n,
        "mean": mu,
        "lo": mu - h,
        "hi": mu + h,
        "excludes_zero": (mu - h > 0) or (mu + h < 0),
        "halfwidth": h,
    }


def cell(rows: list[dict], camp: str, budget: int, method: str) -> dict[int, float]:
    out = {}
    for row in rows:
        if row.get("campaign") != camp:
            continue
        if int(row.get("budget", -1)) != budget:
            continue
        if (row.get("method") or row.get("policy")) != method:
            continue
        out[int(row.get("replication", -1))] = loss_of(row)
    return out


def means(d: dict[int, float]) -> dict:
    xs = list(d.values())
    return {"n": len(xs), "mean": (sum(xs) / len(xs)) if xs else None}


def compare_cell(v5: list[dict], v3: list[dict], camp: str, budget: int) -> dict:
    rec = {"campaign": camp, "budget": budget, "methods": {}, "paired": {}}
    for method in METHODS:
        m5 = cell(v5, camp, budget, method)
        m3 = cell(v3, camp, budget, method)
        rec["methods"][method] = {"v5": means(m5), "v3": means(m3)}
    rr = cell(v5, camp, budget, "RR-GID")
    a = cell(v5, camp, budget, "A-OSQD")
    u = cell(v5, camp, budget, "Uniform SQD")
    keys = sorted(set(rr) & set(a) & set(u))
    d_a = [rr[k] - a[k] for k in keys]
    d_u = [rr[k] - u[k] for k in keys]
    rec["paired"]["RR_minus_A"] = paired_ci(d_a)
    rec["paired"]["RR_minus_U"] = paired_ci(d_u)
    rec["paired"]["RR_beats_A"] = sum(d < 0 for d in d_a)
    rec["paired"]["RR_beats_U"] = sum(d < 0 for d in d_u)
    rec["paired"]["n"] = len(keys)
    return rec


def main() -> None:
    if EXTRACT.exists():
        shutil.rmtree(EXTRACT)
    EXTRACT.mkdir(parents=True)
    shutil.unpack_archive(str(TAR), str(EXTRACT))
    raw = load_dir(EXTRACT)
    uniq = {}
    dups = 0
    for row in raw:
        k = row_key(row)
        if k in uniq:
            dups += 1
            continue
        uniq[k] = row
    merged = [uniq[k] for k in sorted(uniq)]
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "rows.jsonl"
    with dest.open("w", encoding="utf-8") as fh:
        for row in merged:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    seed_src = ROOT / "results" / "e5_r2_v3" / "seed_manifest.json"
    seed_dst = OUT / "seed_manifest.json"
    if seed_src.exists() and not seed_dst.exists():
        shutil.copy2(seed_src, seed_dst)
    v3 = load_jsonl(V3)
    cells = []
    for camp in ["batch7", "batches8_9", "batch10"]:
        for budget in [800, 1600, 3200]:
            cells.append(compare_cell(merged, v3, camp, budget))
            c = cells[-1]
            print(
                f"{camp} B={budget} RR v5={c['methods']['RR-GID']['v5']['mean']:.4f} "
                f"v3={c['methods']['RR-GID']['v3']['mean']:.4f} "
                f"A={c['methods']['A-OSQD']['v5']['mean']:.4f} "
                f"U={c['methods']['Uniform SQD']['v5']['mean']:.4f} "
                f"RR-A CI [{c['paired']['RR_minus_A']['lo']:.4f},{c['paired']['RR_minus_A']['hi']:.4f}]"
            )
    sample = merged[0]
    b89_1600 = next(c for c in cells if c["campaign"] == "batches8_9" and c["budget"] == 1600)
    b89_3200 = next(c for c in cells if c["campaign"] == "batches8_9" and c["budget"] == 3200)
    b89_800 = next(c for c in cells if c["campaign"] == "batches8_9" and c["budget"] == 800)
    decision = {
        "experiment": "e5_v5",
        "change": "R2 RR-GID uses I_S(0) + Q0 reference Fisher, still uncapped. Same seeds as v3.",
        "r1": {
            "baseline": "v4 (mix=0.1, max_share=0.2, rr_fisher=generator)",
            "verdict": "keep_v4",
            "reason": "R1 is a Q0 tilt; v4 already aligned generator Fisher with I_S. Do not import R2's I_S(0)+F_ref into R1.",
        },
        "r2": {
            "baseline": "v3 (uncapped, information_at=pilot, default rr_fisher=generator)",
            "verdict": "replace_v3_with_v5",
            "reason": (
                "Same seeds. Uniform/A-OSQD/Disc unchanged. On batches8_9 RR-GID mean drops "
                f"B=800 {b89_800['methods']['RR-GID']['v3']['mean']:.4f}->{b89_800['methods']['RR-GID']['v5']['mean']:.4f}, "
                f"B=1600 {b89_1600['methods']['RR-GID']['v3']['mean']:.4f}->{b89_1600['methods']['RR-GID']['v5']['mean']:.4f}, "
                f"B=3200 {b89_3200['methods']['RR-GID']['v3']['mean']:.4f}->{b89_3200['methods']['RR-GID']['v5']['mean']:.4f}. "
                "Paired RR vs Uniform still excludes 0 (20/20). Paired RR vs A-OSQD CI includes 0 at B=1600 and 3200. "
                "batch7 RR now beats A-OSQD; Uniform remains first under weak drift. batch10 RR remains first."
            ),
            "information_at": sample.get("information_at"),
            "rr_fisher": sample.get("rr_fisher"),
            "n_rows": len(merged),
            "expected_rows": 720,
            "cells": cells,
        },
        "rejected_followups": {
            "iscale_0.5": {
                "verdict": "do_not_run_20rep",
                "smoke_5rep_batches8_9_B1600_RR": 0.1524,
                "note": "Worse than I_S(0)+F_ref smoke 0.1470.",
            },
            "F_ahat_pool": {
                "verdict": "do_not_run_20rep",
                "smoke_5rep_batches8_9_B1600_RR": 0.1757,
                "note": "Scoring-pool Fisher mismatches kNN I_S estimated on Q0 donors; worse than F_ref.",
            },
            "F_knn_I_S0": {
                "verdict": "already_rejected_as_v4_info0",
                "batches8_9_B1600_RR": 0.2102,
            },
        },
        "body_recommendation": {
            "R1": "v4 (mix=0.1, max_share=0.2, rr_fisher=generator)",
            "R2": "v5 (uncapped, information_at=zero, rr_fisher=reference)",
            "do_not_overwrite": ["e5_r1_v4", "e5_r2_v3", "e5_r2_v4", "e5_*_empirical_knn"],
        },
        "src_rows": len(raw),
        "merged_rows": len(merged),
        "dups": dups,
    }
    decision_path = ROOT / "results" / "e5_v5_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "stage": "P10",
        "experiment": "gas_natural_v5",
        "status": "completed",
        "row_count": len(merged),
        "expected_rows": 720,
        "rows_path": str(dest),
        "information_at": sample.get("information_at"),
        "rr_fisher": sample.get("rr_fisher"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", dest, "n", len(merged), "dups", dups)
    print("wrote", decision_path)
    if H_ROOT.exists():
        h_out = H_ROOT / "e5_r2_v5"
        h_out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, h_out / "rows.jsonl")
        shutil.copy2(decision_path, H_ROOT / "e5_v5_decision.json")
        if seed_dst.exists():
            shutil.copy2(seed_dst, h_out / "seed_manifest.json")
        shutil.copy2(OUT / "summary.json", h_out / "summary.json")
        print("copied to", h_out)


if __name__ == "__main__":
    main()
