---
license: mit
pretty_name: RR-GID-CN paper results
task_categories:
  - other
tags:
  - experimental-design
  - information-geometry
  - qmc
---

# RR-GID-CN paper results (2026-09-13 freeze)

Audited jsonl, figures, tables, and calibration reports for the paper experiments in [15628925702/RR_GID_CN](https://github.com/15628925702/RR_GID_CN).

Code and configs live on GitHub. This dataset is the **number source** for E1–E5. It does **not** include QMC spill caches (tens to hundreds of GB of `basis_*.dat`). Those are resume-only.

Related empty placeholder (do not use for numbers): [`Guaogua/RR-data`](https://huggingface.co/datasets/Guaogua/RR-data).

## Layout

| path | what |
|---|---|
| `e1_synthetic_main/rows.jsonl` | Figure 1, 4000 rows, 200 paired replications |
| `e2_nonlinearity/rows.jsonl` | Figure 2(a), 240 rows, 15 replications |
| `e3_reuse/rows.jsonl` | Figure 2(b), 500 rows, 5 sequences × 50 campaigns |
| `e5_r1_semisynthetic/rows.jsonl` | Figure 3 R1, 320 rows, 20 replications |
| `e5_r2_natural/rows.jsonl` | Figure 3 R2, 720 rows, 20 replications |
| `figures/` | Fig.1–3 pdf/png |
| `paper_tables/` | Table 1a (B=32000, n=200) and Table 1b (B=3200, n=20) |
| `risk_gates/` | E4 query/info/e2e reports, RISK-2/3, fixed↔cached certificate |
| `checkpoints/empirical_knn_gas_v1.json` | Frozen empirical kNN handle used for Gas |

Every `rows.jsonl` has a matching `AUDIT_*.json` with `passed: true` and SHA256.

Primary synthetic metric is `primary_loss` = `risk_ratio_raw`. Gas metric is family-projection loss in `primary_loss`.

Do not use older `n=20` B·KL tables. Table 1a is 200-rep R_risk.
