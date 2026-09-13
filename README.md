# RR-GID-CN

固定预算下的 Reference-Relative Generative Information Design。论文实验已于 2026-09-13 定稿。

执行依据：`核心文档/RR_GID_实验重规划与代码改造执行指南_20260902.md`。  
G0–G4 诊断阶梯已归档，不再进入论文实验。诊断快照在独立仓库 https://github.com/15628925702/RR 。

论文只回答三个问题：固定预算下 RR-GID 是否降低完整 target 估计误差；优势是否来自 nonlinear conditional information；frozen generator 能否跨 campaign 复用。

## 结论（写论文用）

- **E1 合成主实验（充分）**：200 配对 replication，B=32000 时 RR-GID 的 R_risk 为 1.876 vs Uniform 3.346，相对改善 43.9%，配对 CI [1.229, 1.717] 不含 0。A-OSQD 与 Discriminative Score OED 在该格子上差于 Uniform。
- **E2 非线性（充分）**：α≤1 时 RR-GID 约 0.93–1.04；α=1.5 为 1.80，仍低于 Uniform 2.78。A-OSQD 崩到 6.26。
- **E3 复用（弱充分）**：T=50 时 RR-GID 1.06 vs Disc 1.21，但 n=5，只写趋势。
- **E5 Gas（部分充分）**：生成器是冻结经验 kNN，不是 VAEAC。R1 四方法不可分；R2 相对 Uniform 在多数大预算更低。

完整步骤、表格、与 2026-08-20 原稿的对照见：

- `核心文档/RR_GID_论文实验完整总结_20260913.md`
- `核心文档/RR_GID_论文实验完整总结_20260913.pdf`
- 数据：https://huggingface.co/datasets/Guaogua/RR-GID-CN-paper-results
- 代码：https://github.com/15628925702/RR_GID_CN （分支 `refactor/paper-experiments`）

后端是 validated fixed/cached QMC order 14，**不是** exact oracle。Gas **不是** VAEAC。

## 复现入口（论文）

```text
python scripts/calibrate_fast_backend.py --config configs/paper/oracle_calibration.yaml
python scripts/run_synthetic_main.py --config configs/paper/synthetic_main_validated_fixed_order14_j2_200_h.yaml --resume
python scripts/run_nonlinearity.py --config configs/paper/nonlinearity_order14_cloud.yaml --resume
python scripts/run_reuse.py --config configs/paper/reuse_order14_t50_cloud.yaml --resume
python scripts/run_gas_semisynthetic.py --config configs/paper/gas_semisynthetic_empirical_h.yaml --resume
python scripts/run_gas_natural.py --config configs/paper/gas_natural_empirical_h.yaml --resume
```

默认 `pytest` 只跑快速测试，不触发 G-ladder / rejection / adaptive order 16。

## 目录

| 路径 | 用途 |
| --- | --- |
| `src/rr_gid_cn/` | 算法实现 |
| `configs/paper/` | 论文实验配置 |
| `scripts/` | runner、云端分片、审计、出图 |
| `paper/` | 实验定稿 TeX / 图 |
| `paper_tables/` | Table 1a/1b（200-rep E1 / 20-rep E5 R1） |
| `paper_results/` | 审计通过的 jsonl、审计报告、图（不含 QMC spill） |
| `scripts/diagnostics_legacy/` | 已冻结的 G0–G4 / Phase 0–3 脚本 |
| `results/diagnostics_legacy/` | 已冻结的 G0–G3 证书 |

## 权威文档

- `核心文档/RR_GID_CN.pdf`（10 页定稿：原稿理论 + 实验落地）
- `核心文档/RR_GID_CN_draft_20260820.pdf`（8 页计划稿备份）
- `核心文档/RR_GID_实验重规划与代码改造执行指南_20260902.md`
- `核心文档/RR_GID_论文实验完整总结_20260913.pdf`
