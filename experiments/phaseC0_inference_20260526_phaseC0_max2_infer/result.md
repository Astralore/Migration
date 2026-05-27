# Phase C0 推理摸底：放宽每步迁移上限（不重训）

**生成时间**：2026-05-26 11:35:19  
**输出目录**：`experiments\phaseC0_inference_20260526_phaseC0_max2_infer`  
**Checkpoint**：`experiments\medium_validation_20260526_phaseB_reward_align_v1`（阶段 B，未重训）  
**Guard**：Proactive MAX=2，Reactive MAX=2，Proactive budget=6000 ms  
**对比基线**：`experiments\medium_validation_20260526_phaseB_reward_align_v1/results.json` 中阶段 B 推理（MAX=1，budget=3000）

## 推理 Proactive（GAT-MARL）

| Run | Migrations | Mig Decisions | SLA Risk | Severe | P95 (km) | Avg SLA Penalty (ms) | Avg Total Cost (ms) | Pro Decisions | Cost/Node | Cost/Mig Decision | Migration Share |
|-----|------------|---------------|----------|--------|----------|----------------------|---------------------|---------------|-----------|-------------------|-----------------|
| Phase B (MAX=1) | 152 | 152 | 4518 | 857 | 7.86 | 21875.21 | 22146.99 | 116 | 3090.94 | 3090.94 | 1.03% |
| Phase C0 (MAX=2) | 80 | 69 | 5996 | 928 | 7.87 | 16669.66 | 17174.09 | 147 | 19095.59 | 22139.81 | 2.24% |


## 推理 Reactive（GAT-MARL）


| Run | Migrations | Mig Decisions | SLA Risk | Severe | P95 (km) | Avg SLA Penalty (ms) | Avg Total Cost (ms) | Cost/Node | Cost/Mig Decision | Migration Share |
|-----|------------|---------------|----------|--------|----------|----------------------|---------------------|-----------|-------------------|-----------------|
| Phase B (MAX=1) | 122 | 122 | 3360 | 603 | 5.43 | 21022.56 | 22385.45 | 16941.08 | 16941.08 | 5.28% |
| Phase C0 (MAX=2) | 348 | 289 | 4805 | 523 | 4.91 | 18346.15 | 19561.80 | 6612.00 | 7961.85 | 5.84% |



## 迁移效率汇总（GAT-MARL）

| Run | Pro: Cost/Node | Pro: Cost/Mig Dec | Pro: Share | Rea: Cost/Node | Rea: Cost/Mig Dec | Rea: Share |
|-----|----------------|-------------------|------------|----------------|-------------------|------------|
| Phase B | 3090.94 | 3090.94 | 1.03% | 16941.08 | 16941.08 | 5.28% |
| Phase C0 | 19095.59 | 22139.81 | 2.24% | 6612.00 | 7961.85 | 5.84% |


## 说明

- C0 仅改 inference guard；策略权重仍为阶段 B checkpoint。
- **Mig Decisions** = `migration_decision_count`（本步 `migration_cost > 0` 的次数）。
- 若 C0 在 SLA/Severe 上明显优于 B 且迁移可控，再进入阶段 C 重训（train/infer 一致）。
