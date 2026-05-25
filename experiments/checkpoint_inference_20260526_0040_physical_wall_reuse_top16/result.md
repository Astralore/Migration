# Checkpoint 推理对比实验

**生成时间**：2026-05-26 00:42:55  
**输出目录**：`experiments\checkpoint_inference_20260526_0040_physical_wall_reuse_top16`  
**复用 checkpoint**：`experiments\medium_validation_20260525_2241_physical_wall_v1\checkpoints`  
**数据文件**：`data\processed\taxi_cleaned_active100_min100_eps2h_cov50.csv`  
**数据规模**：Top-16 active taxis；source train=37832 rows/9 taxis；inference=21602 rows/7 taxis  
**推理数据口径**：排除上一轮训练 taxi，使用上一轮 test taxi + Top-N 中新增 taxi。  

## 推理结果

| Algorithm | Migrations | SLA Risk Count | Severe SLA Violations | Avg SLA Excess (km) | P95 SLA Excess (km) | Avg SLA Penalty (ms) | Proactive Decisions | Avg Decision Time (ms) | Avg Access Latency (ms) | Avg Total System Cost (ms) |
|-----------|------------|----------------|-----------------------|---------------------|---------------------|----------------------|---------------------|------------------------|-------------------------|----------------------------|
| SA | 74 | 7542 | 2757 | 2.36 | 11.85 | 74896.40 | 214 | 6.82 | 2.11 | 75066.93 |
| Nearest | 1797 | 2413 | 1223 | 1.50 | 8.37 | 134304.94 | 235 | 0.05 | 2.13 | 246764.88 |
| DQN | 3576 | 4953 | 1310 | 1.77 | 8.37 | 100323.76 | 257 | 0.56 | 2.11 | 112895.67 |
| GAT-MARL | 225 | 6356 | 1819 | 1.87 | 11.06 | 72449.56 | 277 | 0.93 | 2.10 | 74873.44 |

| Algorithm | Migrations | SLA Risk Count | Severe SLA Violations | Avg SLA Excess (km) | P95 SLA Excess (km) | Avg SLA Penalty (ms) | Avg Decision Time (ms) | Avg Access Latency (ms) | Avg Total System Cost (ms) |
|-----------|------------|----------------|-----------------------|---------------------|---------------------|----------------------|------------------------|-------------------------|----------------------------|
| SA | 70 | 9331 | 3308 | 2.49 | 12.49 | 63815.63 | 5.03 | 2.10 | 63934.27 |
| Nearest | 1562 | 2570 | 1223 | 1.50 | 8.37 | 138381.12 | 0.06 | 2.13 | 300741.10 |
| DQN | 2516 | 4519 | 2035 | 2.06 | 10.12 | 133420.46 | 0.36 | 2.12 | 158870.11 |
| GAT-MARL | 258 | 8166 | 1296 | 1.60 | 8.37 | 59268.51 | 0.57 | 2.09 | 61131.43 |


## 成本分解均值

| Algorithm | Proactive SLA Penalty (ms) | Proactive Migration (ms) | Reactive SLA Penalty (ms) | Reactive Migration (ms) |
|-----------|----------------------------|--------------------------|---------------------------|-------------------------|
| SA | 74896.40 | 106.75 | 63815.63 | 80.76 |
| Nearest | 134304.94 | 112457.75 | 138381.12 | 162357.84 |
| DQN | 100323.76 | 12465.15 | 133420.46 | 25309.07 |
| GAT-MARL | 72449.56 | 2281.77 | 59268.51 | 1769.62 |


## 说明

- 本实验不重新训练 DQN/GAT-MARL，只加载上一轮 `physical_wall_v1` 保存的 checkpoint。
- SA、Nearest、DQN、GAT-MARL 使用完全相同的 inference dataframe、server dataframe 和 predictor。
- predictor 只用上一轮训练 taxi 拟合，避免用推理 taxi 轨迹泄漏未来信息。
