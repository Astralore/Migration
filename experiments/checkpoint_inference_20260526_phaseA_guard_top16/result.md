# Checkpoint 推理对比实验

**生成时间**：2026-05-26 01:05:29  
**输出目录**：`experiments\checkpoint_inference_20260526_phaseA_guard_top16`  
**复用 checkpoint**：`experiments\medium_validation_20260525_2241_physical_wall_v1\checkpoints`  
**数据文件**：`data\processed\taxi_cleaned_active100_min100_eps2h_cov50.csv`  
**数据规模**：Top-16 active taxis；source train=37832 rows/9 taxis；inference=21602 rows/7 taxis  
**推理数据口径**：排除上一轮训练 taxi，使用上一轮 test taxi + Top-N 中新增 taxi。  

## 推理结果

| Algorithm | Migrations | SLA Risk Count | Severe SLA Violations | Avg SLA Excess (km) | P95 SLA Excess (km) | Avg SLA Penalty (ms) | Proactive Decisions | Avg Decision Time (ms) | Avg Access Latency (ms) | Avg Total System Cost (ms) |
|-----------|------------|----------------|-----------------------|---------------------|---------------------|----------------------|---------------------|------------------------|-------------------------|----------------------------|
| SA | 107 | 7681 | 2783 | 2.24 | 11.85 | 64242.44 | 218 | 6.76 | 2.10 | 64404.08 |
| Nearest | 1797 | 2413 | 1223 | 1.50 | 8.37 | 134304.94 | 235 | 0.05 | 2.13 | 246764.88 |
| DQN | 2019 | 6139 | 1898 | 2.10 | 11.36 | 115518.38 | 235 | 0.56 | 2.12 | 123289.35 |
| GAT-MARL | 237 | 8685 | 1953 | 1.96 | 9.19 | 75697.43 | 275 | 0.91 | 2.10 | 76627.15 |

| Algorithm | Migrations | SLA Risk Count | Severe SLA Violations | Avg SLA Excess (km) | P95 SLA Excess (km) | Avg SLA Penalty (ms) | Avg Decision Time (ms) | Avg Access Latency (ms) | Avg Total System Cost (ms) |
|-----------|------------|----------------|-----------------------|---------------------|---------------------|----------------------|------------------------|-------------------------|----------------------------|
| SA | 59 | 9370 | 3879 | 2.70 | 14.85 | 65724.70 | 5.17 | 2.11 | 65858.72 |
| Nearest | 1562 | 2570 | 1223 | 1.50 | 8.37 | 138381.12 | 0.06 | 2.13 | 300741.10 |
| DQN | 2516 | 4519 | 2035 | 2.06 | 10.12 | 133420.46 | 0.38 | 2.12 | 158870.11 |
| GAT-MARL | 284 | 8635 | 1524 | 1.70 | 8.37 | 72107.87 | 0.60 | 2.10 | 73052.07 |


## 成本分解均值

| Algorithm | Proactive SLA Penalty (ms) | Proactive Migration (ms) | Reactive SLA Penalty (ms) | Reactive Migration (ms) |
|-----------|----------------------------|--------------------------|---------------------------|-------------------------|
| SA | 64242.44 | 113.36 | 65724.70 | 97.96 |
| Nearest | 134304.94 | 112457.75 | 138381.12 | 162357.84 |
| DQN | 115518.38 | 7671.58 | 133420.46 | 25309.07 |
| GAT-MARL | 75697.43 | 783.40 | 72107.87 | 856.73 |


## 说明

- 本实验不重新训练 DQN/GAT-MARL，只加载上一轮 `physical_wall_v1` 保存的 checkpoint。
- SA、Nearest、DQN、GAT-MARL 使用完全相同的 inference dataframe、server dataframe 和 predictor。
- predictor 只用上一轮训练 taxi 拟合，避免用推理 taxi 轨迹泄漏未来信息。
