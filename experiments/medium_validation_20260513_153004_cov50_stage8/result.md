# 中等规模 cov50 数据验证报告

**生成时间**：2026-05-13 17:38:53  
**输出目录**：`experiments\medium_validation_20260513_153004_cov50_stage8`  
**数据文件**：`data\processed\taxi_cleaned_active100_min100_eps2h_cov50.csv`  
**数据规模**：Top-12 active taxis；train=37832 rows/9 taxis；test=11548 rows/3 taxis  
**切分方式**：balanced_by_nearest_server_exposure；test row ratio=0.234；test risk ratio=0.134  
**SAC epochs**：Proactive=6，Reactive=6

## 训练段

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 823 | 5540 | 306 | 5.92 | 21538.82 |
| DQN | 4770 | 5472 | 279 | 0.73 | 23515.36 |
| Hybrid SAC | 2626 | 21572 | 219 | 1.49 | 21042.73 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 1037 | 5553 | 4.37 | 24319.25 |
| DQN | 4308 | 6811 | 0.46 | 21519.71 |
| Hybrid SAC | 1361 | 5524 | 1.37 | 25182.98 |


## 推理段

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 368 | 967 | 169 | 5.81 | 25313.58 |
| DQN | 1152 | 857 | 126 | 0.51 | 25845.09 |
| Hybrid SAC | 720 | 3533 | 85 | 1.32 | 21219.97 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 499 | 975 | 4.28 | 34294.96 |
| DQN | 699 | 957 | 0.19 | 36164.05 |
| Hybrid SAC | 647 | 957 | 1.44 | 37393.40 |


## 本轮验证关注点

- 验证脚本显式使用 cov50 覆盖过滤数据，不再读取旧 cleaned CSV。
- train/test 按最近服务器距离风险暴露平衡切分，减少 proactive opportunity 偏斜。
- Avg Total Cost 使用 `total_cost_ms_sum / decision_count`，包含 SLA / future / tearing。

