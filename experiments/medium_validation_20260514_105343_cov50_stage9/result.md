# 中等规模 cov50 数据验证报告

**生成时间**：2026-05-14 12:16:36  
**输出目录**：`experiments\medium_validation_20260514_105343_cov50_stage9`  
**数据文件**：`data\processed\taxi_cleaned_active100_min100_eps2h_cov50.csv`  
**数据规模**：Top-12 active taxis；train=37832 rows/9 taxis；test=11548 rows/3 taxis  
**切分方式**：balanced_by_nearest_server_exposure；test row ratio=0.234；test risk ratio=0.134  
**GAT-MARL epochs**：Proactive=8，Reactive=8

## 训练段

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 1132 | 5695 | 292 | 6.43 | 20817.02 |
| Nearest | 1625 | 5312 | 288 | 0.06 | 36116.34 |
| DQN | 4186 | 5842 | 276 | 0.72 | 23581.54 |
| GAT-MARL | 507 | 8913 | 250 | 0.53 | 20713.67 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 1163 | 5790 | 4.93 | 23006.52 |
| Nearest | 1349 | 5525 | 0.06 | 46143.46 |
| DQN | 3775 | 6535 | 0.46 | 22991.28 |
| GAT-MARL | 9753 | 17817 | 0.46 | 23603.44 |


## 推理段

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 444 | 975 | 169 | 6.25 | 20032.92 |
| Nearest | 783 | 864 | 124 | 0.06 | 71407.05 |
| DQN | 584 | 4878 | 85 | 0.51 | 16319.35 |
| GAT-MARL | 344 | 868 | 126 | 0.55 | 26851.44 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 506 | 1013 | 4.59 | 29355.47 |
| Nearest | 513 | 956 | 0.06 | 30664.00 |
| DQN | 349 | 1173 | 0.34 | 24124.06 |
| GAT-MARL | 1100 | 4318 | 0.52 | 24780.92 |


## 本轮验证关注点

- 验证脚本显式使用 cov50 覆盖过滤数据，不再读取旧 cleaned CSV。
- train/test 按最近服务器距离风险暴露平衡切分，减少 proactive opportunity 偏斜。
- Avg Total Cost 使用 `total_cost_ms_sum / decision_count`，包含 SLA / future / tearing。

