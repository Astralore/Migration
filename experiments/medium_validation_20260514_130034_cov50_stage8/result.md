# 中等规模 cov50 数据验证报告

**生成时间**：2026-05-14 14:04:19  
**输出目录**：`experiments\medium_validation_20260514_130034_cov50_stage8`  
**数据文件**：`data\processed\taxi_cleaned_active100_min100_eps2h_cov50.csv`  
**数据规模**：Top-12 active taxis；train=37832 rows/9 taxis；test=11548 rows/3 taxis  
**切分方式**：balanced_by_nearest_server_exposure；test row ratio=0.234；test risk ratio=0.134  
**GAT-MARL epochs**：Proactive=8，Reactive=8

## 训练段

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 1135 | 5648 | 315 | 6.32 | 21078.98 |
| Nearest | 1625 | 5312 | 288 | 0.05 | 36116.34 |
| DQN | 4163 | 6207 | 286 | 0.69 | 24651.11 |
| GAT-MARL | 7383 | 11828 | 289 | 0.40 | 25385.22 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 1123 | 6389 | 4.46 | 20354.80 |
| Nearest | 1349 | 5525 | 0.05 | 46143.46 |
| DQN | 4017 | 6481 | 0.37 | 23118.95 |
| GAT-MARL | 22351 | 14942 | 0.38 | 40070.84 |


## 推理段

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 447 | 977 | 156 | 5.71 | 20824.06 |
| Nearest | 783 | 864 | 124 | 0.05 | 71407.05 |
| DQN | 389 | 6665 | 91 | 0.44 | 15316.30 |
| GAT-MARL | 302 | 1722 | 114 | 0.42 | 31023.68 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 480 | 1002 | 4.19 | 28329.03 |
| Nearest | 513 | 956 | 0.05 | 30664.00 |
| DQN | 611 | 1648 | 0.29 | 22284.63 |
| GAT-MARL | 2934 | 1511 | 0.42 | 37311.23 |


## 本轮验证关注点

- 验证脚本显式使用 cov50 覆盖过滤数据，不再读取旧 cleaned CSV。
- train/test 按最近服务器距离风险暴露平衡切分，减少 proactive opportunity 偏斜。
- Avg Total Cost 使用 `total_cost_ms_sum / decision_count`，包含 SLA / future / tearing.

