# 中等规模 cov50 数据验证报告

**生成时间**：2026-05-14 15:46:16  
**输出目录**：`experiments\medium_validation_20260514_143332_cov50_stage8`  
**数据文件**：`data\processed\taxi_cleaned_active100_min100_eps2h_cov50.csv`  
**数据规模**：Top-12 active taxis；train=37832 rows/9 taxis；test=11548 rows/3 taxis  
**切分方式**：balanced_by_nearest_server_exposure；test row ratio=0.234；test risk ratio=0.134  
**GAT-MARL epochs**：Proactive=8，Reactive=8

## 训练段

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 1132 | 5695 | 292 | 6.31 | 20817.02 |
| Nearest | 1625 | 5312 | 288 | 0.06 | 36116.34 |
| DQN | 4286 | 6019 | 287 | 0.73 | 26639.55 |
| GAT-MARL | 77 | 22127 | 200 | 0.43 | 19868.97 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 1137 | 6268 | 4.56 | 21059.19 |
| Nearest | 1349 | 5525 | 0.05 | 46143.46 |
| DQN | 4156 | 6074 | 0.38 | 23523.75 |
| GAT-MARL | 22127 | 5527 | 0.47 | 132853.37 |


## 推理段

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 458 | 974 | 182 | 5.91 | 20258.70 |
| Nearest | 783 | 864 | 124 | 0.05 | 71407.05 |
| DQN | 747 | 978 | 114 | 0.47 | 50111.36 |
| GAT-MARL | 46 | 3309 | 94 | 0.43 | 19643.48 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 464 | 987 | 4.28 | 28580.23 |
| Nearest | 513 | 956 | 0.06 | 30664.00 |
| DQN | 557 | 1144 | 0.31 | 26659.79 |
| GAT-MARL | 3824 | 956 | 0.48 | 125872.76 |


## 本轮验证关注点

- 验证脚本显式使用 cov50 覆盖过滤数据，不再读取旧 cleaned CSV。
- train/test 按最近服务器距离风险暴露平衡切分，减少 proactive opportunity 偏斜。
- Avg Total Cost 使用 `total_cost_ms_sum / decision_count`，包含 SLA / future / tearing.

