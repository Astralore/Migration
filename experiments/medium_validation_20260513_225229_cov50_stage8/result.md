# 中等规模 cov50 数据验证报告

**生成时间**：2026-05-14 00:00:18  
**输出目录**：`experiments\medium_validation_20260513_225229_cov50_stage8`  
**数据文件**：`data\processed\taxi_cleaned_active100_min100_eps2h_cov50.csv`  
**数据规模**：Top-12 active taxis；train=37832 rows/9 taxis；test=11548 rows/3 taxis  
**切分方式**：balanced_by_nearest_server_exposure；test row ratio=0.234；test risk ratio=0.134  
**GAT-MARL epochs**：Proactive=8，Reactive=8

## 训练段

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 1132 | 5695 | 292 | 5.99 | 20817.02 |
| Nearest | 1625 | 5312 | 288 | 0.05 | 36116.34 |
| DQN | 3970 | 6546 | 328 | 0.58 | 24911.76 |
| GAT-MARL | 0 | 23349 | 209 | 0.39 | 19824.72 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 1157 | 11414 | 4.37 | 11465.63 |
| Nearest | 1349 | 5525 | 0.05 | 46143.46 |
| DQN | 4042 | 6415 | 0.37 | 22484.22 |
| GAT-MARL | 13432 | 8778 | 0.43 | 42507.20 |


## 推理段

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 426 | 958 | 164 | 5.81 | 20025.46 |
| Nearest | 783 | 864 | 124 | 0.05 | 71407.05 |
| DQN | 423 | 2776 | 121 | 0.45 | 20923.12 |
| GAT-MARL | 0 | 3533 | 92 | 0.42 | 19494.57 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 463 | 1008 | 4.31 | 27386.07 |
| Nearest | 513 | 956 | 0.05 | 30664.00 |
| DQN | 1507 | 1476 | 0.31 | 27332.75 |
| GAT-MARL | 2899 | 1439 | 0.43 | 40541.78 |


## 本轮验证关注点

- 验证脚本显式使用 cov50 覆盖过滤数据，不再读取旧 cleaned CSV。
- train/test 按最近服务器距离风险暴露平衡切分，减少 proactive opportunity 偏斜。
- Avg Total Cost 使用 `total_cost_ms_sum / decision_count`，包含 SLA / future / tearing。

