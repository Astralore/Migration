# 中等规模 cov50 数据验证报告

**生成时间**：2026-05-19 15:39:15  
**输出目录**：`experiments\medium_validation_20260519_1423_cov50_cfscore_tuned`  
**数据文件**：`data\processed\taxi_cleaned_active100_min100_eps2h_cov50.csv`  
**数据规模**：Top-12 active taxis；train=37832 rows/9 taxis；test=11548 rows/3 taxis  
**切分方式**：balanced_by_nearest_server_exposure；test row ratio=0.234；test risk ratio=0.134  
**GAT-MARL epochs**：Proactive=8，Reactive=8

## 训练段

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 1135 | 5648 | 315 | 5.98 | 21078.98 |
| Nearest | 1625 | 5312 | 288 | 0.05 | 36116.34 |
| DQN | 4945 | 6019 | 279 | 0.68 | 26099.20 |
| GAT-MARL | 532 | 10483 | 309 | 0.81 | 19464.94 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 1190 | 9398 | 4.34 | 14002.20 |
| Nearest | 1349 | 5525 | 0.05 | 46143.46 |
| DQN | 4237 | 6003 | 0.40 | 26353.37 |
| GAT-MARL | 401 | 7718 | 0.79 | 19748.64 |


## 推理段

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 445 | 959 | 153 | 5.83 | 21113.69 |
| Nearest | 783 | 864 | 124 | 0.05 | 71407.05 |
| DQN | 458 | 2582 | 106 | 0.49 | 22310.71 |
| GAT-MARL | 231 | 912 | 171 | 0.84 | 17423.24 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 459 | 1016 | 4.28 | 26309.68 |
| Nearest | 513 | 956 | 0.05 | 30664.00 |
| DQN | 325 | 1472 | 0.31 | 21394.21 |
| GAT-MARL | 166 | 3517 | 0.75 | 18911.02 |


## 本轮验证关注点

- 验证脚本显式使用 cov50 覆盖过滤数据，不再读取旧 cleaned CSV。
- train/test 按最近服务器距离风险暴露平衡切分，减少 proactive opportunity 偏斜。
- Avg Total Cost 使用 `total_cost_ms_sum / decision_count`，包含 SLA / future / tearing.

