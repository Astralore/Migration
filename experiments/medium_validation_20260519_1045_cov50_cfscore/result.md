# 中等规模 cov50 数据验证报告

**生成时间**：2026-05-19 11:57:40  
**输出目录**：`experiments\medium_validation_20260519_1045_cov50_cfscore`  
**数据文件**：`data\processed\taxi_cleaned_active100_min100_eps2h_cov50.csv`  
**数据规模**：Top-12 active taxis；train=37832 rows/9 taxis；test=11548 rows/3 taxis  
**切分方式**：balanced_by_nearest_server_exposure；test row ratio=0.234；test risk ratio=0.134  
**GAT-MARL epochs**：Proactive=8，Reactive=8

## 训练段

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 1115 | 5600 | 293 | 6.02 | 21237.87 |
| Nearest | 1625 | 5312 | 288 | 0.05 | 36116.34 |
| DQN | 4318 | 5981 | 279 | 0.72 | 26039.27 |
| GAT-MARL | 601 | 5583 | 281 | 0.88 | 19711.89 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 1199 | 6081 | 4.52 | 21911.16 |
| Nearest | 1349 | 5525 | 0.05 | 46143.46 |
| DQN | 3446 | 6214 | 0.43 | 23828.91 |
| GAT-MARL | 396 | 5593 | 0.63 | 19738.37 |


## 推理段

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 473 | 973 | 169 | 5.60 | 21365.94 |
| Nearest | 783 | 864 | 124 | 0.05 | 71407.05 |
| DQN | 442 | 1113 | 119 | 0.43 | 24316.89 |
| GAT-MARL | 245 | 938 | 164 | 0.80 | 19364.49 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 464 | 984 | 4.16 | 27205.10 |
| Nearest | 513 | 956 | 0.06 | 30664.00 |
| DQN | 329 | 959 | 0.31 | 23276.54 |
| GAT-MARL | 199 | 995 | 0.65 | 18897.86 |


## 本轮验证关注点

- 验证脚本显式使用 cov50 覆盖过滤数据，不再读取旧 cleaned CSV。
- train/test 按最近服务器距离风险暴露平衡切分，减少 proactive opportunity 偏斜。
- Avg Total Cost 使用 `total_cost_ms_sum / decision_count`，包含 SLA / future / tearing.

