# 中等规模 cov50 数据验证报告

**生成时间**：2026-05-11 16:38:33  
**输出目录**：`experiments\medium_validation_20260511_1610_cov50_predictor_v2`  
**数据文件**：`data\processed\taxi_cleaned_active100_min100_eps2h_cov50.csv`  
**数据规模**：Top-12 active taxis；train=37832 rows/9 taxis；test=11548 rows/3 taxis  
**切分方式**：balanced_by_nearest_server_exposure；test row ratio=0.234；test risk ratio=0.134  
**SAC epochs**：Proactive=2，Reactive=2

## 训练段

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 878 | 5536 | 341 | 5.80 | 21977.07 |
| DQN | 4699 | 5515 | 283 | 0.70 | 23480.16 |
| Hybrid SAC | 846 | 5545 | 282 | 0.99 | 21828.18 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 1050 | 5550 | 4.47 | 24446.29 |
| DQN | 5500 | 7426 | 0.47 | 21657.34 |
| Hybrid SAC | 900 | 5536 | 0.95 | 22617.30 |


## 推理段

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 370 | 976 | 172 | 6.11 | 25054.27 |
| DQN | 510 | 1026 | 132 | 0.47 | 21600.82 |
| Hybrid SAC | 471 | 964 | 138 | 0.92 | 25819.99 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 509 | 985 | 4.42 | 35272.66 |
| DQN | 526 | 972 | 0.21 | 22005.95 |
| Hybrid SAC | 358 | 985 | 0.96 | 25072.01 |


## 本轮验证关注点

- 验证脚本显式使用 cov50 覆盖过滤数据，不再读取旧 cleaned CSV。
- train/test 按最近服务器距离风险暴露平衡切分，减少 proactive opportunity 偏斜。
- Avg Total Cost 使用 `total_cost_ms_sum / decision_count`，包含 SLA / future / tearing。

