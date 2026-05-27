# 中等规模 cov50 数据验证报告

**生成时间**：2026-05-26 14:09:28  
**输出目录**：`experiments\medium_validation_20260526_phaseB_reward_align_v1`  
**数据文件**：`data\processed\taxi_cleaned_active100_min100_eps2h_cov50.csv`  
**数据规模**：Top-12 active taxis；train=37832 rows/9 taxis；test=11548 rows/3 taxis  
**切分方式**：balanced_by_nearest_server_exposure；test row ratio=0.234；test risk ratio=0.134  
**GAT-MARL epochs**：Proactive=2，Reactive=2

## 训练段

| Algorithm | Migrations | SLA Risk Count | Severe SLA Violations | Avg SLA Excess (km) | P95 SLA Excess (km) | Avg SLA Penalty (ms) | Proactive Decisions | Avg Decision Time (ms) | Avg Access Latency (ms) | Avg Total System Cost (ms) |
|-----------|------------|----------------|-----------------------|---------------------|---------------------|----------------------|---------------------|------------------------|-------------------------|----------------------------|
| SA | 156 | 22608 | 7281 | 2.11 | 12.96 | 24107.36 | 244 | 6.60 | 2.08 | 24180.95 |
| Nearest | 2434 | 5312 | 4986 | 1.22 | 11.51 | 50311.08 | 288 | 0.05 | 2.11 | 124857.06 |
| DQN | 3591 | 6566 | 5212 | 1.37 | 11.79 | 53604.96 | 284 | 0.78 | 2.11 | 82489.15 |
| GAT-MARL | 198 | 20610 | 6822 | 1.98 | 11.57 | 22131.42 | 250 | 0.80 | 2.08 | 22441.48 |

| Algorithm | Migrations | SLA Risk Count | Severe SLA Violations | Avg SLA Excess (km) | P95 SLA Excess (km) | Avg SLA Penalty (ms) | Avg Decision Time (ms) | Avg Access Latency (ms) | Avg Total System Cost (ms) |
|-----------|------------|----------------|-----------------------|---------------------|---------------------|----------------------|------------------------|-------------------------|----------------------------|
| SA | 9 | 24923 | 10078 | 3.70 | 18.62 | 46400.66 | 5.05 | 2.10 | 46515.21 |
| Nearest | 1934 | 5525 | 4987 | 1.22 | 11.51 | 50994.03 | 0.05 | 2.11 | 146543.64 |
| DQN | 4581 | 10087 | 5557 | 1.46 | 11.80 | 44773.93 | 0.47 | 2.11 | 64944.30 |
| GAT-MARL | 239 | 19635 | 5215 | 1.43 | 11.56 | 20270.66 | 0.67 | 2.08 | 20862.03 |


## 推理段

| Algorithm | Migrations | SLA Risk Count | Severe SLA Violations | Avg SLA Excess (km) | P95 SLA Excess (km) | Avg SLA Penalty (ms) | Proactive Decisions | Avg Decision Time (ms) | Avg Access Latency (ms) | Avg Total System Cost (ms) |
|-----------|------------|----------------|-----------------------|---------------------|---------------------|----------------------|---------------------|------------------------|-------------------------|----------------------------|
| SA | 78 | 5984 | 961 | 1.18 | 6.72 | 14553.22 | 144 | 53.37 | 2.08 | 14839.34 |
| Nearest | 1078 | 864 | 443 | 0.50 | 4.91 | 32956.28 | 124 | 0.07 | 2.09 | 213085.19 |
| DQN | 1027 | 3197 | 997 | 0.98 | 7.35 | 32192.73 | 159 | 0.65 | 2.09 | 59347.54 |
| GAT-MARL | 147 | 5691 | 943 | 1.31 | 8.37 | 14708.04 | 124 | 1.17 | 2.08 | 15177.41 |

| Algorithm | Migrations | SLA Risk Count | Severe SLA Violations | Avg SLA Excess (km) | P95 SLA Excess (km) | Avg SLA Penalty (ms) | Avg Decision Time (ms) | Avg Access Latency (ms) | Avg Total System Cost (ms) |
|-----------|------------|----------------|-----------------------|---------------------|---------------------|----------------------|------------------------|-------------------------|----------------------------|
| SA | 93 | 5103 | 1143 | 1.09 | 9.27 | 17490.92 | 40.79 | 2.08 | 17812.12 |
| Nearest | 988 | 956 | 443 | 0.50 | 4.91 | 34059.42 | 0.07 | 2.09 | 326779.85 |
| DQN | 1219 | 2430 | 462 | 0.54 | 4.91 | 16489.54 | 0.50 | 2.08 | 24724.20 |
| GAT-MARL | 127 | 3056 | 609 | 0.65 | 5.37 | 20767.47 | 0.97 | 2.08 | 22248.60 |


## 成本分解堆叠图

![Inference Cost Decomposition](cost_decomposition_inference.png)

## 推理阶段成本分解均值

| Algorithm | Proactive SLA Penalty (ms) | Proactive Migration (ms) | Reactive SLA Penalty (ms) | Reactive Migration (ms) |
|-----------|----------------------------|--------------------------|---------------------------|-------------------------|
| SA | 14553.22 | 163.91 | 17490.92 | 259.00 |
| Nearest | 32956.28 | 180126.79 | 34059.42 | 292718.34 |
| DQN | 32192.73 | 27032.67 | 16489.54 | 8174.13 |
| GAT-MARL | 14708.04 | 420.64 | 20767.47 | 1301.52 |


## 推理阶段迁移效率指标

> **Cost/Node** = `total_migration_cost / total_migrations`（每次迁移节点的平均线性物理时延）  
> **Cost/Mig Decision** = `total_migration_cost / migration_decision_count`（仅 `migration_cost > 0` 的决策）  
> **Migration Share** = `total_migration_cost / total_cost_ms_sum`（迁移在总系统成本中的占比）

| Algorithm | Pro: Cost/Node (ms) | Pro: Cost/Mig Decision (ms) | Pro: Migration Share | Rea: Cost/Node (ms) | Rea: Cost/Mig Decision (ms) | Rea: Migration Share |
|-----------|---------------------|-----------------------------|----------------------|---------------------|-----------------------------|----------------------|
| SA | 11961.42 | 23324.77 | 1.10% | 10145.72 | 34946.39 | 1.45% |
| Nearest | 165088.38 | 983233.53 | 84.53% | 283237.58 | 1841044.30 | 89.58% |
| DQN | 43826.09 | 95561.35 | 45.55% | 16294.61 | 26697.76 | 33.06% |
| GAT-MARL | 11672.09 | 16988.09 | 2.77% | 17790.86 | 21936.30 | 5.85% |


## 本轮验证关注点

- 验证脚本显式使用 cov50 覆盖过滤数据，不再读取旧 cleaned CSV。
- train/test 按最近服务器距离风险暴露平衡切分，减少 proactive opportunity 偏斜。
- Avg Decision Time 是算法计算耗时；Avg Access Latency 是真实接入延迟；Avg Total System Cost 使用 `total_cost_ms_sum / decision_count`，包含 SLA penalty 和迁移等系统代价。
- 迁移对比优先看「迁移效率指标」表（按节点 / 按有迁解决策 / 占比），而非 `total_migration_cost / decision_count`（会被大量无迁解决策稀释）。
- SLA Risk Count 是 max-entry 风险次数；Severe SLA Violations 和 SLA Excess 用于区分轻微风险与严重服务质量退化。

