角色设定：
你是一位资深的边缘计算与时空数据预测专家。基于最新的诊断报告 `test.md`，我们发现强化学习系统前瞻机制（Proactive）表现不佳的核心原因是 `prediction/simple_predictor.py` 中的预测逻辑存在严重的 Train/Test 分歧和短期预测迟钝问题。现进入 **【第六阶段：轨迹预测引擎升级】**。

**当前任务修改范围（绝对红线）**：
你 **只能** 修改 `prediction/simple_predictor.py`。严禁修改强化学习底层网络、Reward 或环境主干代码！

**实现边界补充（必须遵守）**：
- 当前主流程只接入 `prediction/simple_predictor.py`，不要在本阶段接入 `prediction/spatioformer.py`。SpatioFormer 涉及模型训练、归一化、checkpoint、推理耗时和输入窗口改造，超出本阶段目标。
- 不修改 SA / DQN / Hybrid SAC 的网络、reward、trigger 主逻辑和环境循环。它们已经通过统一接口调用 `predict_future(...)`。
- 保持 `SimpleTrajectoryPredictor` 对外兼容：现有 `SimpleTrajectoryPredictor(forecast_horizon=...)`、`.fit(df)`、`.predict_future(...)` 调用不能失效。

**核心代码重构任务（对标 test.md 改进建议）：**

**1. 废除 Train/Test 双轨制，统一“局部速度优先”原则**
- **现状**：已见车辆使用长期平均位移，未见车辆使用局部速度，导致训练段无法敏锐捕捉突发越界风险。
- **要求**：在 `predict_future` 函数中，无论 `taxi_id` 是否在训练集中见过，**优先使用最近两个 GPS 点计算出的真实局部速度（Local Velocity）** `v_lon` 和 `v_lat` 作为主要预测依据。
- **实现要求**：局部速度有效条件为：存在 `prev_lon / prev_lat`，且 `delta_t_prev_sec` 或由 `prev_time/current_time` 计算出的 `dt` 满足 `0 < dt <= max_local_dt_sec`。建议 `max_local_dt_sec = 300.0`。

**2. 引入时间间隔截断防漂移（防速度失真）**
- **要求**：在计算局部速度时，检查 `delta_t_prev_sec`。如果该间隔异常巨大（例如超过 300 秒 / 5 分钟），说明中间有严重的信号丢失。此时算出的局部速度极不可靠，必须**强制截断该车的局部速度估算**，回退到历史平均位移，或直接预测其原地不动。
- **fallback 顺序必须明确**：
  1. 局部速度有效 + 有历史平均：使用 EMA Fusion。
  2. 局部速度有效 + 无历史平均：使用纯局部速度预测。
  3. 局部速度无效 + 有历史平均：回退历史平均位移预测。
  4. 局部速度无效 + 无历史平均：原地预测。
- 不建议在已见车辆缺少局部速度时直接原地预测，否则会破坏已有历史信息。

**3. 实现“局部速度 + 历史平均”的指数加权融合 (EMA Fusion)**
- **要求**：针对已见车辆（有 `velocity_factors` 记录），不能完全抛弃历史规律。请实现融合公式：
  `predicted_step_lon = alpha * (local_v_lon * dt) + (1 - alpha) * historical_mean_dx`
  `predicted_step_lat = alpha * (local_v_lat * dt) + (1 - alpha) * historical_mean_dy`
  建议将 `alpha` 设置为 `0.8` 或 `0.9`（极度信任局部近期速度，用历史平均做微小的平滑阻尼）。如果车辆无历史数据或无法算局部速度，则退化为单支路预测。
- **实现细节**：融合公式里的 `dt` 应使用每一个未来预测步的 `dtf`，因为当前 `delta_t_future_sec` 支持标量，也支持序列。不要只使用一个固定 dt。
- 建议在 `SimpleTrajectoryPredictor.__init__` 中增加默认参数或内部属性，例如 `local_velocity_alpha=0.85`、`max_local_dt_sec=300.0`。新增参数必须有默认值，保证旧调用兼容。
- 建议添加轻量统计字段，便于调试但不影响算法行为：
  - `local_velocity_used`
  - `fusion_used`
  - `historical_fallback_used`
  - `stationary_fallback_used`
  - `dt_rejected`

**4. 验证口径修正**
- 当前项目默认 processed 数据已切换为 `data/processed/taxi_cleaned_active100_min100_eps2h_cov50.csv`，Smoke Test 不应再强制使用 `processed_csv=False` 重新跑原始清洗流程。
- 验证应优先使用 `DEFAULT_PROCESSED_TAXI_PATH` 或显式使用 cov50 processed CSV，保证与最新中等验证口径一致。
- 运行极小规模 smoke 时可以截取 Top-10 / Top-12 active taxis，避免重新全量训练。

**自动化测试与交付协议：**
1. 检查代码，确保没有修改类签名，避免破坏外部 DataLoader 调用的兼容性。
2. 运行一次极小规模或中等规模的 Smoke Test，使用 `data/processed/taxi_cleaned_active100_min100_eps2h_cov50.csv` 或 `DEFAULT_PROCESSED_TAXI_PATH`。
3. 验证标准不能只看 Train 的 **Proactive Decisions** 是否显著增长。由于当前 SLA 已违规样本仍会优先归入 Reactive，Train Proactive 可能不会暴涨。
4. 必须同时检查以下指标：
   - 预测器统计中 `fusion_used` 或 `local_velocity_used` 是否明显生效。
   - Train Proactive Decisions 是否有增长，或至少当前未违规但未来违规的候选窗口是否增加。
   - Reactive 数量是否仍占主导，避免误把“当前已违规”问题归咎于预测器。
   - 推理段迁移数量和成本没有异常暴涨。
   - 编译检查和 linter 无错误。
5. 验证通过后，进入下一轮评估！