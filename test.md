# 当前轨迹预测与 Proactive 触发机制总结

本文档覆盖记录当前代码中轨迹预测、Proactive 触发判定、步长对齐修复，以及为什么当前实验中 Proactive 效果仍不明显。

## 一、当前轨迹预测逻辑

轨迹预测实现位于 `prediction/simple_predictor.py`，核心类是 `SimpleTrajectoryPredictor`。

当前预测器有两套分支：

1. **训练集中已见过的 taxi_id**

   `fit(df)` 会按 `taxi_id` 学习平均经纬度位移：

   - `dx = mean(diff(longitude))`
   - `dy = mean(diff(latitude))`

   后续 `predict_future()` 对这些已见车辆直接按平均 `dx/dy` 逐步外推：

   - 第 1 步：`lon += dx`, `lat += dy`
   - 第 2 步：继续加同样的 `dx/dy`
   - 直到 `FORECAST_HORIZON`

   这是一种长期平均运动趋势预测，不是局部速度预测。

2. **训练集中未见过的 taxi_id**

   如果当前车辆不在 `velocity_factors` 中，预测器会尝试使用上一 GPS 点与当前 GPS 点构造局部速度：

   - `v_lon = (current_lon - prev_lon) / dt_prev`
   - `v_lat = (current_lat - prev_lat) / dt_prev`
   - 每个未来步长使用 `delta_t_future_sec`

   如果没有有效上一点、episode 切换、或 `dt <= 0`，则返回原地不动预测。

因此当前预测器的一个重要特点是：**训练集车辆用平均位移，测试集中未见车辆更可能用局部速度**。这会导致训练段和推理段的预测行为不完全一致。

## 二、预测器内部原理解析

当前 `SimpleTrajectoryPredictor` 本质上不是一个学习复杂时空模式的深度模型，而是一个轻量级运动外推器。它的核心思想是：根据历史轨迹估计“每一步车辆大概往哪里移动”，再把这个位移重复应用到未来若干步。

### 1. `fit(df)` 学到了什么

`fit(df)` 遍历训练数据中的每一辆出租车。对于每个 `taxi_id`，它会按 `date_time` 排序，然后计算相邻 GPS 点之间的经纬度差：

```text
dx_i = longitude_i+1 - longitude_i
dy_i = latitude_i+1 - latitude_i
```

然后取平均：

```text
mean_dx = average(dx_i)
mean_dy = average(dy_i)
```

最终存入：

```text
velocity_factors[taxi_id] = (mean_dx, mean_dy)
```

这里的 `velocity_factors` 名字里虽然有 velocity，但对已见车辆来说，它实际保存的是**平均每个轨迹步的经纬度位移**，不是严格意义上的速度。因为它没有除以相邻点时间间隔 `dt`，所以它默认每个训练样本步长是同一种时间粒度。

### 2. 已见车辆为什么是“平均位移外推”

如果 `predict_future()` 发现当前 `taxi_id` 在 `velocity_factors` 中，就直接取出 `(dx, dy)`，然后从当前位置开始重复累加：

```text
future_lon_1 = current_lon + dx
future_lat_1 = current_lat + dy

future_lon_2 = future_lon_1 + dx
future_lat_2 = future_lat_1 + dy
```

这意味着预测轨迹是一条固定方向、固定步长的直线。

这种方法的优点是简单、稳定、计算快；缺点是它会把车辆多天、多路段、多方向的历史运动平均掉。对于出租车这种经常转弯、掉头、停靠、低速移动的轨迹，平均位移可能非常小，甚至接近“原地缓慢漂移”。

因此在训练段中，很多已见车辆即使当前正在靠近 SLA 边界，预测器也可能因为使用长期平均方向而没有及时预测到越界。

### 3. 未见车辆为什么使用“局部速度外推”

如果 `taxi_id` 不在 `velocity_factors` 中，预测器会尝试使用当前点和上一点构造局部速度：

```text
v_lon = (current_lon - prev_lon) / dt_prev
v_lat = (current_lat - prev_lat) / dt_prev
```

然后用未来步长 `delta_t_future_sec` 外推：

```text
future_lon = current_lon + v_lon * delta_t_future_sec
future_lat = current_lat + v_lat * delta_t_future_sec
```

这条分支更接近物理意义上的短时速度模型。它更容易捕捉车辆最近一次移动方向，例如刚刚从服务区内部向边界外移动。

因此推理段中，如果测试车辆没有出现在 predictor 的训练 `velocity_factors` 中，就会进入局部速度分支，反而可能比训练段更容易产生 Proactive 触发。

### 4. `taxi_last` 的作用

`taxi_last` 是仿真循环维护的短期记忆，记录每辆车上一次出现时的位置、时间和 episode：

```text
taxi_last[taxi_id] = (prev_lon, prev_lat, prev_time, prev_episode_id)
```

`build_predict_future_time_kwargs()` 会用它判断当前点是否能和上一点组成连续运动：

- 如果该车第一次出现，没有上一点，则不能估计局部速度。
- 如果 episode 发生变化，说明中间有较长时间断裂，则不能把上一 episode 的最后一点拿来估计速度。
- 如果 `dt <= 0`，说明时间异常，也不能估计速度。
- 如果连续有效，则输出上一点、当前时间、真实 `dt` 和预测步长。

这套机制避免了跨 episode 的错误速度估计，也让预测器可以使用真实采样间隔进行步长对齐。

### 5. 当前预测器对 Proactive 的影响

Proactive 触发依赖预测轨迹是否会在动态窗口内越过 SLA 阈值。因此预测器是否能准确捕捉“即将越界”非常关键。

当前预测器的影响可以概括为：

- 已见车辆：长期平均位移更平滑，但可能低估短期越界风险。
- 未见车辆：局部速度更敏感，更容易发现短期越界趋势。
- 采样间隔较粗时，车辆可能从未违规直接跳到已违规，留给 Proactive 的离散窗口较少。
- 如果预测轨迹基本停留在原地或移动很慢，`ttv_s` 就不会落入动态窗口，Proactive 不会触发。

所以当前 Proactive 不明显，不仅是触发窗口问题，也与预测器的“已见车辆平均位移分支”有关。

### 6. 当前预测器的局限

当前预测器没有使用：

- 路网方向；
- 速度平滑；
- 最近多个点的加权速度；
- 加速度或转向趋势；
- 时间段规律；
- 车辆历史轨迹聚类；
- LSTM、GRU、Transformer 等序列模型。

它更适合作为轻量 baseline，而不是强预测器。对于验证 Proactive migration 的价值，它可能过于保守，尤其是在训练段已见车辆上。

### 7. 后续改进方向

更合理的下一步不是只扩大 Proactive 触发窗口，而是改预测器：

1. 已见车辆也优先使用最近两点或最近多点的局部速度。
2. 将长期平均位移作为 fallback 或平滑项，而不是主预测。
3. 对 `delta_t_prev_sec` 过大的样本做截断，避免长间隔导致速度失真。
4. 使用“局部速度 + 历史平均”的融合预测：

   ```text
   predicted_step = alpha * local_velocity_step + (1 - alpha) * historical_mean_step
   ```

5. 在实验报告中单独统计预测分支命中情况：已见平均位移、未见局部速度、原地 fallback。

这些改动会比单纯调大 `dynamic_window_s` 更直接地提升 Proactive 的可观测性和合理性。

## 三、时间步长如何传递

每个算法循环中都会维护 `taxi_last`，并通过：

`build_predict_future_time_kwargs(taxi_last, taxi_id, row, current_lon, current_lat, current_time)`

为预测器构造时间相关参数。

当前已加入步长对齐字段：

- `delta_t_prev_sec`：当前点与上一点的真实时间间隔。
- `delta_t_future_sec`：未来每一步使用的时间间隔，默认对齐为 `delta_t_prev_sec`。
- `forecast_step_dt_sec`：传给 Proactive 触发器的预测时间步长。

这样做的目的不是硬编码 10s、15s 或 60s，而是让预测和触发窗口自动使用当前轨迹数据的真实采样间隔。

在当前 `cov50` 数据中，采样间隔的中位数约为 `60s`，95 分位可达到数分钟。因此使用真实步长比固定写死 `15s` 更符合当前数据。

## 四、Proactive / Reactive 触发逻辑

触发逻辑位于 `core/context.py`，核心函数是：

`get_trigger_type(...)`

判定顺序如下：

1. 先检查当前位置是否已经 SLA 违规：

   - 空间距离是否超过 `DISTANCE_THRESHOLD_KM = 15.0`
   - 或接入时延是否超过 `USER_SLA_TOLERANCE_MS`

2. 如果当前位置已经违规，立即返回：

   `TRIGGER_REACTIVE`

3. 只有当前位置尚未违规，并且开启 `proactive_enabled`，才会检查预测轨迹。

4. 对预测轨迹逐点计算未来到 gateway server 的距离，找到首次达到 SLA 阈值的时间：

   `ttv_s = time-to-violation`

5. 如果：

   `ttv_s <= estimated_migration_time_s + forecast_step_dt_sec`

   则返回：

   `TRIGGER_PROACTIVE`

否则返回 `None`。

## 五、步长对齐修复

本轮修改的核心是把“迁移耗时 + 一个真实数据步长”作为动态前瞻窗口。

当前代码语义等价于：

```python
dynamic_window_s = estimated_migration_time_s + max(1.0, forecast_step_dt_sec)

if np.isfinite(ttv_s) and ttv_s <= dynamic_window_s:
    return TRIGGER_PROACTIVE
```

对应修改范围：

- `prediction/simple_predictor.py`：输出真实 `forecast_step_dt_sec`。
- `core/context.py`：显式使用 `dynamic_window_s`。
- `algorithms/sa.py`：调用 `get_trigger_type()` 时传入 `forecast_step_dt_sec`。
- `algorithms/dqn.py`：调用 `get_trigger_type()` 时传入 `forecast_step_dt_sec`。
- `algorithms/hybrid_sac.py`：训练与推理两处都传入 `forecast_step_dt_sec`。

这解决的是：仿真只能在离散 GPS 点上观察未来，如果首个预测点已经违规，而窗口没有包含一个完整采样步长，则当前步骤可能不触发 Proactive，下一次观测时就变成 Reactive。

## 六、步长对齐不能完全避免 Reactive 截胡

“步长对齐”只能扩大当前未违规时的前瞻窗口，不能改变 Reactive 的优先级。

当前设计中：

- 如果当前点已经违规，必然返回 `REACTIVE`。
- 如果当前点未违规、预测未来会在迁移耗时 + 一个采样步长内违规，才返回 `PROACTIVE`。

因此它不能“完美避免 Reactive 截胡”。它只能减少这类情况：

当前点未违规 -> 预测下一个点将违规 -> 当前及时触发 Proactive。

如果数据中车辆已经处在违规状态，或者 gateway assignment 长时间未迁移导致当前距离已经超过阈值，那么这些样本仍然会被统计为 Reactive。

## 七、当前实验中 Proactive 仍不明显的原因

基于 `cov50` 中等验证结果：

`experiments/medium_validation_20260511_1432_cov50/result.md`

训练段 Proactive：

- SA：`Proactive Decisions = 1`
- DQN：`Proactive Decisions = 0`
- Hybrid SAC：`Proactive Decisions = 0`

推理段 Proactive：

- SA：`Proactive Decisions = 208`
- DQN：`Proactive Decisions = 178`
- Hybrid SAC：`Proactive Decisions = 193`

这说明问题不是 Proactive 统计完全失效，而是训练段中“当前未违规但未来即将违规”的窗口样本很少。

主要原因包括：

1. **Reactive 优先级高**

   只要当前位置已经违规，就不会再被统计为 Proactive。

2. **训练段风险样本多已是当前违规**

   即使未来预测也会违规，当前状态已违规时仍归入 Reactive。

3. **已见车辆使用平均位移预测**

   训练集中大多数车辆是 predictor 已见车辆，预测使用长期平均 `dx/dy`，对短时转向、加速、靠近边界的局部变化不敏感。

4. **推理段未见车辆更可能走局部速度分支**

   这会让推理段更容易捕捉短期移动趋势，因此出现更多 Proactive。

5. **数据采样粒度偏粗**

   当前数据中位采样间隔约 60s，很多情况下车辆从“未违规”到“已违规”的过渡只在相邻点之间出现，留给 Proactive 的离散窗口天然较窄。

## 八、当前结论

当前代码中的 Proactive 触发逻辑已经具备：

- 当前 SLA 检查；
- 预测轨迹检查；
- time-to-violation；
- 迁移耗时估计；
- GPS 步长对齐的动态前瞻窗口。

但 Proactive 效果仍不明显，核心瓶颈已经不只是触发窗口，而是：

1. 训练数据中大量风险样本已被 Reactive 先捕获；
2. 已见车辆预测器使用平均位移，短期预测能力偏弱；
3. train/test 车辆预测分支存在差异；
4. GPS 采样间隔较粗，离散观测下 proactive 可见窗口有限。

后续若要进一步提升 Proactive 效果，优先建议改进预测器：让已见车辆也使用局部速度或“局部速度 + 历史平均”的融合预测，而不是只使用全局平均 `dx/dy`。
