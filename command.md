角色设定：

你是一位顶级的 AI 数据工程师和系统架构师。为满足顶会论文对**数据协议可复现、训练/测试实体不重叠、子集与全城叙事一致**的要求，对 `core/data_loader.py` 与 `run_comparison.py` 按下列计划重构。方案仍为 **「活跃车辆子集（Strategy D）」+「按车辆零样本切分（Strategy B）」** 的组合；以下条款按**本仓库现有代码路径**做了对齐与细化，实施时以本文为准。

---

## 设计原则（与当前实现对齐）

1. **处理顺序固定**：读 CSV → 保留四列 → `date_time` 解析 → **`sort_values(['taxi_id','date_time'])`** → `dropna`（经纬度）→ **同车时间去重（§1.1）** → **速度阈值跳变清洗** → **剔除单车有效点过少车辆** → **按车计数、取 Top‑`active_users_limit` 活跃车** →（可选）**断层 `episode_id` 标注** → 返回。不在取 Top‑N 之后再做全表级排序破坏车内线序。
2. **`run_comparison` 单次加载、再切分**：全流水线中**同一份**「清洗 + Top‑N」后的 `df_active` 只通过一次 `load_data` 得到；**禁止**训练阶段与推理阶段各自调用 `load_data` 且参数不一致导致子集漂移。
3. **RL 与物理惩罚**：`algorithms/*` 内**单步奖励解析式、Hybrid SAC 核心训练循环**不修改。允许为**合法时空断层**增加**数据契约**（如 `episode_id` 或仿真外层对已存在的 \(\Delta t\) 阈值做状态重置），以避免「惩罚链不合理地跨越换班/长途」；该改动须与现有按时间步驱动逻辑对齐，**不**通过删光断层之后数据实现。
4. **数据集文件**：不删除、不覆盖 `data/` 下原始 CSV。

---

## 阶段一：重构 `core/data_loader.py`（清洗 + 活跃度提纯）

### 1.1 严谨的 GPS 异常清洗（Velocity-based Jump Filtering）

在**每辆车内部**、已按 `date_time` 排序后，**必须先执行时间去重（Drop Duplicates）**：

- **去重规则**：对同一 `taxi_id` 下 **`date_time` 完全相同**的冗余行（设备连发、重传等）予以剔除，只保留一条。实现上在排序、去经纬度空值之后，对 **`['taxi_id', 'date_time']`** 做 **`drop_duplicates(..., keep='first')`**（`keep` 写死为 `first`，并在论文脚注说明）。  
  - **目的**：保证后续相邻有效行之间 **\(\Delta t > 0\)**（在解析精度内），避免 \(\Delta t = 0\) 带来的 **除零、\(\inf\)** 及速度筛选逻辑崩溃；同时去除 \(\Delta t = 0\) 且 \(\Delta d > 0\)（同刻漂移）或 \(\Delta d = 0\)（纯重复）的病态边。  
  - **脚注级说明**：若业务上存在「同秒不同坐标」且非重复噪声，当前主协议仍只保留首条；若未来需亚秒区分，可改为更高精度时间列，**本实验不展开**。

去重完成后，再对相邻两行 \((i, i+1)\) 计算：

- **时间差** \(\Delta t\)（秒；在去重后主路径上应 **> 0**；若仍出现 \(\Delta t \le 0\) 的边角，**跳过该边**不参与速度判定、不除零）与 **Haversine 距离** \(\Delta d\)（公里）。
- **异常漂移剔除（毛刺）**：瞬时速度 \(v = (\Delta d / \Delta t) \times 3600\)（km/h）。当 **\(v > v_{\max}\)**（默认 **`v_max_kmh = 200`**，可配置）时，视为 GPS 定位漂移，**剔除该步的终点行**（与「荒谬边」的后一点一致）；可循环扫描直至稳定或设最大轮次。
- **合法物理断层（换班 / 长途 / 长时间静止后再出发）**：若 **\(v \le v_{\max}\)**，但 **\(\Delta t\)** 超过**断层阈值**（默认 **`gap_dt_hours = 2`**，可配置），或 \(\Delta d\) 极大却与 \(\Delta t\) 一致地对应「合理低速」——**禁止**为「去毛刺」而删除断层**之后**的轨迹；不得把后续数据整体判废。
- **断层与仿真/惩罚的衔接（二选一或组合，实施前在代码中核查后选定）**：
  - **优先**：若环境/外层循环**已对极大 \(\Delta t\)** 做**状态或放置重置**（不把上一刻物理量硬接到下一刻），则数据侧可**仅保留清洗后的连续行**，依赖现有逻辑即可。
  - **否则**：在 DataFrame 上为每条记录增加 **`episode_id`**（建议规则：同 `taxi_id` 内，每当出现「合法断层」边则在后一行递增 `episode_id`），由仿真按 **`(taxi_id, episode_id)` 或按 `episode_id` 分段** 重置与物理惩罚相关的状态，使**惩罚链不跨越时空**；**不删除**断层后数据。
- Haversine 与速度判定用 **纯 NumPy** 实现即可。

### 1.2 高质量活跃子集（Strategy D）

- 新增参数 **`active_users_limit`**（默认 **`100`**；`None` 表示不截车、保留当前清洗后全部车辆）。
- 新增参数 **`min_vehicle_points`**（默认 **`100`**，与 `SimpleTrajectoryPredictor.fit` 内门槛对齐）：在跳变清洗之后、Top‑N 之前，**先剔除**「单车总有效点数 **< `min_vehicle_points`**」的劣质实体。
- 再按 **`taxi_id` 分组计数**（行数），**降序**取前 **`active_users_limit`** 个 `taxi_id`，只保留这些车的所有行。
- 与现有 **`sample_fraction`** 的约定：若论文主实验以「活跃 Top‑N」为准，则 **`run_comparison` 路径下二者互斥**——调用时 **`sample_fraction=1.0`** 且仅用 `active_users_limit`；若未来需随机子样车，在文档中写明顺序为「先 `sample_fraction` 抽车 → 再清洗 → 再滤点 → 再 Top‑N」，**本对比实验不采用该组合**。

### 1.3 行级索引参数 `start_index` / `end_index`（与旧脚本兼容）

- **不再作为 `run_comparison` 的主协议**；论文与默认配置不描述行号切分。
- **建议保留**函数参数并标注 **`legacy`**：仅在显式传入时于「取活跃 Top‑N **之后**」再 `iloc[start:end]`，用于与历史结果对照或快速 smoke test；默认全为 `None`。
- **`chunk_size`**：保留为开发用截断；**主实验关闭**。

### 1.4 返回值与可观测性

- 返回 **DataFrame**；列至少包含 `taxi_id`, `date_time`, `latitude`, `longitude`；若启用断层策略，增加 **`episode_id`**（整型）。
- 日志建议打印：原始行数、**时间去重删除行数**、剔除跳变行数、因点数不足剔除车辆数、最终车辆数、最终记录数。

---

## 阶段二：重构 `run_comparison.py`（Strategy B + 与预测器衔接）

### 2.1 统一数据入口

- 删除对 **`TRAIN_START_INDEX` / `TRAIN_END_INDEX` / `TEST_START_INDEX` / `TEST_END_INDEX`** 的依赖（常量可删或改为仅文档注释中的「历史行为」）。
- 启动时：`df_active = load_data(DEFAULT_TAXI_PATH, sample_fraction=1.0, active_users_limit=100, start_index=None, end_index=None, chunk_size=None)`（参数名以实际实现为准）。

### 2.2 可复现的 Train / Test 按车划分

- `unique_ids = df_active['taxi_id'].unique()`；断言数量与 `active_users_limit` 一致（或 ≤ limit 若数据不足）。
- 使用 **`numpy.random.default_rng(SPLIT_SEED)`**（建议 **`SPLIT_SEED = 42`**）对 `unique_ids` **打散后**按 **80% / 20%** 划分为 `train_taxi_ids` / `test_taxi_ids`（整除时注意余数分配规则写死，例如前 80% 为 train）。
- `train_df = df_active[df_active['taxi_id'].isin(train_taxi_ids)]`，`test_df` 同理；**两车集合不相交**。

### 2.3 训练阶段（`run_training_phase`）

- **`SimpleTrajectoryPredictor`**：**仅在 `train_df` 上 `fit`**。
- **SA / DQN / Hybrid SAC（训练）**：**仅在 `train_df` 上**运行（与当前「整段 df 训 RL」结构一致，只是 df 语义改为「训练车全时段轨迹」）。
- 保存 SAC checkpoint 等行为不变。

### 2.4 推理阶段（`run_inference_phase`）

- **不再第二次调用 `load_data` 生成不同子集**；在进程内复用**同一 `df_active`**（或从 `run_training_phase` 返回的划分结果 / 序列化 seed 可重算），保证与训练阶段同一数据协议。
- **`predictor`**：仍 **仅在 `train_df` 上 `fit`**（与现逻辑一致：推理前用训练车拟合预测器）。
- **评测用 `test_df`**：所有算法入口传入的仿真 **`df` 为 `test_df`**（零样本新车 + 未见该 `taxi_id` 于 predictor 拟合集）。

### 2.5 预测器在测试车上的行为（时间归一化局部运动学，推荐且必须）

真实轨迹**采样间隔非均匀**：若将相邻点的 **\((dx, dy) = (lon_t - lon_{t-1}, lat_t - lat_{t-1})\)** 直接当作「下一步位移」叠加，而环境前瞻对应的是**更短或不同的 wall-clock**，会产生**尺度崩塌**（例如 \(\Delta t_{\text{prev}} = 60\text{s}\) 的位移被误用到 **10 s** 前瞻上放大数倍）。

**本计划采用且必须实现方案 A（唯一主方案）——标准速度向量 + 按前瞻时间积分**：

- 修改 **`SimpleTrajectoryPredictor.predict_future`**：当 **`taxi_id ∉ velocity_factors`** 时，**不**查全局字典中该车条目。使用调用方提供的 **当前步与上一步**的经纬度及时间（或等价地提供 **`Δt_prev > 0`（秒）`**），先算**每秒变化率**（标准速度向量，单位：经度/秒、纬度/秒）：
  \[
  (v_{\text{lon}}, v_{\text{lat}}) = \left(\frac{lon_t - lon_{t-1}}{Δt_{\text{prev}}},\ \frac{lat_t - lat_{t-1}}{Δt_{\text{prev}}}\right),\quad Δt_{\text{prev}} = (t - t_{\text{prev}})\ \text{（秒）}.
  \]
- **未来坐标**：对需要前瞻的 wall-clock 跨度 **`Δt_future`**（或由多步 **`Δt_{future,k}`** 组成的序列），用
  \[
  lon_{\text{future}} = lon_t + v_{\text{lon}} \times Δt_{\text{future}},\quad lat_{\text{future}} = lat_t + v_{\text{lat}} \times Δt_{\text{future}}
  \]
  做外推；**多步**时对各步 **`Δt_{future,k}` 分别乘同一 \((v_{\text{lon}}, v_{\text{lat}})\)** 再累加，或按实现用「当前点 + 段内 \(v \times Δt\)」迭代推进，**禁止**在无时间信息时假定「一步 = 固定秒」。
- **与现有 `steps` 接口衔接**：若当前 API 仍以 **`steps`**（整数步数）为主，调用方必须同时传入 **与各步对应的环境 wall-clock 间隔**（标量「每步 \(Δt\)」或逐点数组），使 **`steps` 仅表示重复外推次数**，每一步乘的 **`Δt_future`** 有明确定义；否则须在环境中将 `steps` 显式绑定到秒级 `Δt_future` 并在论文写清。

**回合边界与「幽灵上一步」（必须）**：

- **调用方改造**：在 **`sa.py` / `dqn.py` / `hybrid_sac.py`** 中，仿真循环内记录并传入 **上一时刻经纬度与时间**（或 **`Δt_prev`**）。  
- **极其重要**：当检测到 **环境 reset**、或 **当前步为新 `episode_id` 的回合第一步**（与 §1.1 合法物理断层对齐）、或 **当前 `taxi_id` 与上一记录不连续** 时，必须将 **`prev_lon` / `prev_lat` / `prev_time`（或 `Δt_prev`）设为 `None`**，或等价地将 **`prev_*` 视为与当前坐标/时间相同** 且 **`Δt_prev` 不可用**，**明确走「无历史 / 原地」分支**；**禁止**用跨断层、跨夜的上一坐标计算瞬时速度，避免「幽灵上一步」。

**无历史的第一步**：`prev_*` 缺失或 `Δt_prev` 无效时，退化为**原地不动**（与现实现对「无信息」一致）。

**接口（示意）**：`predict_future(..., prev_lon=, prev_lat=, prev_time=, current_time=)` 或传入 **`delta_t_prev_sec`**；未来步需 **`delta_t_future_sec`**（逐步或标量）。当 **`taxi_id ∈ velocity_factors`** 时，可保持现有字典支路；若希望全路径时间一致，**可选**将字典支路也升级为同一秒级外推（**非硬性**，以免扩大改动面）。

**论文表述**：在方法节将上述约定命名为 **「对未见实体的 Local Kinematic Estimation（局部运动学估计，时间归一化）」**。

**方案 B（仅作消融）**：全程原地前瞻；须在附录标注，**不作为主结果协议**。

### 2.6 与 `fit` 内 `len(taxi_data) < 100` 的交互

- **数据侧**：§1.2 已在 Top‑N 前剔除 **< `min_vehicle_points`** 的车，与 `fit` 门槛一致，减少「无字典条目」的训练车。
- **训练侧**：`fit` 内仍可保留「单车 **< 100** 不写入 `velocity_factors`」作为保险；此类车若在训练集中仍存在，**优先依赖日志告警**；其前瞻在训练中可走 **§2.5** 的时间归一化局部支路（须传 **`prev_*` + 时间或 `Δt_prev`**，并遵守回合边界）或原地。

---

## 阶段三：实验报告与打印（`result.md` / 控制台）

- **`generate_experiment_report` / `generate_full_pipeline_report`** 中「数据范围」文案：改为描述 **「活跃 Top‑N + 按车 80/20 + seed」**，例如：`active_users_limit=100, train_taxis=80, test_taxis=20, split_seed=42`，**禁止**再写 `[0:10000)` 等行号切片语义为主协议。
- 若保留 legacy 行切片对照跑，在报告中单独一行标注 **legacy**。

---

## 阶段四：其它入口（非阻塞，建议后续统一）

- `run_sa.py` / `run_dqn.py` / `run_hybrid.py` 仍使用 `chunk_size`；与主论文 **`run_comparison`** 协议不一致时，在 README 或脚本头注释中说明；**可选**后续改为调用同一 `load_data(..., active_users_limit=...)`  helper。

---

## 实施验收清单

- [ ] `load_data`：顺序符合 §设计原则；**同车 `date_time` 去重** → **速度阈值**毛刺剔除 + **合法断层**不删后续；**先 `min_vehicle_points` 滤车再 Top‑N**；`episode_id` 或环境 \(\Delta t\) 重置已按 §1.1 落实其一；无除零 / `inf` 泄漏。
- [ ] `run_comparison`：无 `TRAIN_*_INDEX`/`TEST_*_INDEX` 主路径；`df_active` 单次一致；`train_df`/`test_df` 车不交。
- [ ] Predictor：**§2.5 时间归一化局部运动学**；`v_lon/v_lat` 由 **`Δt_prev`** 定义；前瞻乘 **`Δt_future`**；三算法传入 **`prev_*`+时间**；**reset / 新 `episode_id` / 轨迹不连续** 时不传幽灵 `prev_*`。
- [ ] 报告：`result.md` 数据协议描述与代码一致。
- [ ] 原始数据文件未删除。

---

## 交付物（供自检 / 文档）

实施完成后，在 PR 或附录中**贴出或引用**：

1. `data_loader.py` 中 **`taxi_id`+`date_time` 去重 + Haversine 速度阈值清洗 + Top‑活跃车** 的核心片段。
2. `run_comparison.py` 中 **`df_active` → `train_df`/`test_df` 划分 + 训练/推理各传入哪张表** 的核心片段。
3. `simple_predictor.py` 中 **时间归一化 Local Kinematic（`v_lon/v_lat`、`Δt_prev`/`Δt_future`、回合边界）** 片段，以及 **`sa`/`dqn`/`hybrid_sac`** 传入 **`prev_*`+时间** 与 **reset/`episode_id` 置空** 的片段。

---

**硬性约束**：不修改强化学习奖励、物理惩罚与 Hybrid SAC 核心训练循环逻辑；不改变原始数据集文件；本文件为实施蓝本，与历史 `command.md` 中「EVAL_END_INDEX」等笔误以**仓库内实际符号（如 `TEST_*`）及本文**为准。
