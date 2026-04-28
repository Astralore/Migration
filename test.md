# 数据管线与阶段二实施记录（`command.md`）

---

## 阶段一（简要）：`core/data_loader.py`

**范围**：仅 `core/data_loader.py`；未改 CSV。要点：同车 `(taxi_id, date_time)` 去重 → Haversine + \(v>200\) km/h 多轮删点 → `min_vehicle_points` / `active_users_limit` → `episode_id`（>2 h 断层）。默认 `active_users_limit=None` 等保持旧调用可选；主实验显式传 `100`。

验证：`python core/data_loader.py`。典型规模：约 100 车、约 20.7 万行（见前文表）。

---

## 阶段二：`run_comparison.py` + 预测器 + 三算法调用

### 修改范围（**未**再改 `core/data_loader.py` 与原始 CSV）

| 文件 | 内容 |
|------|------|
| `run_comparison.py` | 删除 `TRAIN_*_INDEX` / `TEST_*_INDEX`；`load_data(..., active_users_limit=100, min_vehicle_points=100)` 得 `df_active`；`default_rng(42)` 80/20 划分 `train_df`/`test_df`；训练全程 `train_df`，推理全程 `test_df`；流水线推理复用训练返回的划分（避免重复语义）；报告文案改为 Strategy B 描述 |
| `prediction/simple_predictor.py` | `touch_taxi_last` / `build_predict_future_time_kwargs`（`episode_id` 变化则清空 prev，防幽灵上一步）；`predict_future` 对**未见** `taxi_id`：\((v_{lon},v_{lat})=(\Delta lon/\Delta t_{prev},\Delta lat/\Delta t_{prev})\)，每步 `lon += v_lon * Δt_future`（默认 `Δt_future=\Delta t_{prev}`）；无历史则原地；已知车仍用原字典 `(dx,dy)` 步进；`**_kwargs` 吸收多余关键字 |
| `algorithms/sa.py` | `taxi_last` + 每步 `pf_kw` 传入 `predict_future`；init / 无 trigger / 正常决策路径均 `touch_taxi_last` |
| `algorithms/dqn.py` | 同上 |
| `algorithms/hybrid_sac.py` | 同上；**每个 epoch 开始** `taxi_last = {}`；`evaluate_sac_policy` 同样逻辑 |

### `run_comparison.py` 核心逻辑（摘录）

```python
SPLIT_SEED = 42
ACTIVE_USERS_LIMIT = 100
MIN_VEHICLE_POINTS = 100

def _split_train_test_taxis(df_active):
    shuffled = np.random.default_rng(SPLIT_SEED).permutation(df_active["taxi_id"].unique())
    n_train = int(np.floor(0.8 * len(shuffled)))
    train_ids = set(shuffled[:n_train])
    test_ids = set(shuffled[n_train:])
    ...

def run_training_phase(servers_df):
    df_active = load_data(..., active_users_limit=100, min_vehicle_points=100)
    train_df, test_df, _, _ = _split_train_test_taxis(df_active)
    predictor.fit(train_df)
    run_sa_microservice_fair(train_df, ...); ...  # 全部 train_df

def run_inference_phase(servers_df, train_df=None, test_df=None):
    ...
    predictor.fit(train_df)
    df = test_df
    run_*_microservice_fair(df, ...)  # 仅 test_df
```

---

## 阶段二 Smoke Test（命令行）

**命令**（项目根目录）：

```text
python -c "from core.data_loader import load_data, DEFAULT_TAXI_PATH; from run_comparison import _split_train_test_taxis, ACTIVE_USERS_LIMIT; df=load_data(DEFAULT_TAXI_PATH, active_users_limit=ACTIVE_USERS_LIMIT, min_vehicle_points=100); tr, te, a, b=_split_train_test_taxis(df); print('taxis', len(a), len(b), 'rows', len(tr), len(te)); from prediction.simple_predictor import SimpleTrajectoryPredictor; import pandas as pd; p=SimpleTrajectoryPredictor(3); p.fit(tr); t0=pd.Timestamp('2008-01-01 00:00:00'); t1=pd.Timestamp('2008-01-01 00:05:00'); r=p.predict_future(116.5, 39.9, list(te['taxi_id'].unique())[0], steps=2, prev_lon=116.0, prev_lat=39.9, prev_time=t0, current_time=t1); print('pred', r)"
```

**实际输出（节选）**：

```text
taxis 80 20 rows 159333 48085
pred [(117.0, 39.9), (117.5, 39.9)]
```

- **80 / 20 辆车**，**train 行数 159333**、**test 行数 48085**（与当前 `data/taxi_with_health_info.csv` 及阶段一清洗一致）。
- **未见车**在 \(\Delta t_{prev}=300\) s、\(\Delta lon=0.5\)° 下两步外推：经度每步 +0.5°（`delta_t_future` 默认等于 `delta_t_prev`）。

**模块导入**：

```text
python -c "import run_comparison; print('import ok')"
```

输出：`import ok`。

---

## 未执行项

- 未跑完整 `python run_comparison.py --pipeline`（全量 SA/DQN/Hybrid SAC 多 epoch 耗时较长）；需要完整端到端指标时在本地执行即可。

---

*阶段一完整实现见 `core/data_loader.py`；阶段二见 `run_comparison.py` 与 `prediction/simple_predictor.py` 及三算法文件。*
