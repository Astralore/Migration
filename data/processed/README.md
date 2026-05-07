# 预处理轨迹（清洗 + Top‑N）

本目录存放 **`load_data` 完整阶段一管线** 的输出 CSV，便于重复实验时**直接读取**，不必每次从 `taxi_with_health_info.csv` 做去重与跳变清洗。

## 生成方式（任选其一）

在项目根目录执行：

```text
python -m core.data_loader --export
```

或：

```text
python scripts/export_cleaned_taxi_data.py
```

默认输出文件：`taxi_cleaned_active100_min100_eps2h.csv`（与 `core.data_loader.DEFAULT_PROCESSED_TAXI_PATH` 一致）。

## 使用方式

`run_comparison.py`、`run_sa.py`、`run_dqn.py`、`run_hybrid.py` 已默认传入同一 `processed_csv=...`：**若该文件存在**则秒级加载；**若不存在**则自动回退为从原始 CSV 用 `active_users_limit=100`、`min_vehicle_points=100` 现场清洗（首次较慢）。

其他脚本可：

```python
from core.data_loader import load_data, DEFAULT_PROCESSED_TAXI_PATH
df = load_data(processed_csv=DEFAULT_PROCESSED_TAXI_PATH)
```

若你调整了 `v_max_kmh`、`gap_dt_hours` 或 Top‑N 参数，请重新导出并**改名或改常量**，避免与旧文件混淆。
