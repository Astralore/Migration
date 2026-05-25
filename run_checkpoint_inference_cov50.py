import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from algorithms.dqn import run_dqn_microservice_fair
from algorithms.marl_gat import run_marl_gat_microservice
from algorithms.nearest import run_nearest_microservice_fair
from algorithms.sa import run_sa_microservice_fair
from core.data_loader import DEFAULT_SERVER_PATH, DEFAULT_TAXI_PATH, load_data
from prediction.simple_predictor import SimpleTrajectoryPredictor
from run_comparison import _avg_total_cost_ms
from run_medium_validation_cov50 import _filter_top_active, _summarize_result, _to_jsonable


SOURCE_EXPERIMENT = Path(
    os.environ.get(
        "CHECKPOINT_INFERENCE_SOURCE_EXPERIMENT",
        "experiments/medium_validation_20260525_2241_physical_wall_v1",
    )
)
STAMP = os.environ.get(
    "CHECKPOINT_INFERENCE_STAMP",
    datetime.now().strftime("%Y%m%d_%H%M%S_from_physical_wall_v1"),
)
OUT_DIR = Path("experiments") / f"checkpoint_inference_{STAMP}"
PROCESSED_TAXI_PATH = Path(
    "data/processed/taxi_cleaned_active100_min100_eps2h_cov50.csv"
)
ACTIVE_USERS = int(os.environ.get("CHECKPOINT_INFERENCE_ACTIVE_USERS", "16"))
SPLIT_SEED = 42
FORECAST_HORIZON = 15


def _load_source_payload():
    with open(SOURCE_EXPERIMENT / "results.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _ids(values):
    return {str(v) for v in values}


def _filter_ids(df, keep_ids):
    mask = df["taxi_id"].astype(str).isin(keep_ids)
    return df[mask].reset_index(drop=True)


def _exclude_ids(df, exclude_ids):
    mask = ~df["taxi_id"].astype(str).isin(exclude_ids)
    return df[mask].reset_index(drop=True)


def _row(name, res, proactive=False):
    pro = int(res.get("proactive_decisions") or 0)
    dc = int(res.get("decision_count") or 0)
    avg_access = (float(res.get("total_access_latency") or 0.0) / dc) if dc > 0 else 0.0
    avg_system = _avg_total_cost_ms(res)
    avg_sla_penalty = (float(res.get("total_sla_penalty_ms") or 0.0) / dc) if dc > 0 else 0.0
    if proactive:
        return (
            f"| {name} | {res['total_migrations']} | {res['total_violations']} | "
            f"{res.get('severe_sla_violations', 0)} | {res.get('avg_sla_excess_distance_km', 0):.2f} | "
            f"{res.get('p95_sla_excess_distance_km', 0):.2f} | {avg_sla_penalty:.2f} | "
            f"{pro} | {res.get('avg_decision_time_ms', 0):.2f} | {avg_access:.2f} | {avg_system:.2f} |\n"
        )
    return (
        f"| {name} | {res['total_migrations']} | {res['total_violations']} | "
        f"{res.get('severe_sla_violations', 0)} | {res.get('avg_sla_excess_distance_km', 0):.2f} | "
        f"{res.get('p95_sla_excess_distance_km', 0):.2f} | {avg_sla_penalty:.2f} | "
        f"{res.get('avg_decision_time_ms', 0):.2f} | {avg_access:.2f} | {avg_system:.2f} |\n"
    )


def _cost_row(name, pro_res, rea_res):
    pro_dc = int(pro_res.get("decision_count") or 0)
    rea_dc = int(rea_res.get("decision_count") or 0)
    pro_sla = (float(pro_res.get("total_sla_penalty_ms") or 0.0) / pro_dc) if pro_dc > 0 else 0.0
    pro_mig = (float(pro_res.get("total_migration_cost") or 0.0) / pro_dc) if pro_dc > 0 else 0.0
    rea_sla = (float(rea_res.get("total_sla_penalty_ms") or 0.0) / rea_dc) if rea_dc > 0 else 0.0
    rea_mig = (float(rea_res.get("total_migration_cost") or 0.0) / rea_dc) if rea_dc > 0 else 0.0
    return f"| {name} | {pro_sla:.2f} | {pro_mig:.2f} | {rea_sla:.2f} | {rea_mig:.2f} |\n"


def _write_report(payload):
    pro = payload["inference"]["proactive"]
    rea = payload["inference"]["reactive"]

    def result_table():
        s = "| Algorithm | Migrations | SLA Risk Count | Severe SLA Violations | Avg SLA Excess (km) | P95 SLA Excess (km) | Avg SLA Penalty (ms) | Proactive Decisions | Avg Decision Time (ms) | Avg Access Latency (ms) | Avg Total System Cost (ms) |\n"
        s += "|-----------|------------|----------------|-----------------------|---------------------|---------------------|----------------------|---------------------|------------------------|-------------------------|----------------------------|\n"
        for name in ["SA", "Nearest", "DQN", "GAT-MARL"]:
            if name in pro:
                s += _row(name, pro[name], proactive=True)
        s += "\n| Algorithm | Migrations | SLA Risk Count | Severe SLA Violations | Avg SLA Excess (km) | P95 SLA Excess (km) | Avg SLA Penalty (ms) | Avg Decision Time (ms) | Avg Access Latency (ms) | Avg Total System Cost (ms) |\n"
        s += "|-----------|------------|----------------|-----------------------|---------------------|---------------------|----------------------|------------------------|-------------------------|----------------------------|\n"
        for name in ["SA", "Nearest", "DQN", "GAT-MARL"]:
            if name in rea:
                s += _row(name, rea[name], proactive=False)
        return s

    def cost_table():
        s = "| Algorithm | Proactive SLA Penalty (ms) | Proactive Migration (ms) | Reactive SLA Penalty (ms) | Reactive Migration (ms) |\n"
        s += "|-----------|----------------------------|--------------------------|---------------------------|-------------------------|\n"
        for name in ["SA", "Nearest", "DQN", "GAT-MARL"]:
            s += _cost_row(name, pro.get(name, {}), rea.get(name, {}))
        return s

    report = f"""# Checkpoint 推理对比实验

**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**输出目录**：`{OUT_DIR}`  
**复用 checkpoint**：`{SOURCE_EXPERIMENT / 'checkpoints'}`  
**数据文件**：`{PROCESSED_TAXI_PATH}`  
**数据规模**：Top-{payload['config']['active_users']} active taxis；source train={payload['data']['source_train_rows']} rows/{payload['data']['source_train_taxis']} taxis；inference={payload['data']['inference_rows']} rows/{payload['data']['inference_taxis']} taxis  
**推理数据口径**：排除上一轮训练 taxi，使用上一轮 test taxi + Top-N 中新增 taxi。  

## 推理结果

{result_table()}

## 成本分解均值

{cost_table()}

## 说明

- 本实验不重新训练 DQN/GAT-MARL，只加载上一轮 `physical_wall_v1` 保存的 checkpoint。
- SA、Nearest、DQN、GAT-MARL 使用完全相同的 inference dataframe、server dataframe 和 predictor。
- predictor 只用上一轮训练 taxi 拟合，避免用推理 taxi 轨迹泄漏未来信息。
"""
    with open(OUT_DIR / "result.md", "w", encoding="utf-8") as f:
        f.write(report)


def _save(payload):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)
    _write_report(payload)


def main():
    random.seed(SPLIT_SEED)
    np.random.seed(SPLIT_SEED)

    source = _load_source_payload()
    source_train_ids = _ids(source["data"]["train_ids"])
    source_test_ids = _ids(source["data"]["test_ids"])
    checkpoint_dir = SOURCE_EXPERIMENT / "checkpoints"
    required = [
        checkpoint_dir / "dqn_proactive.pth",
        checkpoint_dir / "dqn_reactive.pth",
        checkpoint_dir / "marl_gat_proactive.pth",
        checkpoint_dir / "marl_gat_reactive.pth",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    start = time.time()
    df_all = load_data(DEFAULT_TAXI_PATH, processed_csv=str(PROCESSED_TAXI_PATH))
    df = _filter_top_active(df_all, ACTIVE_USERS)
    servers_df = pd.read_csv(DEFAULT_SERVER_PATH)

    train_df = _filter_ids(df, source_train_ids)
    inference_df = _exclude_ids(df, source_train_ids)
    if train_df.empty or inference_df.empty:
        raise ValueError("Empty train or inference dataframe after source-id filtering.")

    predictor = SimpleTrajectoryPredictor(forecast_horizon=FORECAST_HORIZON).fit(train_df)

    inference_ids = sorted(map(str, inference_df["taxi_id"].unique()))
    payload = {
        "config": {
            "source_experiment": str(SOURCE_EXPERIMENT),
            "active_users": ACTIVE_USERS,
            "split_seed": SPLIT_SEED,
            "forecast_horizon": FORECAST_HORIZON,
            "processed_taxi_path": str(PROCESSED_TAXI_PATH),
            "started_at": datetime.now().isoformat(),
        },
        "data": {
            "source_train_rows": int(len(train_df)),
            "source_train_taxis": int(train_df["taxi_id"].nunique()),
            "inference_rows": int(len(inference_df)),
            "inference_taxis": int(inference_df["taxi_id"].nunique()),
            "source_train_ids": sorted(source_train_ids),
            "source_test_ids": sorted(source_test_ids),
            "inference_ids": inference_ids,
        },
        "inference": {"proactive": {}, "reactive": {}},
    }
    _save(payload)

    phases = [("proactive", True), ("reactive", False)]
    for mode, proactive in phases:
        target = payload["inference"][mode]

        print(f"\n=== INFERENCE {mode.upper()} ===", flush=True)
        target["SA"] = _summarize_result(
            run_sa_microservice_fair(
                inference_df,
                servers_df,
                predictor=predictor,
                proactive=proactive,
                collect_dag_proactive_stats=proactive,
            )
        )
        _save(payload)

        target["Nearest"] = _summarize_result(
            run_nearest_microservice_fair(
                inference_df,
                servers_df,
                predictor=predictor,
                proactive=proactive,
                collect_dag_proactive_stats=proactive,
            )
        )
        _save(payload)

        target["DQN"] = _summarize_result(
            run_dqn_microservice_fair(
                inference_df,
                servers_df,
                predictor=predictor,
                proactive=proactive,
                inference_mode=True,
                checkpoint_path=str(checkpoint_dir / f"dqn_{mode}.pth"),
            )
        )
        _save(payload)

        target["GAT-MARL"] = _summarize_result(
            run_marl_gat_microservice(
                inference_df,
                servers_df,
                predictor=predictor,
                proactive=proactive,
                inference_mode=True,
                checkpoint_path=str(checkpoint_dir / f"marl_gat_{mode}.pth"),
                collect_dag_proactive_stats=proactive,
            )
        )
        payload["elapsed_seconds_so_far"] = time.time() - start
        _save(payload)

    payload["completed_at"] = datetime.now().isoformat()
    payload["elapsed_seconds"] = time.time() - start
    _save(payload)
    print(f"\n[DONE] Checkpoint inference saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
