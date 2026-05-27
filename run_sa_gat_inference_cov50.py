"""
Inference-only SA vs GAT-MARL comparison on the Phase B cov50 test split.

Uses Phase B checkpoints for GAT-MARL (unchanged) and the updated SA baseline
(total_cost_ms objective, richer neighbourhood, restarts).
"""

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from algorithms.marl_gat import run_marl_gat_microservice
from algorithms.sa import run_sa_microservice_fair
from core.data_loader import DEFAULT_SERVER_PATH, DEFAULT_TAXI_PATH, load_data
from prediction.simple_predictor import SimpleTrajectoryPredictor
from run_comparison import _avg_total_cost_ms
from run_medium_validation_cov50 import (
    ACTIVE_USERS,
    FORECAST_HORIZON,
    PROCESSED_TAXI_PATH,
    SPLIT_SEED,
    _filter_top_active,
    _migration_efficiency_metrics,
    _migration_efficiency_row,
    _split_by_balanced_exposure,
    _summarize_result,
    _to_jsonable,
)

SOURCE_EXPERIMENT = Path(
    os.environ.get(
        "SA_GAT_INFERENCE_SOURCE",
        "experiments/medium_validation_20260526_phaseB_reward_align_v1",
    )
)
STAMP = os.environ.get(
    "SA_GAT_INFERENCE_STAMP",
    datetime.now().strftime("%Y%m%d_%H%M%S_sa_totalcost_vs_gat"),
)
OUT_DIR = Path("experiments") / f"sa_gat_inference_{STAMP}"


def _fmt(value, *, percent=False):
    if value is None:
        return "—"
    if percent:
        return f"{100.0 * value:.2f}%"
    return f"{value:.2f}"


def _compare_row(name, res, proactive=False):
    dc = int(res.get("decision_count") or 0)
    mig = _migration_efficiency_metrics(res)
    avg_sla = (float(res.get("total_sla_penalty_ms") or 0.0) / dc) if dc > 0 else 0.0
    avg_total = _avg_total_cost_ms(res)
    pro = int(res.get("proactive_decisions") or 0)
    if proactive:
        return (
            f"| {name} | {res['total_migrations']} | {res['total_violations']} | "
            f"{res.get('severe_sla_violations', 0)} | {res.get('p95_sla_excess_distance_km', 0):.2f} | "
            f"{avg_sla:.2f} | {avg_total:.2f} | {pro} | {res.get('avg_decision_time_ms', 0):.2f} | "
            f"{_fmt(mig['cost_per_migrated_node_ms'])} | {_fmt(mig['cost_per_migration_decision_ms'])} | "
            f"{_fmt(mig['migration_cost_share'], percent=True)} |\n"
        )
    return (
        f"| {name} | {res['total_migrations']} | {res['total_violations']} | "
        f"{res.get('severe_sla_violations', 0)} | {res.get('p95_sla_excess_distance_km', 0):.2f} | "
        f"{avg_sla:.2f} | {avg_total:.2f} | {res.get('avg_decision_time_ms', 0):.2f} | "
        f"{_fmt(mig['cost_per_migrated_node_ms'])} | {_fmt(mig['cost_per_migration_decision_ms'])} | "
        f"{_fmt(mig['migration_cost_share'], percent=True)} |\n"
    )


def _write_report(payload):
    pro = payload["inference"]["proactive"]
    rea = payload["inference"]["reactive"]
    cfg = payload["config"]

    pro_table = (
        "| Algorithm | Migrations | SLA Risk | Severe | P95 Excess (km) | Avg SLA Penalty (ms) | "
        "Avg Total Cost (ms) | Proactive Decisions | Avg Decision Time (ms) | Cost/Node (ms) | "
        "Cost/Mig Decision (ms) | Migration Share |\n"
        "|-----------|------------|----------|--------|-----------------|----------------------|"
        "---------------------|---------------------|------------------------|----------------|"
        "------------------------|-----------------|\n"
    )
    for name in ["SA", "GAT-MARL"]:
        if name in pro:
            pro_table += _compare_row(name, pro[name], proactive=True)

    rea_table = (
        "\n| Algorithm | Migrations | SLA Risk | Severe | P95 Excess (km) | Avg SLA Penalty (ms) | "
        "Avg Total Cost (ms) | Avg Decision Time (ms) | Cost/Node (ms) | Cost/Mig Decision (ms) | "
        "Migration Share |\n"
        "|-----------|------------|----------|--------|-----------------|----------------------|"
        "---------------------|------------------------|----------------|------------------------|"
        "-----------------|\n"
    )
    for name in ["SA", "GAT-MARL"]:
        if name in rea:
            rea_table += _compare_row(name, rea[name], proactive=False)

    mig_table = (
        "\n| Algorithm | Pro: Cost/Node | Pro: Cost/Mig Decision | Pro: Migration Share | "
        "Rea: Cost/Node | Rea: Cost/Mig Decision | Rea: Migration Share |\n"
        "|-----------|----------------|------------------------|----------------------|"
        "---------------|------------------------|----------------------|\n"
    )
    mig_table += _migration_efficiency_row("SA", pro.get("SA", {}), rea.get("SA", {}))
    mig_table += _migration_efficiency_row("GAT-MARL", pro.get("GAT-MARL", {}), rea.get("GAT-MARL", {}))

    report = f"""# SA (total_cost_ms) vs GAT-MARL 推理对比

**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**输出目录**：`{OUT_DIR}`  
**GAT-MARL checkpoint 来源**：`{cfg['source_experiment']}`  
**数据**：Top-{ACTIVE_USERS} active taxis；test={payload['data']['test_rows']} rows/{payload['data']['test_taxis']} taxis  
**SA 配置**：objective=`total_cost_ms`；max_iter={cfg['sa_max_iter']}；temp={cfg['sa_initial_temp']}；cooling={cfg['sa_cooling_rate']}；restarts={cfg['sa_num_restarts']}

## 推理 Proactive

{pro_table}

## 推理 Reactive

{rea_table}

## 迁移效率汇总

{mig_table}

## 说明

- SA 已改为最小化与报告一致的 `total_cost_ms`，邻域含单节点换服与整 DAG 共置；GAT-MARL 使用 Phase B checkpoint，逻辑未改。
- 对比重点：Avg Total Cost、Severe/P95、迁移效率三指标（Cost/Node / Cost/Mig Decision / Migration Share）。
"""
    (OUT_DIR / "result.md").write_text(report, encoding="utf-8")


def main():
    random.seed(SPLIT_SEED)
    np.random.seed(SPLIT_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    source_payload_path = SOURCE_EXPERIMENT / "results.json"
    if not source_payload_path.exists():
        raise FileNotFoundError(f"Missing source results: {source_payload_path}")

    checkpoint_dir = SOURCE_EXPERIMENT / "checkpoints"
    for name in ("marl_gat_proactive.pth", "marl_gat_reactive.pth"):
        path = checkpoint_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")

    start = time.time()
    df_all = load_data(DEFAULT_TAXI_PATH, processed_csv=PROCESSED_TAXI_PATH)
    df = _filter_top_active(df_all, ACTIVE_USERS)
    servers_df = pd.read_csv(DEFAULT_SERVER_PATH)
    train_df, test_df, train_ids, test_ids, split_meta = _split_by_balanced_exposure(df, servers_df)
    predictor = SimpleTrajectoryPredictor(forecast_horizon=FORECAST_HORIZON).fit(train_df)

    payload = {
        "config": {
            "source_experiment": str(SOURCE_EXPERIMENT),
            "active_users": ACTIVE_USERS,
            "split_seed": SPLIT_SEED,
            "forecast_horizon": FORECAST_HORIZON,
            "processed_taxi_path": PROCESSED_TAXI_PATH,
            "sa_max_iter": int(os.environ.get("SA_MAX_ITER", "150")),
            "sa_initial_temp": float(os.environ.get("SA_INITIAL_TEMP", "50000.0")),
            "sa_cooling_rate": float(os.environ.get("SA_COOLING_RATE", "0.99")),
            "sa_num_restarts": int(os.environ.get("SA_NUM_RESTARTS", "2")),
            "started_at": datetime.now().isoformat(),
        },
        "data": {
            "train_rows": int(len(train_df)),
            "train_taxis": int(train_df["taxi_id"].nunique()),
            "test_rows": int(len(test_df)),
            "test_taxis": int(test_df["taxi_id"].nunique()),
            "train_ids": sorted(map(str, train_ids)),
            "test_ids": sorted(map(str, test_ids)),
            "split": _to_jsonable(split_meta),
        },
        "inference": {"proactive": {}, "reactive": {}},
    }

    def save():
        with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)
        _write_report(payload)

    save()

    for mode, proactive in (("proactive", True), ("reactive", False)):
        print(f"\n=== INFERENCE {mode.upper()} ===", flush=True)
        target = payload["inference"][mode]

        print("  Running SA (total_cost_ms objective)...", flush=True)
        target["SA"] = _summarize_result(
            run_sa_microservice_fair(
                test_df,
                servers_df,
                predictor=predictor,
                proactive=proactive,
                collect_dag_proactive_stats=proactive,
            )
        )
        save()

        print("  Running GAT-MARL (Phase B checkpoint)...", flush=True)
        target["GAT-MARL"] = _summarize_result(
            run_marl_gat_microservice(
                test_df,
                servers_df,
                predictor=predictor,
                proactive=proactive,
                inference_mode=True,
                checkpoint_path=str(checkpoint_dir / f"marl_gat_{mode}.pth"),
                collect_dag_proactive_stats=proactive,
            )
        )
        payload["elapsed_seconds_so_far"] = time.time() - start
        save()

    payload["completed_at"] = datetime.now().isoformat()
    payload["elapsed_seconds"] = time.time() - start
    save()
    print(f"\n[DONE] SA vs GAT-MARL inference saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
