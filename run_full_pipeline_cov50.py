import json
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithms.dqn import run_dqn_microservice_fair
from algorithms.marl_gat import run_marl_gat_microservice
from algorithms.nearest import run_nearest_microservice_fair
from algorithms.sa import run_sa_microservice_fair
from core.context import DISTANCE_THRESHOLD_KM
from core.data_loader import DEFAULT_SERVER_PATH, DEFAULT_TAXI_PATH, load_data
from core.geo import haversine_distance
from prediction.simple_predictor import SimpleTrajectoryPredictor
from run_comparison import _avg_total_cost_ms, build_dag_adaptive_dual_appendix


STAMP = os.environ.get(
    "FULL_PIPELINE_STAMP",
    datetime.now().strftime("%Y%m%d_%H%M%S_cov50_stage9_full"),
)
OUT_DIR = os.path.join("experiments", f"full_pipeline_{STAMP}")
CHECKPOINT_DIR = os.path.join(OUT_DIR, "checkpoints")
PROCESSED_TAXI_PATH = os.path.join(
    "data",
    "processed",
    "taxi_cleaned_active100_min100_eps2h_cov50.csv",
)

ACTIVE_USERS = 100
SPLIT_SEED = 42
FORECAST_HORIZON = 15
MARL_EPOCHS_PROACTIVE = 8  # 与中等规模实验保持一致
MARL_EPOCHS_REACTIVE = 8   # 与中等规模实验保持一致


def _prepare_dirs():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def _checkpoint_path(name):
    return os.path.join(CHECKPOINT_DIR, name)


DQN_CHECKPOINT_PROACTIVE = _checkpoint_path("dqn_proactive.pth")
DQN_CHECKPOINT_REACTIVE = _checkpoint_path("dqn_reactive.pth")
MARL_CHECKPOINT_PROACTIVE = _checkpoint_path("marl_gat_proactive.pth")
MARL_CHECKPOINT_REACTIVE = _checkpoint_path("marl_gat_reactive.pth")


def _to_jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > 50:
            return {
                "len": len(value),
                "head": [_to_jsonable(x) for x in value[:5]],
                "tail": [_to_jsonable(x) for x in value[-5:]],
            }
        return [_to_jsonable(x) for x in value]
    return value


def _summarize_result(res):
    keys = [
        "total_migrations",
        "total_violations",
        "primary_entry_violations",
        "max_entry_violations",
        "severe_sla_violations",
        "total_sla_excess_distance_km",
        "avg_sla_excess_distance_km",
        "p95_sla_excess_distance_km",
        "proactive_decisions",
        "decision_count",
        "total_reward",
        "avg_decision_time_ms",
        "total_cost_ms_sum",
        "total_sla_penalty_ms",
        "total_tearing_penalty_ms",
        "total_future_penalty_ms",
        "total_access_latency",
        "total_communication_cost",
        "total_migration_cost",
        "dag_proactive_migration_stats",
        "eval_action_counts",
        "eval_action_migration_counts",
        "eval_follow_sa_stay",
        "q_filter_checked",
        "q_filter_passed",
        "q_filter_blocked",
        "q_filter_blocked_ratio",
        "eval_action_prob_sums",
        "eval_action_prob_means",
        "eval_q_sums",
        "eval_q_means",
        "eval_sa_migrate_action_counts",
        "eval_sa_stay_action_counts",
        "avg_agents_per_decision",
        "controlled_agents_per_decision",
        "pinned_agents_per_decision",
        "avg_migrated_agents_per_decision",
        "avg_controlled_migrated_agents_per_decision",
        "controlled_migrations",
        "all_agents_migrated_decisions",
        "all_agents_migrated_ratio",
        "controlled_all_agents_migrated_decisions",
        "controlled_all_agents_migrated_ratio",
        "stay_action_ratio",
        "candidate_action_counts",
        "joint_action_distribution",
        "local_migration_cost_sum",
        "edge_split_cost_sum",
        "dense_distance_bonus_sum",
        "entry_sla_bonus_sum",
        "proactive_logit_bias_count",
        "proactive_best_bias_action_counts",
        "reactive_action_clipped_count",
        "counterfactual_score_sum",
        "entry_node_migration_count",
        "sla_improving_action_count",
        "cost_guard_blocked_count",
        "non_entry_distance_only_blocked_count",
        "cost_by_dag_complexity",
        "cost_by_dag_type",
        "migrations_by_dag_type",
        "controlled_migrations_by_dag_type",
        "invalid_action_masked_count",
        "action_mask_fallback_count",
        "lambda_migration",
        "lambda_split",
    ]
    out = {k: _to_jsonable(res.get(k)) for k in keys if k in res}
    out["avg_total_cost_ms"] = _avg_total_cost_ms(res)
    return out


def _nearest_server_distances_km(df, servers_df, chunk_rows=5000):
    server_lats = servers_df["latitude"].to_numpy(dtype=np.float64, copy=False)
    server_lons = servers_df["longitude"].to_numpy(dtype=np.float64, copy=False)
    lat = df["latitude"].to_numpy(dtype=np.float64, copy=False)
    lon = df["longitude"].to_numpy(dtype=np.float64, copy=False)
    out = np.empty(len(df), dtype=np.float64)
    for start in range(0, len(df), int(chunk_rows)):
        end = min(start + int(chunk_rows), len(df))
        dists = haversine_distance(
            lat[start:end, None],
            lon[start:end, None],
            server_lats[None, :],
            server_lons[None, :],
        )
        out[start:end] = np.min(dists, axis=1)
    return out


def _taxi_exposure_stats(df, servers_df):
    rows = []
    for taxi_id, group in df.groupby("taxi_id", sort=True):
        nearest = _nearest_server_distances_km(group, servers_df)
        hard_risk = float(np.sum(nearest > DISTANCE_THRESHOLD_KM))
        boundary_risk = float(np.sum((nearest > 10.0) & (nearest <= DISTANCE_THRESHOLD_KM)))
        rows.append(
            {
                "taxi_id": taxi_id,
                "rows": int(len(group)),
                "risk_score": hard_risk + 0.25 * boundary_risk,
                "nearest_over_15": int(hard_risk),
                "nearest_10_to_15": int(boundary_risk),
                "nearest_p50": float(np.percentile(nearest, 50)),
                "nearest_p95": float(np.percentile(nearest, 95)),
            }
        )
    return pd.DataFrame(rows)


def _split_by_balanced_exposure(df, servers_df, test_ratio=0.2):
    stats = _taxi_exposure_stats(df, servers_df)
    target_test_taxis = max(1, int(round(len(stats) * test_ratio)))
    target_row_ratio = float(test_ratio)
    target_risk_ratio = float(test_ratio)
    total_rows = float(stats["rows"].sum())
    total_risk = float(stats["risk_score"].sum())

    rng = np.random.default_rng(SPLIT_SEED)
    candidates = stats.sample(frac=1.0, random_state=SPLIT_SEED).sort_values(
        ["risk_score", "rows"], ascending=[False, False]
    )
    selected = []
    selected_rows = 0.0
    selected_risk = 0.0
    remaining = candidates.to_dict("records")

    while len(selected) < target_test_taxis and remaining:
        best_idx = None
        best_score = None
        for idx, row in enumerate(remaining):
            new_rows = selected_rows + float(row["rows"])
            new_risk = selected_risk + float(row["risk_score"])
            row_ratio = new_rows / total_rows if total_rows > 0 else 0.0
            risk_ratio = new_risk / total_risk if total_risk > 0 else row_ratio
            count_ratio = (len(selected) + 1) / len(stats)
            score = (
                abs(row_ratio - target_row_ratio)
                + abs(risk_ratio - target_risk_ratio)
                + 0.25 * abs(count_ratio - test_ratio)
                + float(rng.random()) * 1e-9
            )
            if best_score is None or score < best_score:
                best_idx = idx
                best_score = score
        chosen = remaining.pop(best_idx)
        selected.append(chosen["taxi_id"])
        selected_rows += float(chosen["rows"])
        selected_risk += float(chosen["risk_score"])

    test_ids = set(selected)
    train_ids = set(stats["taxi_id"]) - test_ids
    train_df = df[df["taxi_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["taxi_id"].isin(test_ids)].reset_index(drop=True)
    split_meta = {
        "method": "greedy_balanced_by_nearest_server_exposure",
        "target_test_ratio": test_ratio,
        "test_taxi_ratio": len(test_ids) / len(stats),
        "test_row_ratio": selected_rows / total_rows if total_rows > 0 else 0.0,
        "test_risk_ratio": selected_risk / total_risk if total_risk > 0 else 0.0,
        "taxi_exposure": stats.to_dict("records"),
    }
    return train_df, test_df, train_ids, test_ids, split_meta


def _row(name, res, proactive=False):
    pro = int(res.get("proactive_decisions") or 0)
    dc = int(res.get("decision_count") or 0)
    avg_access = (float(res.get("total_access_latency") or 0.0) / dc) if dc > 0 else 0.0
    avg_system = _avg_total_cost_ms(res)
    if proactive:
        return (
            f"| {name} | {res['total_migrations']} | {res['total_violations']} | "
            f"{pro} | {res.get('avg_decision_time_ms', 0):.2f} | {avg_access:.2f} | {avg_system:.2f} |\n"
        )
    return (
        f"| {name} | {res['total_migrations']} | {res['total_violations']} | "
        f"{res.get('avg_decision_time_ms', 0):.2f} | {avg_access:.2f} | {avg_system:.2f} |\n"
    )


def _write_payload(payload):
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)
    _write_report(payload)


def _write_cost_decomposition_chart(payload, stage, filename):
    phases = [("proactive", payload[stage]["proactive"]), ("reactive", payload[stage]["reactive"])]
    algorithms = ["SA", "Nearest", "DQN", "GAT-MARL"]
    components = [
        ("SLA penalty", "total_sla_penalty_ms"),
        ("Migration", "total_migration_cost"),
    ]
    colors = ["#d62728", "#ff7f0e"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (mode, results) in zip(axes, phases):
        x = np.arange(len(algorithms))
        bottoms = np.zeros(len(algorithms), dtype=float)
        for (label, key), color in zip(components, colors):
            values = []
            for name in algorithms:
                res = results.get(name, {})
                dc = int(res.get("decision_count") or 0)
                values.append((float(res.get(key) or 0.0) / dc) if dc > 0 else 0.0)
            ax.bar(x, values, bottom=bottoms, label=label, color=color)
            bottoms += np.asarray(values, dtype=float)
        ax.set_title(f"{stage.capitalize()} {mode.capitalize()}")
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=20, ha="right")
        ax.set_ylabel("Avg cost per decision (ms)")
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return filename


def _write_report(payload):
    train_pro = payload["train"]["proactive"]
    train_rea = payload["train"]["reactive"]
    infer_pro = payload["inference"]["proactive"]
    infer_rea = payload["inference"]["reactive"]
    cost_chart = _write_cost_decomposition_chart(
        payload, "inference", "cost_decomposition_inference.png"
    )

    def table(pro, rea):
        s = "| Algorithm | Migrations | Violations | Proactive Decisions | Avg Decision Time (ms) | Avg Access Latency (ms) | Avg Total System Cost (ms) |\n"
        s += "|-----------|------------|------------|---------------------|------------------------|-------------------------|----------------------------|\n"
        for name in ["SA", "Nearest", "DQN", "GAT-MARL"]:
            s += _row(name, pro[name], proactive=True) if name in pro else f"| {name} | — | — | — | — | — | — |\n"
        s += "\n| Algorithm | Migrations | Violations | Avg Decision Time (ms) | Avg Access Latency (ms) | Avg Total System Cost (ms) |\n"
        s += "|-----------|------------|------------|------------------------|-------------------------|----------------------------|\n"
        for name in ["SA", "Nearest", "DQN", "GAT-MARL"]:
            s += _row(name, rea[name], proactive=False) if name in rea else f"| {name} | — | — | — | — | — |\n"
        return s

    def cost_table(pro, rea):
        s = "| Algorithm | Proactive SLA Penalty (ms) | Proactive Migration (ms) | Reactive SLA Penalty (ms) | Reactive Migration (ms) |\n"
        s += "|-----------|----------------------------|--------------------------|---------------------------|-------------------------|\n"
        for name in ["SA", "Nearest", "DQN", "GAT-MARL"]:
            pro_res = pro.get(name, {})
            rea_res = rea.get(name, {})
            pro_dc = int(pro_res.get("decision_count") or 0)
            rea_dc = int(rea_res.get("decision_count") or 0)
            pro_sla = (float(pro_res.get("total_sla_penalty_ms") or 0.0) / pro_dc) if pro_dc > 0 else 0.0
            pro_mig = (float(pro_res.get("total_migration_cost") or 0.0) / pro_dc) if pro_dc > 0 else 0.0
            rea_sla = (float(rea_res.get("total_sla_penalty_ms") or 0.0) / rea_dc) if rea_dc > 0 else 0.0
            rea_mig = (float(rea_res.get("total_migration_cost") or 0.0) / rea_dc) if rea_dc > 0 else 0.0
            s += f"| {name} | {pro_sla:.2f} | {pro_mig:.2f} | {rea_sla:.2f} | {rea_mig:.2f} |\n"
        return s

    split = payload["data"]["split"]
    report = f"""# 全量 cov50 训练与推理实验报告

**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**输出目录**：`{OUT_DIR}`  
**数据文件**：`{payload['config']['processed_taxi_path']}`  
**数据规模**：Top-{ACTIVE_USERS} active taxis；train={payload['data']['train_rows']} rows/{payload['data']['train_taxis']} taxis；test={payload['data']['test_rows']} rows/{payload['data']['test_taxis']} taxis  
**切分方式**：{split['method']}；test taxi ratio={split['test_taxi_ratio']:.3f}；test row ratio={split['test_row_ratio']:.3f}；test risk ratio={split['test_risk_ratio']:.3f}  
**GAT-MARL epochs**：Proactive={MARL_EPOCHS_PROACTIVE}，Reactive={MARL_EPOCHS_REACTIVE}  
**Checkpoint 目录**：`{CHECKPOINT_DIR}`

## 训练段

{table(train_pro, train_rea)}

## 推理段

{table(infer_pro, infer_rea)}

## 成本分解堆叠图

![Inference Cost Decomposition]({cost_chart})

## 推理阶段成本分解均值

{cost_table(infer_pro, infer_rea)}

## 说明

- 本轮显式使用 cov50 清洗数据，不读取旧 cleaned CSV。
- DQN / GAT-MARL checkpoint 均写入本实验目录，推理阶段只读取本轮新训练权重。
- 结果会在每个算法完成后写入 `results.json` 和本报告，便于长任务中断后检查进度。
- Avg Decision Time 是算法计算耗时；Avg Access Latency 是真实接入延迟；Avg Total System Cost 是包含 SLA penalty 和迁移等代价的综合优化目标。

"""
    if "SA" in infer_pro or "GAT-MARL" in infer_pro or "Nearest" in infer_pro:
        report += build_dag_adaptive_dual_appendix(infer_pro, section_heading="## DAG Proactive 迁移统计")
    with open(os.path.join(OUT_DIR, "result.md"), "w", encoding="utf-8") as f:
        f.write(report)


def _make_payload(df, train_df, test_df, train_ids, test_ids, split_meta):
    return {
        "config": {
            "active_users": ACTIVE_USERS,
            "split_seed": SPLIT_SEED,
            "forecast_horizon": FORECAST_HORIZON,
            "marl_epochs_proactive": MARL_EPOCHS_PROACTIVE,
            "marl_epochs_reactive": MARL_EPOCHS_REACTIVE,
            "processed_taxi_path": PROCESSED_TAXI_PATH,
            "checkpoint_dir": CHECKPOINT_DIR,
            "started_at": datetime.now().isoformat(),
        },
        "data": {
            "rows": int(len(df)),
            "taxis": int(df["taxi_id"].nunique()),
            "train_rows": int(len(train_df)),
            "train_taxis": int(train_df["taxi_id"].nunique()),
            "test_rows": int(len(test_df)),
            "test_taxis": int(test_df["taxi_id"].nunique()),
            "train_ids": sorted(map(str, train_ids)),
            "test_ids": sorted(map(str, test_ids)),
            "split": _to_jsonable(split_meta),
        },
        "train": {"proactive": {}, "reactive": {}},
        "inference": {"proactive": {}, "reactive": {}},
    }


def main():
    _prepare_dirs()
    random.seed(SPLIT_SEED)
    np.random.seed(SPLIT_SEED)
    start = time.time()

    servers_df = pd.read_csv(DEFAULT_SERVER_PATH)
    df = load_data(DEFAULT_TAXI_PATH, processed_csv=PROCESSED_TAXI_PATH)
    counts = df.groupby("taxi_id").size().sort_values(ascending=False)
    df = df[df["taxi_id"].isin(set(counts.head(ACTIVE_USERS).index))].reset_index(drop=True)
    train_df, test_df, train_ids, test_ids, split_meta = _split_by_balanced_exposure(df, servers_df)
    predictor = SimpleTrajectoryPredictor(forecast_horizon=FORECAST_HORIZON).fit(train_df)

    payload_path = os.path.join(OUT_DIR, "results.json")
    if os.path.exists(payload_path):
        with open(payload_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        print(f"[RESUME] Loaded existing payload from {payload_path}", flush=True)
    else:
        payload = _make_payload(df, train_df, test_df, train_ids, test_ids, split_meta)
        _write_payload(payload)

    phases = [
        ("train", "proactive", train_df, True, False),
        ("train", "reactive", train_df, False, False),
        ("inference", "proactive", test_df, True, True),
        ("inference", "reactive", test_df, False, True),
    ]

    for stage, mode, data, proactive, inference in phases:
        print(f"\n=== {stage.upper()} {mode.upper()} ===", flush=True)
        if not inference:
            if "SA" not in payload[stage][mode]:
                res = run_sa_microservice_fair(data, servers_df, predictor=predictor, proactive=proactive)
                payload[stage][mode]["SA"] = _summarize_result(res)
                _write_payload(payload)

            if "Nearest" not in payload[stage][mode]:
                res = run_nearest_microservice_fair(data, servers_df, predictor=predictor, proactive=proactive)
                payload[stage][mode]["Nearest"] = _summarize_result(res)
                _write_payload(payload)

            dqn_ckpt = DQN_CHECKPOINT_PROACTIVE if proactive else DQN_CHECKPOINT_REACTIVE
            if "DQN" not in payload[stage][mode]:
                res = run_dqn_microservice_fair(
                    data,
                    servers_df,
                    predictor=predictor,
                    proactive=proactive,
                    save_checkpoint_path=dqn_ckpt,
                )
                payload[stage][mode]["DQN"] = _summarize_result(res)
                _write_payload(payload)

            marl_ckpt = MARL_CHECKPOINT_PROACTIVE if proactive else MARL_CHECKPOINT_REACTIVE
            if "GAT-MARL" not in payload[stage][mode]:
                res = run_marl_gat_microservice(
                    data,
                    servers_df,
                    predictor=predictor,
                    proactive=proactive,
                    num_epochs=MARL_EPOCHS_PROACTIVE if proactive else MARL_EPOCHS_REACTIVE,
                    inference_mode=False,
                    save_checkpoint_path=marl_ckpt,
                )
                payload[stage][mode]["GAT-MARL"] = _summarize_result(res)
                _write_payload(payload)
        else:
            if "SA" not in payload[stage][mode]:
                res = run_sa_microservice_fair(
                    data,
                    servers_df,
                    predictor=predictor,
                    proactive=proactive,
                    collect_dag_proactive_stats=proactive,
                )
                payload[stage][mode]["SA"] = _summarize_result(res)
                _write_payload(payload)

            if "Nearest" not in payload[stage][mode]:
                res = run_nearest_microservice_fair(
                    data,
                    servers_df,
                    predictor=predictor,
                    proactive=proactive,
                    collect_dag_proactive_stats=proactive,
                )
                payload[stage][mode]["Nearest"] = _summarize_result(res)
                _write_payload(payload)

            dqn_ckpt = DQN_CHECKPOINT_PROACTIVE if proactive else DQN_CHECKPOINT_REACTIVE
            if "DQN" not in payload[stage][mode]:
                res = run_dqn_microservice_fair(
                    data,
                    servers_df,
                    predictor=predictor,
                    proactive=proactive,
                    inference_mode=True,
                    checkpoint_path=dqn_ckpt,
                )
                payload[stage][mode]["DQN"] = _summarize_result(res)
                _write_payload(payload)

            marl_ckpt = MARL_CHECKPOINT_PROACTIVE if proactive else MARL_CHECKPOINT_REACTIVE
            if "GAT-MARL" not in payload[stage][mode]:
                res = run_marl_gat_microservice(
                    data,
                    servers_df,
                    predictor=predictor,
                    proactive=proactive,
                    inference_mode=True,
                    checkpoint_path=marl_ckpt,
                    collect_dag_proactive_stats=proactive,
                )
                payload[stage][mode]["GAT-MARL"] = _summarize_result(res)
                _write_payload(payload)

        payload["elapsed_seconds_so_far"] = time.time() - start
        _write_payload(payload)

    payload["completed_at"] = datetime.now().isoformat()
    payload["elapsed_seconds"] = time.time() - start
    _write_payload(payload)
    print(f"\n[DONE] Full pipeline saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
