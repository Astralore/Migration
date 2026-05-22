import itertools
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
from run_comparison import _avg_total_cost_ms


STAMP = os.environ.get(
    "MEDIUM_VALIDATION_STAMP",
    datetime.now().strftime("%Y%m%d_%H%M%S_cov50_stage8"),
)
OUT_DIR = os.path.join("experiments", f"medium_validation_{STAMP}")
CHECKPOINT_DIR = os.path.join(OUT_DIR, "checkpoints")
PROCESSED_TAXI_PATH = os.path.join(
    "data",
    "processed",
    "taxi_cleaned_active100_min100_eps2h_cov50.csv",
)
ACTIVE_USERS = 12
SPLIT_SEED = 42
MARL_EPOCHS = int(os.environ.get("MEDIUM_VALIDATION_MARL_EPOCHS", "8"))
FORECAST_HORIZON = 15


def _prepare_dirs():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


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


def _filter_top_active(df, n):
    counts = df.groupby("taxi_id").size().sort_values(ascending=False)
    keep = set(counts.head(n).index)
    return df[df["taxi_id"].isin(keep)].reset_index(drop=True)


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
                # Boundary exposure helps split future-risk samples instead of only current violations.
                "risk_score": hard_risk + 0.25 * boundary_risk,
                "nearest_over_15": int(hard_risk),
                "nearest_10_to_15": int(boundary_risk),
                "nearest_p50": float(np.percentile(nearest, 50)),
                "nearest_p95": float(np.percentile(nearest, 95)),
            }
        )
    return pd.DataFrame(rows)


def _split_by_balanced_exposure(df, servers_df):
    stats = _taxi_exposure_stats(df, servers_df)
    ids = list(stats["taxi_id"])
    n_test = max(1, len(ids) - max(1, int(np.floor(0.8 * len(ids)))))
    target = n_test / len(ids)
    total_rows = float(stats["rows"].sum())
    total_risk = float(stats["risk_score"].sum())
    rng = np.random.default_rng(SPLIT_SEED)

    best = None
    id_to_row = stats.set_index("taxi_id")
    for combo in itertools.combinations(ids, n_test):
        combo_set = set(combo)
        row_ratio = float(id_to_row.loc[list(combo_set), "rows"].sum()) / total_rows
        risk_ratio = (
            float(id_to_row.loc[list(combo_set), "risk_score"].sum()) / total_risk
            if total_risk > 0
            else row_ratio
        )
        score = abs(row_ratio - target) + abs(risk_ratio - target)
        # Stable but seed-dependent tie breaker.
        score += float(rng.random()) * 1e-9
        if best is None or score < best[0]:
            best = (score, combo_set, row_ratio, risk_ratio)

    test_ids = set(best[1])
    train_ids = set(ids) - test_ids
    train_df = df[df["taxi_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["taxi_id"].isin(test_ids)].reset_index(drop=True)
    split_meta = {
        "method": "balanced_by_nearest_server_exposure",
        "target_test_taxi_ratio": target,
        "test_row_ratio": best[2],
        "test_risk_ratio": best[3],
        "taxi_exposure": stats.to_dict("records"),
    }
    return train_df, test_df, train_ids, test_ids, split_meta


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


def _cost_row(name, res):
    dc = int(res.get("decision_count") or 0)
    avg_sla = (float(res.get("total_sla_penalty_ms") or 0.0) / dc) if dc > 0 else 0.0
    avg_migration = (float(res.get("total_migration_cost") or 0.0) / dc) if dc > 0 else 0.0
    return f"| {name} | {avg_sla:.2f} | {avg_migration:.2f} |\n"


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
        s = "| Algorithm | Migrations | SLA Risk Count | Severe SLA Violations | Avg SLA Excess (km) | P95 SLA Excess (km) | Avg SLA Penalty (ms) | Proactive Decisions | Avg Decision Time (ms) | Avg Access Latency (ms) | Avg Total System Cost (ms) |\n"
        s += "|-----------|------------|----------------|-----------------------|---------------------|---------------------|----------------------|---------------------|------------------------|-------------------------|----------------------------|\n"
        for name in ["SA", "Nearest", "DQN", "GAT-MARL"]:
            if name in pro:
                s += _row(name, pro[name], proactive=True)
            else:
                s += f"| {name} | — | — | — | — | — | — | — | — | — | — |\n"
        s += "\n| Algorithm | Migrations | SLA Risk Count | Severe SLA Violations | Avg SLA Excess (km) | P95 SLA Excess (km) | Avg SLA Penalty (ms) | Avg Decision Time (ms) | Avg Access Latency (ms) | Avg Total System Cost (ms) |\n"
        s += "|-----------|------------|----------------|-----------------------|---------------------|---------------------|----------------------|------------------------|-------------------------|----------------------------|\n"
        for name in ["SA", "Nearest", "DQN", "GAT-MARL"]:
            if name in rea:
                s += _row(name, rea[name], proactive=False)
            else:
                s += f"| {name} | — | — | — | — | — | — | — | — | — |\n"
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

    report = f"""# 中等规模 cov50 数据验证报告

**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**输出目录**：`{OUT_DIR}`  
**数据文件**：`{payload['config']['processed_taxi_path']}`  
**数据规模**：Top-{ACTIVE_USERS} active taxis；train={payload['data']['train_rows']} rows/{payload['data']['train_taxis']} taxis；test={payload['data']['test_rows']} rows/{payload['data']['test_taxis']} taxis  
**切分方式**：{payload['data']['split']['method']}；test row ratio={payload['data']['split']['test_row_ratio']:.3f}；test risk ratio={payload['data']['split']['test_risk_ratio']:.3f}  
**GAT-MARL epochs**：Proactive={MARL_EPOCHS}，Reactive={MARL_EPOCHS}

## 训练段

{table(train_pro, train_rea)}

## 推理段

{table(infer_pro, infer_rea)}

## 成本分解堆叠图

![Inference Cost Decomposition]({cost_chart})

## 推理阶段成本分解均值

{cost_table(infer_pro, infer_rea)}

## 本轮验证关注点

- 验证脚本显式使用 cov50 覆盖过滤数据，不再读取旧 cleaned CSV。
- train/test 按最近服务器距离风险暴露平衡切分，减少 proactive opportunity 偏斜。
- Avg Decision Time 是算法计算耗时；Avg Access Latency 是真实接入延迟；Avg Total System Cost 使用 `total_cost_ms_sum / decision_count`，包含 SLA penalty 和迁移等系统代价。
- SLA Risk Count 是 max-entry 风险次数；Severe SLA Violations 和 SLA Excess 用于区分轻微风险与严重服务质量退化。

"""
    with open(os.path.join(OUT_DIR, "result.md"), "w", encoding="utf-8") as f:
        f.write(report)


def main():
    _prepare_dirs()
    random.seed(SPLIT_SEED)
    np.random.seed(SPLIT_SEED)

    start = time.time()
    df_all = load_data(DEFAULT_TAXI_PATH, processed_csv=PROCESSED_TAXI_PATH)
    df = _filter_top_active(df_all, ACTIVE_USERS)
    servers_df = pd.read_csv(DEFAULT_SERVER_PATH)
    train_df, test_df, train_ids, test_ids, split_meta = _split_by_balanced_exposure(df, servers_df)

    predictor = SimpleTrajectoryPredictor(forecast_horizon=FORECAST_HORIZON).fit(train_df)

    payload_path = os.path.join(OUT_DIR, "results.json")
    if os.path.exists(payload_path):
        with open(payload_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        print(f"[RESUME] Loaded existing payload from {payload_path}", flush=True)
    else:
        payload = {
            "config": {
                "active_users": ACTIVE_USERS,
                "split_seed": SPLIT_SEED,
                "marl_epochs": MARL_EPOCHS,
                "forecast_horizon": FORECAST_HORIZON,
                "processed_taxi_path": PROCESSED_TAXI_PATH,
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

    phases = [
        ("train", "proactive", train_df, True, False),
        ("train", "reactive", train_df, False, False),
        ("inference", "proactive", test_df, True, True),
        ("inference", "reactive", test_df, False, True),
    ]

    for stage, mode, data, proactive, inference in phases:
        if all(name in payload[stage][mode] for name in ["SA", "Nearest", "DQN", "GAT-MARL"]):
            print(f"\n=== SKIP {stage.upper()} {mode.upper()} (already complete) ===", flush=True)
            continue

        print(f"\n=== {stage.upper()} {mode.upper()} ===", flush=True)
        if not inference:
            if "SA" not in payload[stage][mode]:
                sa_res = run_sa_microservice_fair(data, servers_df, predictor=predictor, proactive=proactive)
                payload[stage][mode]["SA"] = _summarize_result(sa_res)

            if "Nearest" not in payload[stage][mode]:
                nearest_res = run_nearest_microservice_fair(
                    data, servers_df, predictor=predictor, proactive=proactive
                )
                payload[stage][mode]["Nearest"] = _summarize_result(nearest_res)

            dqn_ckpt = os.path.join(CHECKPOINT_DIR, f"dqn_{mode}.pth")
            if "DQN" not in payload[stage][mode]:
                dqn_res = run_dqn_microservice_fair(
                    data,
                    servers_df,
                    predictor=predictor,
                    proactive=proactive,
                    save_checkpoint_path=dqn_ckpt,
                )
                payload[stage][mode]["DQN"] = _summarize_result(dqn_res)

            marl_ckpt = os.path.join(CHECKPOINT_DIR, f"marl_gat_{mode}.pth")
            if "GAT-MARL" not in payload[stage][mode]:
                marl_res = run_marl_gat_microservice(
                    data,
                    servers_df,
                    predictor=predictor,
                    proactive=proactive,
                    num_epochs=MARL_EPOCHS,
                    save_checkpoint_path=marl_ckpt,
                )
                payload[stage][mode]["GAT-MARL"] = _summarize_result(marl_res)
        else:
            if "SA" not in payload[stage][mode]:
                sa_res = run_sa_microservice_fair(
                    data,
                    servers_df,
                    predictor=predictor,
                    proactive=proactive,
                    collect_dag_proactive_stats=proactive,
                )
                payload[stage][mode]["SA"] = _summarize_result(sa_res)

            if "Nearest" not in payload[stage][mode]:
                nearest_res = run_nearest_microservice_fair(
                    data,
                    servers_df,
                    predictor=predictor,
                    proactive=proactive,
                    collect_dag_proactive_stats=proactive,
                )
                payload[stage][mode]["Nearest"] = _summarize_result(nearest_res)

            if "DQN" not in payload[stage][mode]:
                dqn_res = run_dqn_microservice_fair(
                    data,
                    servers_df,
                    predictor=predictor,
                    proactive=proactive,
                    inference_mode=True,
                    checkpoint_path=os.path.join(CHECKPOINT_DIR, f"dqn_{mode}.pth"),
                )
                payload[stage][mode]["DQN"] = _summarize_result(dqn_res)

            if "GAT-MARL" not in payload[stage][mode]:
                marl_res = run_marl_gat_microservice(
                    data,
                    servers_df,
                    predictor=predictor,
                    proactive=proactive,
                    inference_mode=True,
                    checkpoint_path=os.path.join(CHECKPOINT_DIR, f"marl_gat_{mode}.pth"),
                    collect_dag_proactive_stats=proactive,
                )
                payload[stage][mode]["GAT-MARL"] = _summarize_result(marl_res)

        payload["elapsed_seconds_so_far"] = time.time() - start
        with open(payload_path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)
        _write_report(payload)

    payload["completed_at"] = datetime.now().isoformat()
    payload["elapsed_seconds"] = time.time() - start
    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)
    _write_report(payload)
    print(f"\n[DONE] Medium validation saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
