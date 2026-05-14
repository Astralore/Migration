import itertools
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd

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
MARL_EPOCHS = int(os.environ.get("MEDIUM_VALIDATION_MARL_EPOCHS", "4"))
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
    if proactive:
        return (
            f"| {name} | {res['total_migrations']} | {res['total_violations']} | "
            f"{pro} | {res.get('avg_decision_time_ms', 0):.2f} | {_avg_total_cost_ms(res):.2f} |\n"
        )
    return (
        f"| {name} | {res['total_migrations']} | {res['total_violations']} | "
        f"{res.get('avg_decision_time_ms', 0):.2f} | {_avg_total_cost_ms(res):.2f} |\n"
    )


def _write_report(payload):
    train_pro = payload["train"]["proactive"]
    train_rea = payload["train"]["reactive"]
    infer_pro = payload["inference"]["proactive"]
    infer_rea = payload["inference"]["reactive"]

    def table(pro, rea):
        s = "| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |\n"
        s += "|-----------|------------|------------|---------------------|------------------|---------------------|\n"
        for name in ["SA", "Nearest", "DQN", "GAT-MARL"]:
            if name in pro:
                s += _row(name, pro[name], proactive=True)
            else:
                s += f"| {name} | — | — | — | — | — |\n"
        s += "\n| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |\n"
        s += "|-----------|------------|------------|------------------|---------------------|\n"
        for name in ["SA", "Nearest", "DQN", "GAT-MARL"]:
            if name in rea:
                s += _row(name, rea[name], proactive=False)
            else:
                s += f"| {name} | — | — | — | — |\n"
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

## 本轮验证关注点

- 验证脚本显式使用 cov50 覆盖过滤数据，不再读取旧 cleaned CSV。
- train/test 按最近服务器距离风险暴露平衡切分，减少 proactive opportunity 偏斜。
- Avg Total Cost 使用 `total_cost_ms_sum / decision_count`，包含 SLA / future / tearing。

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
