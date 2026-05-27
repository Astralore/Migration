import os

# Set before tqdm is imported anywhere (via algorithms.*).
if os.environ.get("MEDIUM_VALIDATION_QUIET", "0") == "1":
    os.environ["TQDM_DISABLE"] = "1"

import itertools
import json
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
ACTIVE_USERS = int(os.environ.get("MEDIUM_VALIDATION_ACTIVE_USERS", "12"))
TEST_TAXI_RATIO = float(os.environ.get("MEDIUM_VALIDATION_TEST_TAXI_RATIO", "0.2"))
SPLIT_SEED = 42
MARL_EPOCHS = int(os.environ.get("MEDIUM_VALIDATION_MARL_EPOCHS", "8"))
FORECAST_HORIZON = 15
INFERENCE_ONLY = os.environ.get("MEDIUM_VALIDATION_INFERENCE_ONLY", "0") == "1"
INFERENCE_FORCE = os.environ.get("MEDIUM_VALIDATION_INFERENCE_FORCE", "0") == "1"
REACTIVE_ONLY = os.environ.get("MEDIUM_VALIDATION_REACTIVE_ONLY", "0") == "1"


def _progress(msg: str) -> None:
    print(msg, flush=True)


def _reward_scheme_label() -> str:
    try:
        from core.reward import reward_scheme, use_reward_v2_internal_path

        scheme = reward_scheme()
        if scheme == "v2" and use_reward_v2_internal_path():
            return "v2.1"
        return scheme
    except Exception:
        return "unknown"


def _algo_done_line(stage: str, mode: str, name: str, res: dict, elapsed_s: float) -> str:
    dc = int(res.get("decision_count") or 0)
    mig = int(res.get("total_migrations") or 0)
    avg_cost = _avg_total_cost_ms(res) if dc > 0 else 0.0
    return (
        f"<<< [{stage}/{mode}] {name} done in {elapsed_s / 60.0:.1f} min | "
        f"migrations={mig} decisions={dc} avg_total_cost_ms={avg_cost:.1f}"
    )


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
        "total_internal_path_ms",
        "total_migration_cost",
        "migration_decision_count",
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
    test_ratio = min(max(TEST_TAXI_RATIO, 0.05), 0.8)
    n_test = max(1, int(np.ceil(test_ratio * len(ids))))
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


def _migration_efficiency_metrics(res):
    """Derived migration comparison metrics from a single algorithm run."""
    mig_total = float(res.get("total_migration_cost") or 0.0)
    migrations = int(res.get("total_migrations") or 0)
    cost_sum = float(res.get("total_cost_ms_sum") or 0.0)
    mig_decisions = int(res.get("migration_decision_count") or 0)

    per_node = (mig_total / migrations) if migrations > 0 else None
    per_mig_decision = (mig_total / mig_decisions) if mig_decisions > 0 else None
    share = (mig_total / cost_sum) if cost_sum > 0 else None
    return {
        "cost_per_migrated_node_ms": per_node,
        "cost_per_migration_decision_ms": per_mig_decision,
        "migration_cost_share": share,
        "migration_decision_count": mig_decisions,
    }


def _fmt_migration_metric(value, *, percent=False):
    if value is None:
        return "—"
    if percent:
        return f"{100.0 * value:.2f}%"
    return f"{value:.2f}"


def _migration_efficiency_row(name, pro_res, rea_res):
    pro = _migration_efficiency_metrics(pro_res)
    rea = _migration_efficiency_metrics(rea_res)
    return (
        f"| {name} "
        f"| {_fmt_migration_metric(pro['cost_per_migrated_node_ms'])} "
        f"| {_fmt_migration_metric(pro['cost_per_migration_decision_ms'])} "
        f"| {_fmt_migration_metric(pro['migration_cost_share'], percent=True)} "
        f"| {_fmt_migration_metric(rea['cost_per_migrated_node_ms'])} "
        f"| {_fmt_migration_metric(rea['cost_per_migration_decision_ms'])} "
        f"| {_fmt_migration_metric(rea['migration_cost_share'], percent=True)} |\n"
    )


def _migration_efficiency_table(pro, rea):
    s = (
        "| Algorithm | Pro: Cost/Node (ms) | Pro: Cost/Mig Decision (ms) | Pro: Migration Share | "
        "Rea: Cost/Node (ms) | Rea: Cost/Mig Decision (ms) | Rea: Migration Share |\n"
    )
    s += (
        "|-----------|---------------------|-----------------------------|----------------------|"
        "---------------------|-----------------------------|----------------------|\n"
    )
    for name in ["SA", "Nearest", "DQN", "GAT-MARL"]:
        s += _migration_efficiency_row(name, pro.get(name, {}), rea.get(name, {}))
    return s


def _cost_row(name, res):
    dc = int(res.get("decision_count") or 0)
    avg_sla = (float(res.get("total_sla_penalty_ms") or 0.0) / dc) if dc > 0 else 0.0
    avg_migration = (float(res.get("total_migration_cost") or 0.0) / dc) if dc > 0 else 0.0
    return f"| {name} | {avg_sla:.2f} | {avg_migration:.2f} |\n"


def _write_cost_decomposition_chart(payload, stage, filename):
    if REACTIVE_ONLY:
        phases = [("reactive", payload[stage]["reactive"])]
    else:
        phases = [("proactive", payload[stage]["proactive"]), ("reactive", payload[stage]["reactive"])]
    algorithms = ["SA", "Nearest", "DQN", "GAT-MARL"]
    components = [
        ("SLA penalty", "total_sla_penalty_ms"),
        ("Migration", "total_migration_cost"),
    ]
    colors = ["#d62728", "#ff7f0e"]

    ncols = len(phases)
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5), sharey=True)
    if ncols == 1:
        axes = [axes]
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
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
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
    reactive_only = bool(payload.get("config", {}).get("reactive_only"))

    def reactive_table(results):
        s = (
            "| Algorithm | Migrations | SLA Risk | Severe | P95 Excess (km) | "
            "Avg SLA Penalty (ms) | Avg Total Cost (ms) | stay_ratio |\n"
        )
        s += (
            "|-----------|------------|----------|--------|-----------------|----------------------|"
            "---------------------|------------|\n"
        )
        for name in ["SA", "Nearest", "DQN", "GAT-MARL"]:
            res = results.get(name, {})
            if not res:
                s += f"| {name} | — | — | — | — | — | — | — |\n"
                continue
            stay = res.get("stay_action_ratio")
            stay_s = f"{float(stay):.4f}" if stay is not None else "—"
            dc = int(res.get("decision_count") or 0)
            avg_sla = (
                float(res.get("total_sla_penalty_ms") or 0.0) / dc if dc > 0 else 0.0
            )
            s += (
                f"| {name} | {res.get('total_migrations', '—')} | {res.get('total_violations', '—')} | "
                f"{res.get('severe_sla_violations', '—')} | {res.get('p95_sla_excess_distance_km', '—')} | "
                f"{avg_sla:.2f} | {res.get('avg_total_cost_ms', '—')} | {stay_s} |\n"
            )
        return s

    def dag_type_section(stage, mode, name):
        res = payload.get(stage, {}).get(mode, {}).get(name, {})
        mig = res.get("migrations_by_dag_type") or {}
        cost = res.get("cost_by_dag_type") or {}
        if not mig and not cost:
            return ""
        lines = [f"### {stage} {mode} — {name} by DAG type\n", "| DAG type | migrations | decisions | avg_total_cost_ms |\n", "|----------|------------|-----------|-------------------|\n"]
        keys = sorted(set(mig.keys()) | set(cost.keys()))
        for k in keys:
            c = cost.get(k, {})
            dc = int(c.get("decision_count") or 0)
            avg = (float(c.get("total_cost_ms_sum") or 0.0) / dc) if dc > 0 else 0.0
            lines.append(f"| {k} | {mig.get(k, 0)} | {dc} | {avg:.2f} |\n")
        return "".join(lines) + "\n"

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

    if reactive_only:
        body = f"""# 中等规模 cov50 数据验证报告（仅 Reactive）

**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**输出目录**：`{OUT_DIR}`  
**模式**：仅 Reactive（无轨迹预测 / 无 Proactive TTV）  
**GAT-MARL epochs**：{MARL_EPOCHS}（reactive）

## 训练段（Reactive）

{reactive_table(train_rea)}

## 推理段（Reactive）

{reactive_table(infer_rea)}

{dag_type_section("inference", "reactive", "GAT-MARL")}
{dag_type_section("inference", "reactive", "SA")}
"""
    else:
        body = f"""# 中等规模 cov50 数据验证报告

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
"""

    if reactive_only:
        report = body + f"""
## 成本分解堆叠图（Reactive）

![Inference Cost Decomposition]({cost_chart})

## 本轮验证关注点

- 仅 Reactive：无轨迹预测器、无 Proactive TTV。
- `stay_action_ratio` 接近 1.0 表示策略几乎不迁移；`candidate_action_counts` 中迁移动作计数应逐步上升。
- `migrations_by_dag_type` / `cost_by_dag_type` 用于观察反撕裂（Data_Heavy / FanOut 少迁、Simple 可拆）。
- SA 为 GAT-MARL 锚点（D0 后 L_internal 计入总成本，SA 应倾向少拆图）。

"""
    else:
        report = body + f"""

## 成本分解堆叠图

![Inference Cost Decomposition]({cost_chart})

## 推理阶段成本分解均值

{cost_table(infer_pro, infer_rea)}

## 推理阶段迁移效率指标

> **Cost/Node** = `total_migration_cost / total_migrations`（每次迁移节点的平均线性物理时延）  
> **Cost/Mig Decision** = `total_migration_cost / migration_decision_count`（仅 `migration_cost > 0` 的决策）  
> **Migration Share** = `total_migration_cost / total_cost_ms_sum`（迁移在总系统成本中的占比）

{_migration_efficiency_table(infer_pro, infer_rea)}

## 本轮验证关注点

- 验证脚本显式使用 cov50 覆盖过滤数据，不再读取旧 cleaned CSV。
- train/test 按最近服务器距离风险暴露平衡切分，减少 proactive opportunity 偏斜。
- Avg Decision Time 是算法计算耗时；Avg Access Latency 是真实接入延迟；Avg Total System Cost 使用 `total_cost_ms_sum / decision_count`，包含 SLA penalty 和迁移等系统代价。
- 迁移对比优先看「迁移效率指标」表（按节点 / 按有迁解决策 / 占比），而非 `total_migration_cost / decision_count`（会被大量无迁解决策稀释）。
- SLA Risk Count 是 max-entry 风险次数；Severe SLA Violations 和 SLA Excess 用于区分轻微风险与严重服务质量退化。

"""
    with open(os.path.join(OUT_DIR, "result.md"), "w", encoding="utf-8") as f:
        f.write(report)


def main():
    _prepare_dirs()
    random.seed(SPLIT_SEED)
    np.random.seed(SPLIT_SEED)

    start = time.time()
    _progress(
        f"[START] medium_validation stamp={STAMP} | out={OUT_DIR} | "
        f"marl_epochs={MARL_EPOCHS} | reward_scheme={_reward_scheme_label()} | "
        f"reactive_only={int(REACTIVE_ONLY)} | "
        f"quiet_tqdm={os.environ.get('MEDIUM_VALIDATION_QUIET', '0')}"
    )
    df_all = load_data(DEFAULT_TAXI_PATH, processed_csv=PROCESSED_TAXI_PATH)
    df = _filter_top_active(df_all, ACTIVE_USERS)
    servers_df = pd.read_csv(DEFAULT_SERVER_PATH)
    train_df, test_df, train_ids, test_ids, split_meta = _split_by_balanced_exposure(df, servers_df)

    if REACTIVE_ONLY:
        predictor = None
        _progress("[CONFIG] REACTIVE_ONLY=1: skip trajectory predictor; proactive phases disabled.")
    else:
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
                "test_taxi_ratio": TEST_TAXI_RATIO,
                "split_seed": SPLIT_SEED,
                "marl_epochs": MARL_EPOCHS,
                "forecast_horizon": FORECAST_HORIZON,
                "processed_taxi_path": PROCESSED_TAXI_PATH,
                "started_at": datetime.now().isoformat(),
                "reactive_only": REACTIVE_ONLY,
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
    if REACTIVE_ONLY:
        phases = [phase for phase in phases if phase[1] == "reactive"]
        print("[REACTIVE_ONLY] Running train/inference reactive only (no predictor).", flush=True)
    if INFERENCE_ONLY:
        phases = [phase for phase in phases if phase[0] == "inference"]
        print("[INFERENCE_ONLY] Skipping training phases.", flush=True)
    if INFERENCE_FORCE:
        payload["inference"]["proactive"] = {}
        payload["inference"]["reactive"] = {}
        print("[INFERENCE_FORCE] Cleared inference results; will re-run inference.", flush=True)

    for stage, mode, data, proactive, inference in phases:
        if (
            all(name in payload[stage][mode] for name in ["SA", "Nearest", "DQN", "GAT-MARL"])
            and not (inference and INFERENCE_FORCE)
        ):
            print(f"\n=== SKIP {stage.upper()} {mode.upper()} (already complete) ===", flush=True)
            continue

        if inference and INFERENCE_FORCE:
            print(f"\n=== FORCE {stage.upper()} {mode.upper()} (refresh inference metrics) ===", flush=True)
        else:
            print(f"\n=== {stage.upper()} {mode.upper()} ===", flush=True)
        if not inference:
            if "SA" not in payload[stage][mode]:
                _progress(f">>> [{stage}/{mode}] SA (train) starting...")
                t0 = time.time()
                sa_res = run_sa_microservice_fair(data, servers_df, predictor=predictor, proactive=proactive)
                payload[stage][mode]["SA"] = _summarize_result(sa_res)
                _progress(_algo_done_line(stage, mode, "SA", sa_res, time.time() - t0))

            if "Nearest" not in payload[stage][mode]:
                _progress(f">>> [{stage}/{mode}] Nearest (train) starting...")
                t0 = time.time()
                nearest_res = run_nearest_microservice_fair(
                    data, servers_df, predictor=predictor, proactive=proactive
                )
                payload[stage][mode]["Nearest"] = _summarize_result(nearest_res)
                _progress(_algo_done_line(stage, mode, "Nearest", nearest_res, time.time() - t0))

            dqn_ckpt = os.path.join(CHECKPOINT_DIR, f"dqn_{mode}.pth")
            if "DQN" not in payload[stage][mode]:
                _progress(f">>> [{stage}/{mode}] DQN (train) starting...")
                t0 = time.time()
                dqn_res = run_dqn_microservice_fair(
                    data,
                    servers_df,
                    predictor=predictor,
                    proactive=proactive,
                    save_checkpoint_path=dqn_ckpt,
                )
                payload[stage][mode]["DQN"] = _summarize_result(dqn_res)
                _progress(_algo_done_line(stage, mode, "DQN", dqn_res, time.time() - t0))

            marl_ckpt = os.path.join(CHECKPOINT_DIR, f"marl_gat_{mode}.pth")
            if "GAT-MARL" not in payload[stage][mode]:
                _progress(
                    f">>> [{stage}/{mode}] GAT-MARL (train, epochs={MARL_EPOCHS}) starting..."
                )
                t0 = time.time()
                marl_res = run_marl_gat_microservice(
                    data,
                    servers_df,
                    predictor=predictor,
                    proactive=proactive,
                    num_epochs=MARL_EPOCHS,
                    save_checkpoint_path=marl_ckpt,
                )
                payload[stage][mode]["GAT-MARL"] = _summarize_result(marl_res)
                _progress(_algo_done_line(stage, mode, "GAT-MARL", marl_res, time.time() - t0))
        else:
            if "SA" not in payload[stage][mode]:
                _progress(f">>> [{stage}/{mode}] SA (inference) starting...")
                t0 = time.time()
                sa_res = run_sa_microservice_fair(
                    data,
                    servers_df,
                    predictor=predictor,
                    proactive=proactive,
                    collect_dag_proactive_stats=proactive,
                )
                payload[stage][mode]["SA"] = _summarize_result(sa_res)
                _progress(_algo_done_line(stage, mode, "SA", sa_res, time.time() - t0))

            if "Nearest" not in payload[stage][mode]:
                _progress(f">>> [{stage}/{mode}] Nearest (inference) starting...")
                t0 = time.time()
                nearest_res = run_nearest_microservice_fair(
                    data,
                    servers_df,
                    predictor=predictor,
                    proactive=proactive,
                    collect_dag_proactive_stats=proactive,
                )
                payload[stage][mode]["Nearest"] = _summarize_result(nearest_res)
                _progress(_algo_done_line(stage, mode, "Nearest", nearest_res, time.time() - t0))

            if "DQN" not in payload[stage][mode]:
                _progress(f">>> [{stage}/{mode}] DQN (inference) starting...")
                t0 = time.time()
                dqn_res = run_dqn_microservice_fair(
                    data,
                    servers_df,
                    predictor=predictor,
                    proactive=proactive,
                    inference_mode=True,
                    checkpoint_path=os.path.join(CHECKPOINT_DIR, f"dqn_{mode}.pth"),
                )
                payload[stage][mode]["DQN"] = _summarize_result(dqn_res)
                _progress(_algo_done_line(stage, mode, "DQN", dqn_res, time.time() - t0))

            if "GAT-MARL" not in payload[stage][mode]:
                _progress(f">>> [{stage}/{mode}] GAT-MARL (inference) starting...")
                t0 = time.time()
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
                _progress(_algo_done_line(stage, mode, "GAT-MARL", marl_res, time.time() - t0))

        payload["elapsed_seconds_so_far"] = time.time() - start
        with open(payload_path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)
        _write_report(payload)
        _progress(
            f"[CHECKPOINT] {stage}/{mode} saved | elapsed={payload['elapsed_seconds_so_far'] / 60.0:.1f} min"
        )

    payload["completed_at"] = datetime.now().isoformat()
    payload["elapsed_seconds"] = time.time() - start
    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)
    _write_report(payload)
    print(f"\n[DONE] Medium validation saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
