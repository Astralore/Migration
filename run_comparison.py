#!/usr/bin/env python
"""
Entry point: run all three microservice migration algorithms and compare.
Supports both Proactive and Reactive modes for full experimental analysis.

  python run_comparison.py              # 单模式：由 INFERENCE_MODE 决定训练或推理
  python run_comparison.py --pipeline   # 全量：删 SAC 旧权重 → 训练 → 测试段推理 → 合并 result.md
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from core.data_loader import (
    load_data,
    DEFAULT_TAXI_PATH,
    DEFAULT_SERVER_PATH,
    DEFAULT_PROCESSED_TAXI_PATH,
)
from prediction.simple_predictor import SimpleTrajectoryPredictor
from algorithms.dqn import run_dqn_microservice_fair
from algorithms.sa import run_sa_microservice_fair
from algorithms.hybrid_sac import run_hybrid_sac_microservice
from evaluation.metrics import print_proactive_analysis, print_ranking_with_latency
from evaluation.plot import plot_training_curves, plot_cost_breakdown, plot_performance_metrics

# =============================================================================
# 工程化配置
# =============================================================================
INFERENCE_MODE = False  # False=训练模式（全量对比）, True=推理模式（加载 checkpoint 在测试段评测）

# Strategy B：活跃 Top‑N + 按车 80/20（command.md 阶段二）
SPLIT_SEED = 42
ACTIVE_USERS_LIMIT = 100
MIN_VEHICLE_POINTS = 100

# 若存在则直接读清洗结果 CSV，避免每次从原始 CSV 全量清洗（首次请运行 ``python -m core.data_loader --export``）

# 权重保存路径
CHECKPOINT_DIR = "checkpoints"
SAC_CHECKPOINT_PROACTIVE = "checkpoints/sac_proactive.pth"
SAC_CHECKPOINT_REACTIVE = "checkpoints/sac_reactive.pth"
DQN_CHECKPOINT_PROACTIVE = "checkpoints/dqn_proactive.pth"
DQN_CHECKPOINT_REACTIVE = "checkpoints/dqn_reactive.pth"

PROACTIVE = True
FORECAST_HORIZON = 15  # Extended horizon for better proactive detection


def _split_train_test_taxis(df_active):
    """80/20 disjoint split by taxi_id; reproducible with default_rng(SPLIT_SEED)."""
    unique_ids = df_active["taxi_id"].unique()
    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = rng.permutation(unique_ids)
    n = len(shuffled)
    n_train = int(np.floor(0.8 * n))
    train_ids = set(shuffled[:n_train])
    test_ids = set(shuffled[n_train:])
    train_df = df_active[df_active["taxi_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df_active[df_active["taxi_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df, train_ids, test_ids


def _avg_total_cost_ms(res):
    """单次触发决策的平均真实总代价（优先用 reward.details['total_cost_ms'] 聚合），单位 ms。"""
    dc = int(res.get("decision_count") or 0)
    if dc <= 0:
        return 0.0
    if "total_cost_ms_sum" in res:
        return float(res.get("total_cost_ms_sum") or 0.0) / dc
    lat = float(res.get("total_access_latency") or 0)
    comm = float(res.get("total_communication_cost") or 0)
    mig = float(res.get("total_migration_cost") or 0)
    total_ms = lat + comm + mig
    return total_ms / dc


def _data_protocol_line(train_df, test_df, phase_label):
    return (
        f"{phase_label}: load_data(active_users_limit={ACTIVE_USERS_LIMIT}, "
        f"min_vehicle_points={MIN_VEHICLE_POINTS}) + default_rng({SPLIT_SEED}) 80/20 by taxi_id; "
        f"train_taxis={train_df['taxi_id'].nunique()}, test_taxis={test_df['taxi_id'].nunique()}"
    )


def _dag_stats_markdown_table(stats):
    """与论文表头一致：DAG Name | Proactive Decisions | Total Migrated Nodes | Avg Nodes per Decision"""
    lines = [
        "| DAG Name | Proactive Decisions | Total Migrated Nodes | Avg Nodes per Decision |\n",
        "|----------|---------------------|----------------------|------------------------|\n",
    ]
    if not stats:
        lines.append("| *（无样本）* | — | — | — |\n")
        return "".join(lines)
    for dag_name in sorted(stats.keys()):
        b = stats[dag_name]
        pd_c = int(b["proactive_decisions"])
        mn = int(b["migrated_nodes"])
        avg = (mn / pd_c) if pd_c > 0 else 0.0
        lines.append(f"| {dag_name} | {pd_c} | {mn} | {avg:.2f} |\n")
    return "".join(lines)


def build_dag_adaptive_dual_appendix(proactive_results, section_heading="## 五、"):
    """
    SA 与 Hybrid SAC 的 Proactive 按 DAG 统计（与 hybrid_sac / sa 源码旁路条件一致）。
    section_heading: 单文件推理报告建议用 ``## 四、``；流水线合并报告用 ``## 五、``。
    """
    sa_stats = (proactive_results.get("SA") or {}).get("dag_proactive_migration_stats") or {}
    sac_stats = (proactive_results.get("Hybrid SAC") or {}).get("dag_proactive_migration_stats") or {}
    return (
        f"\n{section_heading}Proactive 按 DAG 自适应迁移统计（SA 与 Hybrid SAC 同口径）\n\n"
        "**统一条件**：已启用前瞻（`use_proactive`）；`get_trigger_type(...) == PROACTIVE`；"
        "单次决策内在 **同一拓扑序 `sorted_nodes`** 上比较 `previous_assignments` 与决策后节点放置，"
        "统计发生变更的节点数 `migrated_nodes_count`；按 **DAG Name**（`dag_type`）聚合 "
        "`proactive_decisions` 与 `migrated_nodes`；**Avg = migrated / proactive_decisions**（保留两位小数）。\n\n"
        "**开关对齐（仅推理实验）**：Hybrid SAC 仅在 `inference_mode=True` 时分配并写入；"
        "SA 仅在 `run_inference_phase` 的 Proactive 分支传入 `collect_dag_proactive_stats=True`。"
        "**训练阶段**两种算法均不采集本统计。\n\n"
        "### SA（Simulated Annealing）\n\n"
        + _dag_stats_markdown_table(sa_stats)
        + "\n### Hybrid SAC\n\n"
        + _dag_stats_markdown_table(sac_stats)
    )


def _remove_sac_checkpoints_for_fresh_train():
    """删除本次流水线使用的 SAC 权重，避免旧 checkpoint 影响训练后推理。"""
    removed = []
    for path in (SAC_CHECKPOINT_PROACTIVE, SAC_CHECKPOINT_REACTIVE):
        try:
            if os.path.isfile(path):
                os.remove(path)
                removed.append(path)
        except OSError as e:
            print(f"  [WARN] Could not remove {path}: {e}")
    if removed:
        print(f"  [PIPELINE] Removed old SAC checkpoints: {removed}")
    else:
        print("  [PIPELINE] No existing SAC checkpoints to remove (fresh train).")


def generate_experiment_report(
    proactive_results, reactive_results, is_inference_mode,
    train_df=None, test_df=None,
):
    """自动生成 result.md 实验报告（单模式）。"""
    mode_str = "推理模式 (Inference)" if is_inference_mode else "训练模式 (Training)"
    if train_df is not None and test_df is not None:
        data_range = _data_protocol_line(
            train_df, test_df, "推理 test_df" if is_inference_mode else "训练 train_df"
        )
    else:
        data_range = (
            f"load_data(active_users_limit={ACTIVE_USERS_LIMIT}, "
            f"min_vehicle_points={MIN_VEHICLE_POINTS}); default_rng({SPLIT_SEED}) 80/20 split"
        )

    report = f"""# 微服务迁移算法对比实验报告

**运行模式**：{mode_str}  
**数据范围**：{data_range}  
**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 一、Proactive 模式结果

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
"""

    for name, res in proactive_results.items():
        latency = res.get('avg_decision_time_ms', 0)
        avg_cost = _avg_total_cost_ms(res)
        report += (
            f"| {name} | {res['total_migrations']} | {res['total_violations']} | "
            f"{res.get('proactive_decisions', 0)} | {latency:.2f} | {avg_cost:.2f} |\n"
        )

    report += """
---

## 二、Reactive 模式结果

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
"""

    for name, res in reactive_results.items():
        latency = res.get('avg_decision_time_ms', 0)
        avg_cost = _avg_total_cost_ms(res)
        report += (
            f"| {name} | {res['total_migrations']} | {res['total_violations']} | "
            f"{latency:.2f} | {avg_cost:.2f} |\n"
        )

    report += """
---

## 三、时延对比分析

"""

    if 'Hybrid SAC' in proactive_results and 'SA' in proactive_results:
        sac_latency = proactive_results['Hybrid SAC'].get('avg_decision_time_ms', 0)
        sa_latency = proactive_results['SA'].get('avg_decision_time_ms', 0)

        if sa_latency > 0:
            speedup = sa_latency / sac_latency if sac_latency > 0 else float('inf')
            report += f"- **Hybrid SAC 平均决策时延**: {sac_latency:.2f} ms\n"
            report += f"- **SA 平均决策时延**: {sa_latency:.2f} ms\n"
            report += f"- **加速比**: SAC 比 SA 快 **{speedup:.1f}x**\n"

    if is_inference_mode:
        report += build_dag_adaptive_dual_appendix(proactive_results, section_heading="## 四、")
    report += "\n---\n\n*报告自动生成*\n"

    with open("result.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n  [REPORT] Experiment report saved to result.md")


def _metrics_row(name, res, proactive_table=False):
    latency = res.get('avg_decision_time_ms', 0)
    avg_cost = _avg_total_cost_ms(res)
    pro_d = res.get('proactive_decisions', 0)
    if proactive_table:
        return (
            f"| {name} | {res['total_migrations']} | {res['total_violations']} | {pro_d} | "
            f"{latency:.2f} | {avg_cost:.2f} |\n"
        )
    return (
        f"| {name} | {res['total_migrations']} | {res['total_violations']} | "
        f"{latency:.2f} | {avg_cost:.2f} |\n"
    )


def generate_full_pipeline_report(
    train_proactive,
    train_reactive,
    infer_proactive,
    infer_reactive,
    wall_train_s,
    wall_infer_s,
    train_df,
    test_df,
):
    """
    训练段 + 测试段推理合并报告；记录与近期工程改动相关的指标说明。
    """
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def table_pro(pro, rea, title):
        s = f"### {title}\n\n"
        s += "| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |\n"
        s += "|-----------|------------|------------|---------------------|------------------|---------------------|\n"
        for name in ["SA", "DQN", "Hybrid SAC"]:
            if name in pro:
                s += _metrics_row(name, pro[name], proactive_table=True)
        s += "\n| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |\n"
        s += "|-----------|------------|------------|------------------|---------------------|\n"
        for name in ["SA", "DQN", "Hybrid SAC"]:
            if name in rea:
                s += _metrics_row(name, rea[name], proactive_table=False)
        return s

    sac_tr_p = train_proactive.get("Hybrid SAC", {})
    sac_tr_r = train_reactive.get("Hybrid SAC", {})
    sac_if_p = infer_proactive.get("Hybrid SAC", {})
    sac_if_r = infer_reactive.get("Hybrid SAC", {})

    def delta_line(label, tr, inf, key):
        a = tr.get(key, 0)
        b = inf.get(key, 0)
        return f"- **{label}**（训练段 → 测试段）: {a} → {b}\n"

    proto = _data_protocol_line(train_df, test_df, "数据协议")

    report = f"""# 微服务迁移算法对比实验报告（全量流水线）

**生成时间**：{ts}  
**流程**：启动前已删除 `sac_proactive.pth` / `sac_reactive.pth`（若存在）→ **训练**（仅 `train_df`）→ 保存新权重 → **推理**（仅 `test_df`，同一划分）加载新权重评测 Hybrid SAC。

**数据协议（Strategy B）**：{proto}

**工程上下文（与指标相关）**：

- **物理与奖励**：`total_cost_ms`（接入/迁移/tearing/通信/future/SLA）与 `context` 触发解耦（Reactive 空间 + QoS）。
- **Hybrid SAC**：离散 Actor **Logits 动作掩码**（合法 NEAREST、全非法回退 FOLLOW_SA），训练 `num_epochs=6`。
- **性能路径**：`core/geo.py`、`context.py`、`reward.py`、`state_builder.py` **NumPy 向量化**（Haversine 批量、近邻 `argpartition` 等），降低仿真墙钟时间但不改变公式。

**墙钟时间**：训练阶段约 **{wall_train_s:.0f} s**，推理阶段约 **{wall_infer_s:.0f} s**。

---

## 一、训练段结果（train_df）

{table_pro(train_proactive, train_reactive, "Proactive（上表）/ Reactive（下表）")}

---

## 二、测试段推理结果（test_df）

*Hybrid SAC 与 DQN 在测试段加载训练段保存的 checkpoint；SA 无磁盘权重，在测试段按既有脚本逻辑运行。*

{table_pro(infer_proactive, infer_reactive, "Proactive（上表）/ Reactive（下表）")}

---

## 三、Hybrid SAC 泛化对比（训练 → 测试）

{delta_line("Proactive Violations", sac_tr_p, sac_if_p, "total_violations")}
{delta_line("Proactive Migrations", sac_tr_p, sac_if_p, "total_migrations")}
{delta_line("Reactive Violations", sac_tr_r, sac_if_r, "total_violations")}
{delta_line("Reactive Migrations", sac_tr_r, sac_if_r, "total_migrations")}

**测试段决策时延（Hybrid SAC）**：

- Proactive：**{sac_if_p.get("avg_decision_time_ms", 0):.2f} ms**（训练段末次 eval 统计：**{sac_tr_p.get("avg_decision_time_ms", 0):.2f} ms**）
- Reactive：**{sac_if_r.get("avg_decision_time_ms", 0):.2f} ms**（训练段：**{sac_tr_r.get("avg_decision_time_ms", 0):.2f} ms**）

---

## 四、时延对比（测试段 Proactive：SAC vs SA）

"""

    if 'Hybrid SAC' in infer_proactive and 'SA' in infer_proactive:
        sac_l = infer_proactive['Hybrid SAC'].get('avg_decision_time_ms', 0)
        sa_l = infer_proactive['SA'].get('avg_decision_time_ms', 0)
        if sa_l > 0 and sac_l > 0:
            report += f"- Hybrid SAC: **{sac_l:.2f} ms**；SA: **{sa_l:.2f} ms**；比值 SA/SAC ≈ **{sa_l/sac_l:.1f}x**\n"
        else:
            report += "- （时延数据不足，略）\n"
    else:
        report += "- （略）\n"

    report += build_dag_adaptive_dual_appendix(infer_proactive, section_heading="## 五、")
    report += "\n---\n\n*报告由 `run_comparison.py --pipeline` 自动生成*\n"

    with open("result.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  [REPORT] Full pipeline report saved to result.md")


def run_training_phase(servers_df):
    """训练段：仅 train_df；返回结果、predictor、train_df、test_df（供推理复用划分）。"""
    df_active = load_data(
        DEFAULT_TAXI_PATH,
        sample_fraction=1.0,
        active_users_limit=ACTIVE_USERS_LIMIT,
        min_vehicle_points=MIN_VEHICLE_POINTS,
        processed_csv=DEFAULT_PROCESSED_TAXI_PATH,
    )
    train_df, test_df, _, _ = _split_train_test_taxis(df_active)
    print("\n[MODE] Training — Strategy B (train_df only)")
    print(
        f"  { _data_protocol_line(train_df, test_df, 'split') }"
    )

    predictor = SimpleTrajectoryPredictor(forecast_horizon=FORECAST_HORIZON)
    predictor.fit(train_df)
    print(f"  Predictor fitted: {len(predictor.velocity_factors)} taxis with velocity data")

    proactive_results = {}

    print("\n" + "#" * 80)
    print("  PHASE 1: Proactive Mode (Training)")
    print("#" * 80)
    t0 = time.time()
    proactive_results["SA"] = run_sa_microservice_fair(
        train_df, servers_df, predictor=predictor, proactive=True
    )
    print(f"  SA done in {time.time() - t0:.1f}s")

    t0 = time.time()
    proactive_results["DQN"] = run_dqn_microservice_fair(
        train_df, servers_df, predictor=predictor, proactive=True,
        save_checkpoint_path=DQN_CHECKPOINT_PROACTIVE,
    )
    print(f"  DQN done in {time.time() - t0:.1f}s")

    t0 = time.time()
    proactive_results["Hybrid SAC"] = run_hybrid_sac_microservice(
        train_df, servers_df, predictor=predictor, proactive=True, num_epochs=6,
        inference_mode=False,
        save_checkpoint_path=SAC_CHECKPOINT_PROACTIVE,
    )
    print(f"  Hybrid SAC done in {time.time() - t0:.1f}s")

    reactive_results = {}
    print("\n" + "#" * 80)
    print("  PHASE 2: Reactive Mode (Training)")
    print("#" * 80)

    t0 = time.time()
    reactive_results["SA"] = run_sa_microservice_fair(
        train_df, servers_df, predictor=predictor, proactive=False
    )
    print(f"  SA done in {time.time() - t0:.1f}s")

    t0 = time.time()
    reactive_results["DQN"] = run_dqn_microservice_fair(
        train_df, servers_df, predictor=predictor, proactive=False,
        save_checkpoint_path=DQN_CHECKPOINT_REACTIVE,
    )
    print(f"  DQN done in {time.time() - t0:.1f}s")

    t0 = time.time()
    reactive_results["Hybrid SAC"] = run_hybrid_sac_microservice(
        train_df, servers_df, predictor=predictor, proactive=False, num_epochs=2,
        inference_mode=False,
        save_checkpoint_path=SAC_CHECKPOINT_REACTIVE,
    )
    print(f"  Hybrid SAC done in {time.time() - t0:.1f}s")

    return proactive_results, reactive_results, predictor, train_df, test_df


def run_inference_phase(servers_df, train_df=None, test_df=None):
    """测试段推理：predictor 仅在 train_df 上 fit；算法仿真仅使用 test_df。"""
    if train_df is None or test_df is None:
        df_active = load_data(
            DEFAULT_TAXI_PATH,
            sample_fraction=1.0,
            active_users_limit=ACTIVE_USERS_LIMIT,
            min_vehicle_points=MIN_VEHICLE_POINTS,
            processed_csv=DEFAULT_PROCESSED_TAXI_PATH,
        )
        train_df, test_df, _, _ = _split_train_test_taxis(df_active)
        print("\n[MODE] Inference — loaded df_active + split (standalone inference)")
    else:
        print("\n[MODE] Inference — reusing train_df/test_df from training phase (single load_data in train)")

    print(f"  { _data_protocol_line(train_df, test_df, 'split') }")

    predictor = SimpleTrajectoryPredictor(forecast_horizon=FORECAST_HORIZON)
    predictor.fit(train_df)
    print("  Predictor fitted on train_df only")

    df = test_df

    proactive_results = {}
    print("\n" + "#" * 80)
    print("  PHASE 1: Proactive Mode (Inference)")
    print("#" * 80)

    t0 = time.time()
    proactive_results["SA"] = run_sa_microservice_fair(
        df, servers_df, predictor=predictor, proactive=True,
        collect_dag_proactive_stats=True,
    )
    print(f"  SA done in {time.time() - t0:.1f}s")

    t0 = time.time()
    proactive_results["DQN"] = run_dqn_microservice_fair(
        df, servers_df, predictor=predictor, proactive=True,
        inference_mode=True,
        checkpoint_path=DQN_CHECKPOINT_PROACTIVE,
    )
    print(f"  DQN done in {time.time() - t0:.1f}s")

    t0 = time.time()
    proactive_results["Hybrid SAC"] = run_hybrid_sac_microservice(
        df, servers_df, predictor=predictor, proactive=True,
        inference_mode=True,
        checkpoint_path=SAC_CHECKPOINT_PROACTIVE,
    )
    print(f"  Hybrid SAC done in {time.time() - t0:.1f}s")

    reactive_results = {}
    print("\n" + "#" * 80)
    print("  PHASE 2: Reactive Mode (Inference)")
    print("#" * 80)

    t0 = time.time()
    reactive_results["SA"] = run_sa_microservice_fair(
        df, servers_df, predictor=predictor, proactive=False
    )
    print(f"  SA done in {time.time() - t0:.1f}s")

    t0 = time.time()
    reactive_results["DQN"] = run_dqn_microservice_fair(
        df, servers_df, predictor=predictor, proactive=False,
        inference_mode=True,
        checkpoint_path=DQN_CHECKPOINT_REACTIVE,
    )
    print(f"  DQN done in {time.time() - t0:.1f}s")

    t0 = time.time()
    reactive_results["Hybrid SAC"] = run_hybrid_sac_microservice(
        df, servers_df, predictor=predictor, proactive=False,
        inference_mode=True,
        checkpoint_path=SAC_CHECKPOINT_REACTIVE,
    )
    print(f"  Hybrid SAC done in {time.time() - t0:.1f}s")

    return proactive_results, reactive_results, train_df, test_df


def run_all_algorithms(df, servers_df, predictor, proactive, label=""):
    """Run SA, DQN, Hybrid and return results dict."""
    results = {}
    mode_str = "Proactive" if proactive else "Reactive"

    # SA
    print(f"\n{'=' * 60}")
    print(f"  [{label}] Running SA ({mode_str}) ...")
    print(f"{'=' * 60}")
    t0 = time.time()
    results["SA"] = run_sa_microservice_fair(
        df, servers_df, predictor=predictor, proactive=proactive,
    )
    print(f"  SA done in {time.time() - t0:.1f}s")

    # DQN
    print(f"\n{'=' * 60}")
    print(f"  [{label}] Running DQN ({mode_str}) ...")
    print(f"{'=' * 60}")
    t0 = time.time()
    results["DQN"] = run_dqn_microservice_fair(
        df, servers_df, predictor=predictor, proactive=proactive,
    )
    print(f"  DQN done in {time.time() - t0:.1f}s")

    # Hybrid SAC (新的 Trigger-Conditioned GAT + Discrete SAC)
    # v3.6: Train/Eval split — last epoch deterministic eval (fair vs SA)
    print(f"\n{'=' * 60}")
    print(f"  [{label}] Running Hybrid SAC v3.6 ({mode_str}) ...")
    print(f"  [v3.6: 5 train + 1 eval, argmax eval, no replay/optimize on eval]")
    print(f"{'=' * 60}")
    t0 = time.time()
    results["Hybrid SAC"] = run_hybrid_sac_microservice(
        df, servers_df, predictor=predictor, proactive=proactive, num_epochs=6,
    )
    print(f"  Hybrid SAC done in {time.time() - t0:.1f}s")

    return results


def _print_results_and_plots(proactive_results, reactive_results):
    print("\n" + "#" * 80)
    print("  Proactive Mode Results (with Latency)")
    print("#" * 80)
    print_ranking_with_latency(proactive_results)

    print("\n" + "#" * 80)
    print("  Reactive Mode Results (with Latency)")
    print("#" * 80)
    print_ranking_with_latency(reactive_results)

    print_proactive_analysis(proactive_results, reactive_results)

    print("\n" + "=" * 80)
    print("  PAPER SUMMARY")
    print("=" * 80)
    for name in ["SA", "DQN", "Hybrid SAC"]:
        pro = proactive_results[name]
        rea = reactive_results[name]
        pro_v, rea_v = pro['total_violations'], rea['total_violations']
        pro_m, rea_m = pro['total_migrations'], rea['total_migrations']
        pro_d = pro.get('proactive_decisions', 0)
        pro_latency = pro.get('avg_decision_time_ms', 0)

        if rea_v > 0:
            v_reduction = (rea_v - pro_v) / rea_v * 100
        else:
            v_reduction = 0

        print(f"\n  {name}:")
        print(f"    - Proactive decisions: {pro_d}")
        print(f"    - Real Violations: {rea_v} -> {pro_v} ({v_reduction:+.1f}%)")
        print(f"    - Migrations: {rea_m} -> {pro_m}")
        print(f"    - Avg Decision Latency: {pro_latency:.2f} ms")
    print("\n" + "=" * 80)

    os.makedirs("outputs", exist_ok=True)
    print("\n" + "#" * 80)
    print("  GENERATING VISUALIZATIONS")
    print("#" * 80)

    for name, key in [("DQN", "DQN"), ("Hybrid_SAC", "Hybrid SAC")]:
        for mode, results in [("proactive", proactive_results), ("reactive", reactive_results)]:
            res = results[key]
            if res.get('loss_history'):
                plot_training_curves(
                    res,
                    save_path=f"outputs/{name.lower()}_{mode}_training.png",
                    title_prefix=f"{name} ({mode.capitalize()})",
                )

    plot_cost_breakdown(proactive_results, save_path="outputs/cost_breakdown.png")
    plot_performance_metrics(proactive_results, reactive_results, save_path="outputs/violation_comparison.png")

    print("\n  All visualizations generated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Microservice migration algorithm comparison")
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="删除 SAC checkpoint → 全量训练 → 测试段推理，并写入合并版 result.md",
    )
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("=" * 80)
    print("  Microservice Migration — Full Comparison (Asymmetric Cost Model)")
    if args.pipeline:
        print("  Mode: FULL PIPELINE (train + inference, fresh SAC weights)")
    else:
        print(f"  Mode: {'INFERENCE' if INFERENCE_MODE else 'TRAINING'}")
    print("=" * 80)

    servers_df = pd.read_csv(DEFAULT_SERVER_PATH)
    print(f"  Edge servers loaded: {len(servers_df)}")

    if args.pipeline:
        print("\n" + "#" * 80)
        print("  FULL PIPELINE: remove SAC weights → train → infer → result.md")
        print("#" * 80)
        _remove_sac_checkpoints_for_fresh_train()

        t_train = time.time()
        train_pro, train_rea, _, train_df, test_df = run_training_phase(servers_df)
        wall_train = time.time() - t_train

        _print_results_and_plots(train_pro, train_rea)

        t_inf = time.time()
        infer_pro, infer_rea, _, _ = run_inference_phase(servers_df, train_df, test_df)
        wall_infer = time.time() - t_inf

        print("\n" + "#" * 80)
        print("  INFERENCE PHASE Results (with Latency)")
        print("#" * 80)
        print_ranking_with_latency(infer_pro)
        print_ranking_with_latency(infer_rea)

        generate_full_pipeline_report(
            train_pro, train_rea, infer_pro, infer_rea,
            wall_train_s=wall_train,
            wall_infer_s=wall_infer,
            train_df=train_df,
            test_df=test_df,
        )

        # 曲线以训练段为准（含完整 loss_history）
        os.makedirs("outputs", exist_ok=True)
        for name, key in [("DQN", "DQN"), ("Hybrid_SAC", "Hybrid SAC")]:
            for mode, results in [("proactive", train_pro), ("reactive", train_rea)]:
                res = results[key]
                if res.get('loss_history'):
                    plot_training_curves(
                        res,
                        save_path=f"outputs/{name.lower()}_{mode}_training.png",
                        title_prefix=f"{name} ({mode.capitalize()})",
                    )
        plot_cost_breakdown(train_pro, save_path="outputs/cost_breakdown.png")
        plot_performance_metrics(train_pro, train_rea, save_path="outputs/violation_comparison.png")
        print("\n  [PIPELINE] Visualizations updated from training phase.")
        return

    if not INFERENCE_MODE:
        train_pro, train_rea, _, train_df, test_df = run_training_phase(servers_df)
        proactive_results, reactive_results = train_pro, train_rea
    else:
        proactive_results, reactive_results, train_df, test_df = run_inference_phase(servers_df)

    _print_results_and_plots(proactive_results, reactive_results)
    generate_experiment_report(
        proactive_results, reactive_results, INFERENCE_MODE,
        train_df=train_df, test_df=test_df,
    )


if __name__ == "__main__":
    main()
