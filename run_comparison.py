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
import pandas as pd

from core.data_loader import load_data, DEFAULT_TAXI_PATH, DEFAULT_SERVER_PATH
from prediction.simple_predictor import SimpleTrajectoryPredictor
from algorithms.dqn import run_dqn_microservice_fair
from algorithms.sa import run_sa_microservice_fair
from algorithms.hybrid_sa_dqn import run_hybrid_microservice_fair
from algorithms.hybrid_sac import run_hybrid_sac_microservice
from evaluation.metrics import print_ranking, print_proactive_analysis, print_ranking_with_latency
from evaluation.plot import plot_training_curves, plot_cost_breakdown, plot_performance_metrics

# =============================================================================
# 工程化配置
# =============================================================================
INFERENCE_MODE = False  # False=训练模式（全量对比）, True=推理模式（加载 checkpoint 在测试段评测）

# 数据切分配置
TRAIN_START_INDEX = 0
TRAIN_END_INDEX = 10000      # 训练数据: [0, 10000)
TEST_START_INDEX = 10000
TEST_END_INDEX = 15000       # 测试数据: [10000, 15000)

# 权重保存路径
CHECKPOINT_DIR = "checkpoints"
SAC_CHECKPOINT_PROACTIVE = "checkpoints/sac_proactive.pth"
SAC_CHECKPOINT_REACTIVE = "checkpoints/sac_reactive.pth"

# 原有配置
CHUNK_SIZE = 10000
PROACTIVE = True
FORECAST_HORIZON = 15  # Extended horizon for better proactive detection


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


def generate_experiment_report(proactive_results, reactive_results, is_inference_mode):
    """自动生成 result.md 实验报告（单模式）。"""
    mode_str = "推理模式 (Inference)" if is_inference_mode else "训练模式 (Training)"
    data_range = f"[{TEST_START_INDEX}:{TEST_END_INDEX}]" if is_inference_mode else f"[{TRAIN_START_INDEX}:{TRAIN_END_INDEX}]"

    report = f"""# 微服务迁移算法对比实验报告

**运行模式**：{mode_str}  
**数据范围**：{data_range}  
**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 一、Proactive 模式结果

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Score |
|-----------|------------|------------|---------------------|------------------|-------|
"""

    for name, res in proactive_results.items():
        latency = res.get('avg_decision_time_ms', 0)
        score = res['total_migrations'] + 0.5 * res['total_violations']
        report += f"| {name} | {res['total_migrations']} | {res['total_violations']} | {res.get('proactive_decisions', 0)} | {latency:.2f} | {score:.1f} |\n"

    report += """
---

## 二、Reactive 模式结果

| Algorithm | Migrations | Violations | Avg Latency (ms) | Score |
|-----------|------------|------------|------------------|-------|
"""

    for name, res in reactive_results.items():
        latency = res.get('avg_decision_time_ms', 0)
        score = res['total_migrations'] + 0.5 * res['total_violations']
        report += f"| {name} | {res['total_migrations']} | {res['total_violations']} | {latency:.2f} | {score:.1f} |\n"

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

    report += "\n---\n\n*报告自动生成*\n"

    with open("result.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n  [REPORT] Experiment report saved to result.md")


def _metrics_row(name, res, proactive_table=False):
    latency = res.get('avg_decision_time_ms', 0)
    score = res['total_migrations'] + 0.5 * res['total_violations']
    pro_d = res.get('proactive_decisions', 0)
    if proactive_table:
        return f"| {name} | {res['total_migrations']} | {res['total_violations']} | {pro_d} | {latency:.2f} | {score:.1f} |\n"
    return f"| {name} | {res['total_migrations']} | {res['total_violations']} | {latency:.2f} | {score:.1f} |\n"


def generate_full_pipeline_report(
    train_proactive,
    train_reactive,
    infer_proactive,
    infer_reactive,
    wall_train_s,
    wall_infer_s,
):
    """
    训练段 + 测试段推理合并报告；记录与近期工程改动相关的指标说明。
    """
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def table_pro(pro, rea, title):
        s = f"### {title}\n\n"
        s += "| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Score |\n"
        s += "|-----------|------------|------------|---------------------|------------------|-------|\n"
        for name in ["SA", "DQN", "Hybrid SAC"]:
            if name in pro:
                s += _metrics_row(name, pro[name], proactive_table=True)
        s += "\n| Algorithm | Migrations | Violations | Avg Latency (ms) | Score |\n"
        s += "|-----------|------------|------------|------------------|-------|\n"
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

    report = f"""# 微服务迁移算法对比实验报告（全量流水线）

**生成时间**：{ts}  
**流程**：启动前已删除 `sac_proactive.pth` / `sac_reactive.pth`（若存在）→ **训练** `[{TRAIN_START_INDEX}:{TRAIN_END_INDEX})` → 保存新权重 → **推理** `[{TEST_START_INDEX}:{TEST_END_INDEX})` 加载新权重评测 Hybrid SAC。

**工程上下文（与指标相关）**：

- **物理与奖励**：`total_cost_ms`（接入/迁移/tearing/通信/future/SLA）与 `context` 触发解耦（Reactive 空间 + QoS）。
- **Hybrid SAC**：离散 Actor **Logits 动作掩码**（合法 NEAREST、全非法回退 FOLLOW_SA），训练 `num_epochs=6`。
- **性能路径**：`core/geo.py`、`context.py`、`reward.py`、`state_builder.py` **NumPy 向量化**（Haversine 批量、近邻 `argpartition` 等），降低仿真墙钟时间但不改变公式。

**墙钟时间**：训练阶段约 **{wall_train_s:.0f} s**，推理阶段约 **{wall_infer_s:.0f} s**。

---

## 一、训练段结果 `[{TRAIN_START_INDEX}:{TRAIN_END_INDEX})`

{table_pro(train_proactive, train_reactive, "Proactive（上表）/ Reactive（下表）")}

---

## 二、测试段推理结果 `[{TEST_START_INDEX}:{TEST_END_INDEX})`

*Hybrid SAC 使用训练段刚写入的 checkpoint；DQN/SA 无磁盘权重，在测试段上按既有脚本逻辑运行。*

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

    report += "\n---\n\n*报告由 `run_comparison.py --pipeline` 自动生成*\n"

    with open("result.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  [REPORT] Full pipeline report saved to result.md")


def run_training_phase(servers_df):
    """训练段：返回 proactive_results, reactive_results。"""
    print(f"\n[MODE] Training Mode - Data: [{TRAIN_START_INDEX}:{TRAIN_END_INDEX}]")
    df = load_data(DEFAULT_TAXI_PATH, start_index=TRAIN_START_INDEX, end_index=TRAIN_END_INDEX)

    predictor = SimpleTrajectoryPredictor(forecast_horizon=FORECAST_HORIZON)
    predictor.fit(df)
    print(f"  Predictor fitted: {len(predictor.velocity_factors)} taxis with velocity data")

    proactive_results = {}

    print("\n" + "#" * 80)
    print("  PHASE 1: Proactive Mode (Training)")
    print("#" * 80)
    t0 = time.time()
    proactive_results["SA"] = run_sa_microservice_fair(
        df, servers_df, predictor=predictor, proactive=True
    )
    print(f"  SA done in {time.time() - t0:.1f}s")

    t0 = time.time()
    proactive_results["DQN"] = run_dqn_microservice_fair(
        df, servers_df, predictor=predictor, proactive=True
    )
    print(f"  DQN done in {time.time() - t0:.1f}s")

    t0 = time.time()
    proactive_results["Hybrid SAC"] = run_hybrid_sac_microservice(
        df, servers_df, predictor=predictor, proactive=True, num_epochs=6,
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
        df, servers_df, predictor=predictor, proactive=False
    )
    print(f"  SA done in {time.time() - t0:.1f}s")

    t0 = time.time()
    reactive_results["DQN"] = run_dqn_microservice_fair(
        df, servers_df, predictor=predictor, proactive=False
    )
    print(f"  DQN done in {time.time() - t0:.1f}s")

    t0 = time.time()
    reactive_results["Hybrid SAC"] = run_hybrid_sac_microservice(
        df, servers_df, predictor=predictor, proactive=False, num_epochs=6,
        inference_mode=False,
        save_checkpoint_path=SAC_CHECKPOINT_REACTIVE,
    )
    print(f"  Hybrid SAC done in {time.time() - t0:.1f}s")

    return proactive_results, reactive_results, predictor


def run_inference_phase(servers_df):
    """测试段推理：预测器在训练段拟合，评测在测试段。"""
    print(f"\n[MODE] Inference Mode - Data: [{TEST_START_INDEX}:{TEST_END_INDEX}]")

    train_df = load_data(DEFAULT_TAXI_PATH, start_index=TRAIN_START_INDEX, end_index=TRAIN_END_INDEX)
    predictor = SimpleTrajectoryPredictor(forecast_horizon=FORECAST_HORIZON)
    predictor.fit(train_df)
    print(f"  Predictor fitted on TRAINING data [{TRAIN_START_INDEX}:{TRAIN_END_INDEX}]")

    df = load_data(DEFAULT_TAXI_PATH, start_index=TEST_START_INDEX, end_index=TEST_END_INDEX)

    proactive_results = {}
    print("\n" + "#" * 80)
    print("  PHASE 1: Proactive Mode (Inference)")
    print("#" * 80)

    t0 = time.time()
    proactive_results["SA"] = run_sa_microservice_fair(
        df, servers_df, predictor=predictor, proactive=True
    )
    print(f"  SA done in {time.time() - t0:.1f}s")

    t0 = time.time()
    proactive_results["DQN"] = run_dqn_microservice_fair(
        df, servers_df, predictor=predictor, proactive=True
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
        df, servers_df, predictor=predictor, proactive=False
    )
    print(f"  DQN done in {time.time() - t0:.1f}s")

    t0 = time.time()
    reactive_results["Hybrid SAC"] = run_hybrid_sac_microservice(
        df, servers_df, predictor=predictor, proactive=False,
        inference_mode=True,
        checkpoint_path=SAC_CHECKPOINT_REACTIVE,
    )
    print(f"  Hybrid SAC done in {time.time() - t0:.1f}s")

    return proactive_results, reactive_results


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
        train_pro, train_rea, _ = run_training_phase(servers_df)
        wall_train = time.time() - t_train

        _print_results_and_plots(train_pro, train_rea)

        t_inf = time.time()
        infer_pro, infer_rea = run_inference_phase(servers_df)
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
        train_pro, train_rea, _ = run_training_phase(servers_df)
        proactive_results, reactive_results = train_pro, train_rea
    else:
        proactive_results, reactive_results = run_inference_phase(servers_df)

    _print_results_and_plots(proactive_results, reactive_results)
    generate_experiment_report(proactive_results, reactive_results, INFERENCE_MODE)


if __name__ == "__main__":
    main()
