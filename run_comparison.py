#!/usr/bin/env python
"""
Entry point: run all three microservice migration algorithms and compare.
Supports both Proactive and Reactive modes for full experimental analysis.
"""

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
INFERENCE_MODE = True  # False=训练模式, True=推理模式

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


def generate_experiment_report(proactive_results, reactive_results, is_inference_mode):
    """自动生成 result.md 实验报告。"""
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
    
    # 提取时延数据进行对比
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


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print("=" * 80)
    print("  Microservice Migration — Full Comparison (Asymmetric Cost Model)")
    print(f"  Mode: {'INFERENCE' if INFERENCE_MODE else 'TRAINING'}")
    print("=" * 80)

    # 加载服务器数据（两种模式都需要）
    servers_df = pd.read_csv(DEFAULT_SERVER_PATH)
    print(f"  Edge servers loaded: {len(servers_df)}")

    if not INFERENCE_MODE:
        # =====================================================================
        # 训练模式：使用训练数据，训练后保存权重
        # =====================================================================
        print(f"\n[MODE] Training Mode - Data: [{TRAIN_START_INDEX}:{TRAIN_END_INDEX}]")
        df = load_data(DEFAULT_TAXI_PATH, start_index=TRAIN_START_INDEX, end_index=TRAIN_END_INDEX)
        
        # 训练预测器
        predictor = SimpleTrajectoryPredictor(forecast_horizon=FORECAST_HORIZON)
        predictor.fit(df)
        print(f"  Predictor fitted: {len(predictor.velocity_factors)} taxis with velocity data")
        
        # --- Proactive 模式 ---
        print("\n" + "#" * 80)
        print("  PHASE 1: Proactive Mode (Training)")
        print("#" * 80)
        proactive_results = {}
        
        # SA (无需训练)
        print(f"\n{'=' * 60}")
        print(f"  [Proactive] Running SA ...")
        print(f"{'=' * 60}")
        t0 = time.time()
        proactive_results["SA"] = run_sa_microservice_fair(
            df, servers_df, predictor=predictor, proactive=True
        )
        print(f"  SA done in {time.time() - t0:.1f}s")
        
        # DQN (需要训练)
        print(f"\n{'=' * 60}")
        print(f"  [Proactive] Running DQN ...")
        print(f"{'=' * 60}")
        t0 = time.time()
        proactive_results["DQN"] = run_dqn_microservice_fair(
            df, servers_df, predictor=predictor, proactive=True
        )
        print(f"  DQN done in {time.time() - t0:.1f}s")
        
        # Hybrid SAC (训练 + 保存权重)
        print(f"\n{'=' * 60}")
        print(f"  [Proactive] Running Hybrid SAC (Training + Save) ...")
        print(f"{'=' * 60}")
        t0 = time.time()
        proactive_results["Hybrid SAC"] = run_hybrid_sac_microservice(
            df, servers_df, predictor=predictor, proactive=True, num_epochs=6,
            inference_mode=False,
            save_checkpoint_path=SAC_CHECKPOINT_PROACTIVE,
        )
        print(f"  Hybrid SAC done in {time.time() - t0:.1f}s")
        
        # --- Reactive 模式 ---
        print("\n" + "#" * 80)
        print("  PHASE 2: Reactive Mode (Training)")
        print("#" * 80)
        reactive_results = {}
        
        print(f"\n{'=' * 60}")
        print(f"  [Reactive] Running SA ...")
        print(f"{'=' * 60}")
        t0 = time.time()
        reactive_results["SA"] = run_sa_microservice_fair(
            df, servers_df, predictor=predictor, proactive=False
        )
        print(f"  SA done in {time.time() - t0:.1f}s")
        
        print(f"\n{'=' * 60}")
        print(f"  [Reactive] Running DQN ...")
        print(f"{'=' * 60}")
        t0 = time.time()
        reactive_results["DQN"] = run_dqn_microservice_fair(
            df, servers_df, predictor=predictor, proactive=False
        )
        print(f"  DQN done in {time.time() - t0:.1f}s")
        
        print(f"\n{'=' * 60}")
        print(f"  [Reactive] Running Hybrid SAC (Training + Save) ...")
        print(f"{'=' * 60}")
        t0 = time.time()
        reactive_results["Hybrid SAC"] = run_hybrid_sac_microservice(
            df, servers_df, predictor=predictor, proactive=False, num_epochs=6,
            inference_mode=False,
            save_checkpoint_path=SAC_CHECKPOINT_REACTIVE,
        )
        print(f"  Hybrid SAC done in {time.time() - t0:.1f}s")
        
    else:
        # =====================================================================
        # 推理模式：使用测试数据，加载权重，纯推理对比
        # =====================================================================
        print(f"\n[MODE] Inference Mode - Data: [{TEST_START_INDEX}:{TEST_END_INDEX}]")
        
        # ⚠️ 关键：用训练数据拟合预测器（保持历史学习到的移动模式）
        # 这样才是真正的"在线推理"场景：预测器基于历史数据，面对全新轨迹
        train_df = load_data(DEFAULT_TAXI_PATH, start_index=TRAIN_START_INDEX, end_index=TRAIN_END_INDEX)
        predictor = SimpleTrajectoryPredictor(forecast_horizon=FORECAST_HORIZON)
        predictor.fit(train_df)
        print(f"  Predictor fitted on TRAINING data [{TRAIN_START_INDEX}:{TRAIN_END_INDEX}]")
        
        # 加载测试数据进行评测（预测器从未见过这些数据）
        df = load_data(DEFAULT_TAXI_PATH, start_index=TEST_START_INDEX, end_index=TEST_END_INDEX)
        
        # --- Proactive 模式 ---
        print("\n" + "#" * 80)
        print("  PHASE 1: Proactive Mode (Inference)")
        print("#" * 80)
        proactive_results = {}
        
        # SA (启发式，每次都重新计算)
        print(f"\n{'=' * 60}")
        print(f"  [Proactive] Running SA (Inference) ...")
        print(f"{'=' * 60}")
        t0 = time.time()
        proactive_results["SA"] = run_sa_microservice_fair(
            df, servers_df, predictor=predictor, proactive=True
        )
        print(f"  SA done in {time.time() - t0:.1f}s")
        
        # DQN (也需要在测试集上运行)
        print(f"\n{'=' * 60}")
        print(f"  [Proactive] Running DQN (Inference) ...")
        print(f"{'=' * 60}")
        t0 = time.time()
        proactive_results["DQN"] = run_dqn_microservice_fair(
            df, servers_df, predictor=predictor, proactive=True
        )
        print(f"  DQN done in {time.time() - t0:.1f}s")
        
        # Hybrid SAC (加载权重，纯推理)
        print(f"\n{'=' * 60}")
        print(f"  [Proactive] Running Hybrid SAC (Inference, Load Weights) ...")
        print(f"{'=' * 60}")
        t0 = time.time()
        proactive_results["Hybrid SAC"] = run_hybrid_sac_microservice(
            df, servers_df, predictor=predictor, proactive=True,
            inference_mode=True,
            checkpoint_path=SAC_CHECKPOINT_PROACTIVE,
        )
        print(f"  Hybrid SAC done in {time.time() - t0:.1f}s")
        
        # --- Reactive 模式 ---
        print("\n" + "#" * 80)
        print("  PHASE 2: Reactive Mode (Inference)")
        print("#" * 80)
        reactive_results = {}
        
        print(f"\n{'=' * 60}")
        print(f"  [Reactive] Running SA (Inference) ...")
        print(f"{'=' * 60}")
        t0 = time.time()
        reactive_results["SA"] = run_sa_microservice_fair(
            df, servers_df, predictor=predictor, proactive=False
        )
        print(f"  SA done in {time.time() - t0:.1f}s")
        
        print(f"\n{'=' * 60}")
        print(f"  [Reactive] Running DQN (Inference) ...")
        print(f"{'=' * 60}")
        t0 = time.time()
        reactive_results["DQN"] = run_dqn_microservice_fair(
            df, servers_df, predictor=predictor, proactive=False
        )
        print(f"  DQN done in {time.time() - t0:.1f}s")
        
        print(f"\n{'=' * 60}")
        print(f"  [Reactive] Running Hybrid SAC (Inference, Load Weights) ...")
        print(f"{'=' * 60}")
        t0 = time.time()
        reactive_results["Hybrid SAC"] = run_hybrid_sac_microservice(
            df, servers_df, predictor=predictor, proactive=False,
            inference_mode=True,
            checkpoint_path=SAC_CHECKPOINT_REACTIVE,
        )
        print(f"  Hybrid SAC done in {time.time() - t0:.1f}s")
    
    # 打印结果（含时延对比）
    print("\n" + "#" * 80)
    print("  Proactive Mode Results (with Latency)")
    print("#" * 80)
    print_ranking_with_latency(proactive_results)
    
    print("\n" + "#" * 80)
    print("  Reactive Mode Results (with Latency)")
    print("#" * 80)
    print_ranking_with_latency(reactive_results)
    
    # 自动生成实验报告
    generate_experiment_report(proactive_results, reactive_results, INFERENCE_MODE)
    
    # Print comparative analysis
    print_proactive_analysis(proactive_results, reactive_results)

    # Final summary for paper
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

    # Generate visualization charts (only if not in inference mode or if outputs dir exists)
    os.makedirs("outputs", exist_ok=True)
    print("\n" + "#" * 80)
    print("  GENERATING VISUALIZATIONS")
    print("#" * 80)

    # Training curves for DQN-based methods
    for name, key in [("DQN", "DQN"), ("Hybrid_SAC", "Hybrid SAC")]:
        for mode, results in [("proactive", proactive_results), ("reactive", reactive_results)]:
            res = results[key]
            if res.get('loss_history'):
                plot_training_curves(
                    res,
                    save_path=f"outputs/{name.lower()}_{mode}_training.png",
                    title_prefix=f"{name} ({mode.capitalize()})",
                )

    # Cost breakdown chart (Proactive mode)
    plot_cost_breakdown(proactive_results, save_path="outputs/cost_breakdown.png")

    # Violation comparison chart (Proactive vs Reactive)
    plot_performance_metrics(proactive_results, reactive_results, save_path="outputs/violation_comparison.png")

    print("\n  All visualizations generated successfully!")


if __name__ == "__main__":
    main()
