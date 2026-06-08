# 历史实验启动脚本（归档）

根目录仅保留：
- `run_medium_validation_cov50.py` — 主流水线
- `run_reward_v21_p3_cov50.py` — **当前 B1.1b 推荐入口（reactive-only）**

本目录脚本用于**复现旧实验**，日常开发不必使用。

| 脚本 | 用途 |
|------|------|
| `run_reward_v2_medium_test_cov50.py` | v2.0 快筛（S=10000，2 epoch） |
| `run_reward_v2_softguard_train_cov50.py` | v2.0 全量 + proactive |
| `run_reward_v2_phaseB_aligned_cov50.py` | v2 vs v1 公平对照（MAX=1） |
| `run_phaseC_softguard_train_cov50.py` | v1 Phase C 标杆 |
| `run_full_pipeline_cov50.py` | 旧全量 pipeline（含 appendix） |

用法（在项目根目录）：

```bash
python -u scripts/archive/run_reward_v2_medium_test_cov50.py
```
