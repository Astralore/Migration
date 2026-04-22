# Reference — 原始 C-Migrate 单体服务迁移代码

此目录包含原始 C-Migrate 项目中的单体服务迁移算法实现，仅供参考对比，**不参与新微服务迁移系统的运行**。

## 文件说明

| 文件 | 内容 |
|------|------|
| `Context_Integratoin.py` | 主对比框架：8 种算法的统一运行入口 + 共享工具函数 |
| `DQN_Migration.py` | 单体 DQN 迁移 |
| `SA_Migration.py` | 单体模拟退火迁移 |
| `Q-Learning.py` | 表格型 Q-Learning 迁移 |
| `SA-Qlearning.py` | 混合 SA + Q-Learning |
| `ILP_Migration.py` | 整数线性规划迁移 |
| `Heuristic_Migratoins.py` | 启发式迁移策略 |
| `Main_func_Migration.py` | 主函数入口 |
| `Run_Algos_Migration.py` | 批量运行脚本 |
| `Predictive_Analysis_Functions.py` | 轨迹预测评估函数 |
