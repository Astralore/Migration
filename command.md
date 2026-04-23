角色设定：
你是一位顶尖的边缘计算与微服务架构专家。我们正在将现有的“单体服务迁移”仿真环境升级为“微服务 DAG 协同迁移”环境。

前置上下文：
请阅读项目中处理环境搭建、用户请求生成和状态记录的代码文件（如引擎初始化部分），以及 @microservice_knowledge_base.py 中的 MICROSERVICE_DAGS 字典。

Phase 1 核心任务：底层数据结构与感知环境重构
请帮我完成以下两步底层改造：

任务 1：动态分配 DAG 请求类型
在读取用户（出租车）轨迹并发起请求的初始化逻辑处，新增一个机制：当一个用户首次出现并发出请求时，使用 `np.random.choice` 根据 MICROSERVICE_DAGS 中各个 DAG 的 `probability`，为其随机分配一个真实的业务调用链类型（如 'Data_Heavy_DAG'），并将其持久化记录在该用户的属性中。

任务 2：重构分配状态字典 (State Tracking)
将原来记录设备分配位置的字典（例如原有的 `taxi_assignments[taxi_id] = server_id`）彻底重构为**嵌套字典**结构。
新的结构必须是：`taxi_dag_assignments[taxi_id][microservice_node_name] = server_id`。
初始化逻辑：当车辆首次被分配边缘服务器时，遍历该用户对应的 DAG 中的所有 `nodes`，将它们默认**全部部署在距离用户当前位置最近的同一个边缘服务器上**。

输出要求：
1. 请给出具体的代码修改方案和代码片段。
2. 现阶段**绝对不要**修改深度强化学习（DQN）的网络结构、Reward计算和动作决策逻辑（留到后续阶段）。
3. 请在修改完后，提供一段简短的 print 调试代码，用于在控制台打印出一辆车的 `taxi_dag_assignments` 结构，以验证嵌套字典是否生成成功。
Phase 1 的底层结构已经完美跑通。现在我们严格按照计划书，进入 Phase 2。

请参考 @dqn_microservice_migration_55cdc05c.plan.md 中的 `[id: reward-fn]` 模块要求，在 `DQN_Microservice_Migration.py` 中帮我实现全新的奖励计算核心函数：
`calculate_microservice_reward(taxi_id, dag_info, current_assignments, previous_assignments, user_location, servers_info, alpha=1.0, beta=0.01, gamma=1.0)`

核心实现要求（请严格遵守计划书逻辑）：
1. 外部延迟惩罚 (access_latency)：利用之前写好的 `get_entry_nodes` 找到入口节点，计算真实物理用户（user_location）到这些入口节点所在服务器的距离之和。
2. 图内通信惩罚 (communication_cost)：遍历该 DAG 的所有 `edges`。若某条边的两个微服务被分配在不同的服务器，计算这两个服务器之间的物理距离 `dist`。惩罚值 = `(traffic / max_traffic) * dist`（注意：必须对 traffic 进行内部归一化，除以该 DAG 中的最大 traffic，以防数值爆炸）。若同服则距离惩罚为 0。
3. 状态迁移惩罚 (migration_cost)：遍历所有微服务，若 `current_assignments[node]` != `previous_assignments[node]`，说明发生了迁移。惩罚值 = `(image_mb + state_mb) / 100.0`（假设带宽 BANDWIDTH 为 100.0）。
4. 汇总返回：返回总的 Reward（这三项加权和的负数），以及一个包含这三项明细的字典（方便后续我们打 log 分析）。

请给出这个函数的完整 Python 实现代码，并保持清晰的代码注释。注意处理可能存在的边界情况（如 max_traffic 为 0 的除零防御）。依然不要在这一步编写 DQN 网络和决策循环。
Phase 2 的核心奖励函数已经完美通过验证！现在我们严格按照计划书 @dqn_microservice_migration_55cdc05c.plan.md 进入最具挑战的 Phase 3。

请帮我在 `DQN_Microservice_Migration.py` 中实现 `[id: dqn-network]` 和 `[id: decision-loop]` 模块。具体包含以下三大任务：

**任务 1：重构 DQN 神经网络**
定义 `class MicroserviceDQN(nn.Module)`：
- `input_size = 14` (由于增加了微服务节点和拓扑特征)
- `hidden_size = 128` (增加容量以处理更复杂的特征)
- `action_size = 4` (0:保持, 1:候选1, 2:候选2, 3:候选3)
- 网络结构：Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Q-values。

**任务 2：实现状态构建函数 `build_node_state`**
编写一个辅助函数 `build_node_state(taxi, ms_node, dag_info, current_assignments, candidates)`，严格按照计划书组装并返回 14 维的归一化状态向量（list 或 tensor 均可）。
特征顺序如下：
1-6 (全局): 纬度/90, 经度/180, 距离/50, 小时/24, 星期/7, 候选数/10。
7-10 (节点): image_mb/200, state_mb/256, is_stateful(float), 节点关联总流量/最大流量(注意防除零)。
11-14 (拓扑): 邻居同服比例(注意防除零), 当前节点服务器距出租车距离/50, 是否为入口节点(float), DAG节点总数/10。

**任务 3：重构决策主循环 (Sequential Decision Loop)**
在主仿真函数 `run_dqn_microservice_fair()` 的核心逻辑中（当出租车发生违规并触发迁移时），实现微服务逐层安置逻辑：
1. 获取该车的 `dag_info`，备份 `old_assignments`。
2. 使用 `topological_sort(dag_info)` 获取节点遍历顺序。
3. **内循环**：遍历 sorted_nodes。针对每个 `ms_node`：
   - 调用 `build_node_state`
   - epsilon-greedy 选择 action
   - 将该微服务部署到选定的 new_server，**立即更新** `taxi_dag_assignments`
   - 将 `(state, action)` 暂存到 `node_transitions` 列表
4. **DAG 级评估**：内循环结束后，调用 `calculate_microservice_reward` 计算全局 reward。
5. **稀疏奖励与经验回放 (Sparse Reward & Replay Buffer)**：
   - 遍历 `node_transitions`，构造马尔可夫链。
   - 只有最后一个节点 `is_last = True` 时，`step_reward = 全局 reward`，`next_state = 当前 state`，`done = True`。
   - 中间节点 `step_reward = 0.0`，`next_state = 下一个节点的 state`，`done = False`。
   - 压入 `memory.append`。
6. **统计 M 和 V**：
   - 按照计划书，M 累加实际发生跨服变更的**微服务节点数**（逐一对比 old vs new）。
   - V 依然统计入口网关节点是否距离 > 15km。

请一步步思考，给出包含这三个任务的核心代码。保持代码结构清晰，注释详尽。
Phase 3 的神经网络和决策主循环验证完美！这套稀疏奖励的马尔可夫链非常标准。现在我们进入计划书的最后一步：Phase 4 [id: main-sim]。

请帮我完成 `DQN_Microservice_Migration.py` 的全流程组装、网络训练逻辑以及独立运行入口。包含以下三大任务：

**任务 1：实现 DQN 的经验回放与训练函数 (Experience Replay & Optimization)**
请在类外或类内实现标准的 DQN `optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma)` 函数。
- 从 memory 中随机采样一个 batch_size (如 32) 的 (state, action, reward, next_state, done)。
- 计算当前 Q 值: `Q(s, a)`。
- 计算目标 Q 值: 对于 done=True 的，`target = reward`；对于 done=False 的，`target = reward + gamma * max(Q(next_s))`。
- 计算 MSE Loss，执行 `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`。
- 请在 run_dqn_microservice_fair 的每一帧循环末尾调用此函数，并定期（如每 50 步）将 policy_net 的权重视配给 target_net。记录平均 loss 以备画图。

**任务 2：完善真实时序主循环 (Time-step Loop)**
确保 `run_dqn_microservice_fair` 能够接收真实的 DataFrame (`df` 和 `servers_df`)，并像旧代码一样按时间步推进：
`for current_time, group in df.groupby('date_time'):`
在循环中：
- 为首次出现的出租车分配 DAG 和初始服务器。
- 判断违规（网关节点距离 > 15km）及 check_migration_criteria 健康门控。
- 触发迁移时，执行我们 Phase 3 写好的 DAG 逐节点决策循环。
- 统计总指标：`total_migrations` (M) 累加实际发生跨服变更的微服务节点数；`total_violations` (V) 累加网关节点超距的次数。

**任务 3：独立运行入口与可视化**
编写 `if __name__ == '__main__':` 块：
- 调用 `load_data` (复用原有或写个简易加载) 读取真实数据集。
- 实例化网络并调用 `run_dqn_microservice_fair`。
- 运行结束后，在控制台打印格式化结果：`Migrations (M): xxx, Violations (V): xxx, Score (M+0.5V): xxx`。
- （可选）使用 matplotlib 绘制一条随着 Episode/TimeStep 变化的 Loss 曲线或 Total Reward 曲线，保存为 `dqn_microservice_training.png`。

请给出这部分完整的组装代码。确保它可以作为一个独立脚本直接运行（Python DQN_Microservice_Migration.py）。太棒了！Phase 4 的 DQN 微服务仿真已经完美跑通，指标和曲线都符合预期。
现在我们进入 Phase 5：重构混合算法的前置模块 —— 拓扑感知模拟退火算法 (Topology-Aware SA)。

请帮我创建一个新文件 `SA_Microservice_Migration.py`。
我们的目标是实现一个纯 SA 的微服务迁移对比基线。请包含以下核心任务：

**任务 1：复用底层生态**
- 导入 `microservice_knowledge_base.py` 中的 `MICROSERVICE_DAGS`。
- 从 `DQN_Microservice_Migration.py` 中引入我们写好的 `calculate_microservice_reward`，以及 `check_migration_criteria`, `topological_sort`, `get_entry_nodes` 等基础辅助函数。

**任务 2：实现微服务级别的模拟退火 (Simulated Annealing)**
编写核心函数 `microservice_simulated_annealing(taxi_id, dag_info, current_assignments, candidates, user_location, servers_info, temp=100.0, cooling_rate=0.95, max_iter=30)`。
- 目标函数 (Cost)：`cost = -calculate_microservice_reward(...)的返回值`（因为 Reward 是负数惩罚，我们要最小化 Cost）。
- 邻域扰动逻辑 (Neighbor Generation)：每次迭代，不要改变所有节点！**随机从 DAG 中挑选 1 个微服务节点**，将其在 `current_assignments` 中的位置随机更改为 `candidates` 中的另一个服务器，生成 `neighbor_assignments`。
- 接受准则：计算 `neighbor_cost`。如果 `delta = neighbor_cost - current_cost < 0`，则接受；否则以 `exp(-delta/temp)` 的概率接受。
- 返回跑完迭代后的 `best_assignments`。

**任务 3：主仿真循环与独立运行**
编写 `run_sa_microservice_fair()`：
- 整体时间步、DataFrame 读取、DAG 类型首次分配、违规触发逻辑与 DQN 完全保持一致。
- 唯一区别：当触发迁移时，不再调用 DQN 网络，而是直接调用 `microservice_simulated_annealing` 获取新的部署方案，并更新字典。
- 同样统计并返回 `Migrations (M)` 和 `Violations (V)`，以及 `Score (M + 0.5V)`。

请给出完整的代码实现，并提供 `if __name__ == '__main__':` 运行入口。
太棒了！Phase 4 的 DQN 微服务仿真已经完美跑通。现在我们保持当前的扁平目录结构不变，直接进入 Phase 5：重构混合算法的前置模块 —— 拓扑感知模拟退火算法 (Topology-Aware SA)。

请帮我创建一个新文件 `SA_Microservice_Migration.py`。
我们的目标是实现一个纯 SA 的微服务迁移对比基线。请包含以下核心任务：

**任务 1：复用底层生态**
- 导入 `microservice_knowledge_base.py` 中的 `MICROSERVICE_DAGS`。
- 从 `DQN_Microservice_Migration.py` 中引入我们写好的 `calculate_microservice_reward`，以及 `check_migration_criteria`, `topological_sort`, `get_entry_nodes` 等基础辅助函数。

**任务 2：实现微服务级别的模拟退火 (Microservice SA)**
编写核心函数 `microservice_simulated_annealing(taxi_id, dag_info, current_assignments, candidates, user_location, servers_info, temp=100.0, cooling_rate=0.95, max_iter=30)`。
- 目标函数 (Cost)：`cost = -calculate_microservice_reward(...)` 的总 Reward 返回值（因为 Reward 是负数惩罚，我们要最小化 Cost）。
- 邻域扰动逻辑 (Neighbor Generation)：每次迭代，**不要改变所有节点！** 随机从 DAG 中挑选 **1 个**微服务节点，将其在 `current_assignments` 中的位置随机更改为 `candidates` 中的另一个服务器，生成 `neighbor_assignments`。
- 接受准则 (Acceptance Criteria)：计算 `neighbor_cost`。如果 `delta = neighbor_cost - current_cost < 0`，则接受新解；否则以 `math.exp(-delta/temp)` 的概率接受。
- 降温：`temp *= cooling_rate`。
- 返回跑完所有迭代后的 `best_assignments`。

**任务 3：主仿真循环与独立运行**
编写 `run_sa_microservice_fair()`：
- 整体时间步 `for current_time, group in df.groupby('date_time'):` 的读取、DAG 类型首次分配、违规触发逻辑，与 DQN 文件中完全保持一致。
- **唯一区别**：当触发迁移时，不再调用 DQN 网络，而是直接调用 `microservice_simulated_annealing` 获取新的部署方案，并更新字典 `taxi_dag_assignments`。
- 同样统计并返回 `Migrations (M)` 和 `Violations (V)`，以及 `Score (M + 0.5V)`。

请给出完整的代码实现，并提供 `if __name__ == '__main__':` 运行入口（打印出最终的 M, V, 和 Score）。保持代码结构清晰。
Phase 5 的纯 SA 基线已经完美跑通并给出了符合预期的对比结果！现在我们正式进入整个研究的核心创新点 —— Phase 6：微服务混合决策算法 (Hybrid RL-Refined SA)。

请帮我创建一个新文件 `Hybrid_Microservice_Migration.py`。
我们将把 SA 的全局启发式搜索能力和 DQN 的序列化价值评估能力结合起来。具体任务如下：

**任务 1：复用基础设施与函数**
- 导入 `microservice_knowledge_base.py`。
- 从 `DQN_Microservice_Migration.py` 或 `SA_Microservice_Migration.py` 中引入所需的所有底层通用函数（Reward计算、拓扑排序、SA核心逻辑等）。

**任务 2：重构 HybridDQN 神经网络与状态空间**
定义 `class HybridMicroserviceDQN(nn.Module)`：
- **Input Size 增加到 16 维**：在原来 14 维的基础上，追加 2 维“SA 先验建议特征”：
  1. `sa_server_dist / 50.0` (SA 为该节点建议的服务器到出租车的距离)
  2. `1.0 if sa_proposed_server == current_server else 0.0` (SA 是否建议该节点保持原地)
- **Action Size 缩减为 3 维**：
  - `action = 0`：保持当前服务器 (Stay)
  - `action = 1`：接受 SA 为该节点推荐的服务器 (Follow SA)
  - `action = 2`：强制移动到绝对距离最近的候选服务器 (Nearest)
- 网络结构保持 `Linear(128) -> ReLU -> Linear(128) -> ReLU`。

**任务 3：实现 Hybrid 决策主循环**
重写迁移触发时的决策逻辑 `run_hybrid_microservice_fair()`：
1. **获取先验草案**：当违规触发迁移时，首先调用 `microservice_simulated_annealing(...)`。获取 SA 给出的全局建议字典，命名为 `sa_proposal_assignments`。
2. **DQN 逐节点审查**：按照拓扑排序遍历 DAG 的每个节点。
3. **构建状态**：调用修改后的 `build_hybrid_node_state`，将 `sa_proposal_assignments[ms_node]` 的相关特征拼接入 16 维状态向量。
4. **动作映射**：DQN 输出 action (0, 1, 或 2)，根据上述映射规则转化为具体的 `target_server_id`，并立即更新到 `taxi_dag_assignments` 环境中。
5. **稀疏奖励与训练**：同 DQN 版本，等所有节点决策完毕后计算全局 Reward，分配给马尔可夫链的最后一步并存入 Replay Buffer，进行 `optimize_model`。

**任务 4：独立运行入口**
- 同样包含数据加载和主循环，并在控制台输出最终的 M、V 和 Score。
- 绘制 Loss 曲线保存为 `hybrid_microservice_training.png`。

请给出完整的代码实现，重点确保 DQN 的 Action 映射和 State 构建逻辑清晰无误。
我的项目已经重构为标准的模块化目录。现在我们需要进行关键的“去背景化”改造。我们的研究聚焦于“边缘微服务协同迁移”，因此必须彻底剥离原有代码中与“医疗/健康/心血管风险”相关的所有逻辑。

请帮我修改 `core` 目录和 `algorithms/hybrid_sa_dqn.py` 中的代码，完成以下三大任务：

**任务 1：清洗数据源 (`core/data_loader.py`)**
- 修改 `load_data` 函数。在读取 `taxi_with_health_info.csv`（或任何 DataFrame）时，忽略或删除所有健康特征列（如 `age`, `heart_rate`, `risk_level`, `CVD_risk` 等）。
- 确保返回的 `df` 只包含物理移动相关的核心字段：`taxi_id`, `date_time`, `latitude`, `longitude`。

**任务 2：重构触发门控 (`core/context.py`)**
- 删除原有的 `check_migration_criteria` 函数（它依赖健康数据）。
- 新增函数 `check_sla_violation(user_lat, user_lon, gateway_server_lat, gateway_server_lon, current_dag_reward)`。
- **触发逻辑**：当满足以下任一条件时，返回 True（触发迁移）：
  1. **空间违规**：`haversine_distance` 计算的用户与网关节点服务器距离 > 15.0 (km)。
  2. **性能违规 (QoS/SLA)**：`current_dag_reward` < -5.0（意味着当前微服务拓扑的通信/延迟开销过大，-5.0 作为一个可调的 SLA_THRESHOLD）。

**任务 3：净化状态空间 (`core/state_builder.py`)**
- 检查 `build_node_state` 和 `build_hybrid_node_state` 函数。
- 确保最终生成的 14/16 维状态向量中，没有任何占位或遗留的健康特征。
- 如果之前有空缺的维度，请替换为更符合微服务研究的“系统上下文特征”：例如添加一维 `server_load`（目前可以用 `random.uniform(0.1, 0.9)` 模拟目标服务器的 CPU 负载，后续我们再接入真实统计算法）。

**任务 4：更新主循环 (`algorithms/hybrid_sa_dqn.py`)**
- 在 `run_hybrid_microservice_fair` 的时间步循环中，移除所有获取 `health_info` 的代码。
- 将触发判断替换为调用新的 `check_sla_violation` 函数（你需要先计算当前状态的 reward 传入，或者简化为先判断距离，如果距离没超，再单独测算通信开销）。
- 清理所有 `print` 日志，将 "Health Risk / High Risk" 等字眼替换为 "SLA Violation / QoS Degradation"。

请一步步思考，并给出上述文件的具体修改代码。保持代码的模块化调用关系不被破坏。
Phase 7 的 SLA 去背景化重构完美成功！现在我们进入 Phase 8：接入轨迹预测模块，实现“主动式微服务迁移（Proactive Migration）”。

请帮我修改相关模块，严格参考我们在论文中关于 Predictive Performance Reward 和 Proactive Trigger 的数学定义。具体任务如下：

**任务 1：在主循环中接入预测器 (`algorithms/hybrid_sa_dqn.py` 等主入口)**
- 引入 `prediction/simple_predictor.py` 中的轨迹预测器（或者当前项目中现成的预测类）。
- 在时间步循环 `for current_time, group in df.groupby('date_time'):` 中，对当前出租车调用预测器，获取其未来 $H$ 步的预测坐标列表 `predicted_locations = [(lat1, lon1), ..., (lat_H, lon_H)]`（设 $H=5$）。

**任务 2：重构触发门控 —— 增加主动触发 (`core/context.py`)**
- 修改 `check_sla_violation`（或新增 `check_proactive_sla_violation`）。
- **新逻辑**：不仅检查当前距离是否 > 15km 和当前 QoS 是否 < -5.0，还要遍历 `predicted_locations`。
- 如果预测在未来 $H$ 步内，当前网关节点距离将 > 15km，则**提前触发**迁移（返回 True）。

**任务 3：升级 Reward 函数 —— 引入未来违规惩罚 (`core/reward.py`)**
- 修改 `calculate_microservice_reward` 的签名，增加参数 `predicted_locations`（默认为 None）。
- 如果传入了预测轨迹，请增加一项 **$C_{future}$ (未来拓扑违规惩罚)**：
  遍历 `predicted_locations`，计算每个预测点到候选网关节点服务器的距离。如果距离 > 15km，则施加惩罚（可以随时间步增加而衰减权重，例如 $w_h = 0.9^h$）。
- 最终的 `total_reward` 变为：`alpha * access_latency + beta * communication + gamma * migration + delta * future_violation_penalty`。

**任务 4：(可选) 状态空间升级 (`core/state_builder.py`)**
- 如果状态空间 16 维还有余量，或者可以扩充到 18 维，请将预测轨迹的“移动趋势”（如经纬度的平均变化率 $\Delta lat, \Delta lon$）加入 DQN 的输入特征中，让 DQN 能“看到”车辆的运动方向。

请一步步思考，修改这四个模块的逻辑，确保我们可以通过简单的开关控制是否开启“主动预测 (Proactive)”功能，以便后续做对比实验。
我们需要修复主循环中的评价指标统计逻辑。当前代码在 Proactive 模式下，错误地将“主动预测触发的迁移”计入了“实际违规次数 (V)”，导致指标失真。

请帮我修改 `algorithms/hybrid_sa_dqn.py` (以及对比脚本) 中的主时间步循环（`for current_time, group in df.groupby('date_time'):`），将“记分逻辑”与“触发逻辑”彻底分离：

**任务 1：剥离实际违规统计 (V) 的计算**
在遍历每个出租车，且在进行任何迁移判定（`check_proactive_sla_violation`）**之前**，先进行纯粹的“记分”：
- 获取该出租车当前分配的入口网关服务器（Gateway Server）的坐标。
- 计算 `actual_distance = haversine_distance(user_lat, user_lon, gateway_lat, gateway_lon)`。
- 如果 `actual_distance > 15.0`，则 `total_violations += 1`。
- *(注意：这是唯一的 V 累加入口！无论后续是否触发迁移，V 的统计只看这一个客观事实。)*

**任务 2：独立记录触发次数 (Decisions) - 可选但建议**
- 新增一个统计变量 `total_decisions = 0`。
- 当 `check_proactive_sla_violation` 返回 True 并启动 Hybrid 算法进行重部署时，执行 `total_decisions += 1`。

**任务 3：修正控制台打印和绘图输出**
- 运行结束后的打印格式改为：`Migrations (M): xxx, Real Violations (V): xxx, Proactive Decisions (D): xxx, Score (M+0.5V): xxx`。
- 确保所有的对比算法（Reactive DQN, SA, Hybrid）都采用这套分离的统计逻辑。

请严格执行，确保 `V` 的物理意义恢复为“用户真实感知到的 SLA 断连次数”。
指标分离已经完美验证了 Proactive 的威力！现在我们要在此基础上实现论文的核心创新点：“计算与状态解耦的非对称主动迁移（Asymmetric Proactive Migration）”。

请帮我修改 `core/reward.py` 和 `algorithms/hybrid_sa_dqn.py`（及其他相关算法脚本），将底层物理传输的“后台静默同步”与“前台阻塞迁移”的区别体现在 Reward 数学模型中。

**任务 1：修改奖励函数 (`core/reward.py`)**
- 修改 `calculate_microservice_reward`，新增一个布尔参数 `is_proactive_trigger=False`。
- **重构 Migration Cost 的计算逻辑**：
  遍历需要迁移的微服务节点，判断其是否为“有状态” (`node_info.get('is_stateful', False)`)。
  - **如果是 Proactive 触发 (`is_proactive_trigger == True`)**：说明我们在未来违规前有充足的时间。有状态节点的 `state_mb` 可以在后台慢速同步，不阻塞当前业务。
    `cost = (image_mb / 100.0) + (state_mb / 1000.0)`  # 极低的后台网络带宽惩罚
  - **如果是 Reactive 触发 (`is_proactive_trigger == False`)**：说明用户已经断连（距离>15km 或 QoS<-5.0），此时必须强行阻塞业务进行全量拷贝，代价极其高昂。
    `cost = (image_mb / 100.0) + (state_mb / 10.0)`   # 极其高昂的前台阻塞惩罚（权重放大100倍）
  - 无状态节点（`is_stateful == False`）不受影响，依然是 `(image_mb + 0) / 100.0`。

**任务 2：更新触发侧的传参 (`algorithms/hybrid_sa_dqn.py` 等)**
- 在时间步循环中，我们需要知道当前触发到底是哪种类型：
  ```python
  # 优先判断是否发生真实 SLA 崩溃 (Reactive)
  is_reactive = check_sla_violation(...) 
  # 其次判断是否未来会崩溃 (Proactive)
  is_proactive = False
  if not is_reactive and PROACTIVE_MODE:
      is_proactive = check_proactive_sla_violation(..., predicted_locations)

  if is_reactive or is_proactive:
      # 触发迁移
      ...
      # 在计算 Reward 时传入触发类型
      reward = calculate_microservice_reward(..., is_proactive_trigger=is_proactive)
      我们现在的项目处于 Phase 8（主动式预测接入阶段）。为了生成最终的顶会论文实验数据，我需要你在现有的模块化架构下，实现论文的两个核心亮点：“评估指标的双轨解耦”与“计算/状态非对称主动迁移”。

请严格遵循现有目录结构修改以下文件：

**任务 1：实现“非对称主动迁移”底层逻辑 (`core/reward.py` & `core/context.py`)**
1. 修改 `core/context.py` 中的门控判断逻辑。需要明确区分触发类型，建议让检查函数返回具体的 trigger_type，如 `'REACTIVE'`（已违规）、`'PROACTIVE'`（预测将违规）或 `None`。
2. 修改 `core/reward.py` 中的 `calculate_microservice_reward`，增加参数 `trigger_type='REACTIVE'`。
3. **核心公式修改**：在计算 `migration_cost` 时，遍历微服务节点：
   - 对于无状态节点（`is_stateful == False`）：代价保持不变，即 `image_mb / 100.0`。
   - 对于有状态节点（`is_stateful == True`）：
     - 如果 `trigger_type == 'PROACTIVE'`：采用后台静默同步极低惩罚：`cost = (image_mb / 100.0) + (state_mb / 1000.0)`
     - 如果 `trigger_type == 'REACTIVE'`：采用前台阻塞断网极高惩罚：`cost = (image_mb / 100.0) + (state_mb / 10.0)`
   - 累加作为最终的迁移惩罚。

**任务 2：统计算法的决策传参 (`algorithms/hybrid_sa_dqn.py` 及其他算法)**
1. 在主时间步循环中，调用 `core/context.py` 获取确切的 `trigger_type`。
2. 将该 `trigger_type` 传入 Hybrid 算法（以及 SA 草案评估）的 `calculate_microservice_reward` 中，确保环境反馈真实的非对称 Reward。
3. **指标统计剥离**：
   - **Real_V (真实违规)**：在每个时间步判定前，单独计算当前实际网关距离，若 >15km 则 `Real_V += 1`。这代表用户真实感知的断连。
   - **Decisions_D (主动决策)**：若 `trigger_type == 'PROACTIVE'`，则 `Decisions_D += 1`。代表系统底层进行的隐性预判。
   - **Migrations_M (迁移数)**：依然统计实际发生了跨服调度的微服务节点数。

**任务 3：完善对比实验报告输出 (`evaluation/metrics.py` & `run_comparison.py`)**
1. 升级最终结果打印格式。对于每一种算法，必须打印出以下完整信息：
   `[{Algorithm Name}] M (Nodes Moved): xxx | D (Proactive Decisions): xxx | Real_V (SLA Drops): xxx | Score (M + 0.5*Real_V): xxx`
2. 在 `run_comparison.py` 的总结处，利用 logging 或 print 打印一段“实验分析说明”（用于辅助写论文）：
   - 如果运行了 Reactive 算法和 Proactive 算法，自动计算并打印：“相比被动模式，Proactive 模式在底层多执行了 {Delta_D} 次预判，成功将真实服务中断 (Real_V) 从 {Old_V} 压降至 {New_V}，下降率 {Drop_Rate}%！”

请一步步思考并执行上述修改，保证代码的整洁性和逻辑的严密性。完成后提供一个简短的总结说明修改了哪些核心行。