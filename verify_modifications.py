#!/usr/bin/env python3
"""
快速验证修改的代码可运行性
"""

import sys
import torch

# 验证导入
try:
    from algorithms.marl_gat import run_marl_gat_microservice
    from core.marl_reward import calculate_marl_rewards
    from core.context import TRIGGER_PROACTIVE, TRIGGER_REACTIVE
    print("✓ 所有导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 验证函数签名
import inspect

sig = inspect.signature(calculate_marl_rewards)
params = list(sig.parameters.keys())

required_params = [
    'taxi_id', 'dag_info', 'current_assignments', 'previous_assignments',
    'user_location', 'servers_info', 'dense_distance_bonus_max'
]

for param in required_params:
    if param not in params:
        print(f"✗ 缺少参数: {param}")
        sys.exit(1)

print(f"✓ calculate_marl_rewards 签名正确，包含 dense_distance_bonus_max 参数")

# 验证默认值
defaults = {
    name: param.default 
    for name, param in sig.parameters.items()
    if param.default != inspect.Parameter.empty
}

if 'dense_distance_bonus_max' in defaults:
    print(f"✓ dense_distance_bonus_max 默认值: {defaults['dense_distance_bonus_max']}")
else:
    print(f"✓ dense_distance_bonus_max 默认值: None (函数内部设定)")

print("\n" + "="*60)
print("✓ 所有基础验证通过！可以运行实验")
print("="*60)
