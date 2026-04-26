# 微服务迁移算法 — 数据来源说明

本项目的训练和实验数据主要来自**两个数据集**：

---

## 1. 出租车轨迹数据 — T-Drive 数据集

**来源**：[微软亚洲研究院 T-Drive 项目](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/)

| 属性 | 说明 |
|------|------|
| **地理范围** | 北京市 |
| **时间跨度** | 2008年2月 |
| **数据规模** | ~1,048,575 条 GPS 记录 |
| **出租车数量** | 约 10,000+ 辆（实验中采样 9 辆） |
| **采样间隔** | 约 10 分钟 |
| **坐标范围** | 经度 116.1°~116.8°, 纬度 39.7°~40.1° |

**原始数据格式** (`data/taxi_raw/*.txt`)：

```
taxi_id, timestamp, longitude, latitude
1, 2008-02-02 15:36:08, 116.51172, 39.92123
```

**用途**：模拟移动用户的真实轨迹，驱动微服务迁移决策

---

## 2. 边缘服务器位置数据

**文件**：`data/edge_server_locations.csv`

| 属性 | 说明 |
|------|------|
| **服务器数量** | 600 个边缘节点 |
| **地理覆盖** | 与北京出租车轨迹范围对齐 |
| **数据字段** | `edge_server_id`, `Name`, `longitude`, `latitude` |

**示例**：

```csv
edge_server_id,Name,longitude,latitude
10000002,Fort Hill Wharf DARWIN,116.36833,39.90514
10000004,Optus Tower,116.4545278,39.95381622
```

**用途**：作为微服务部署的候选边缘节点

---

## 3. 实验配置

在 `run_comparison.py` 中实际使用的数据配置：

```python
CHUNK_SIZE = 10000  # 截取前 10,000 条轨迹记录用于实验
```

**实验规模**：

| 参数 | 数值 |
|------|------|
| GPS 记录数 | 10,000 条 |
| 出租车数量 | 9 辆 |
| 边缘服务器数量 | 600 个 |
| 时间步数量 | 6,550 个 |

---

## 4. 数据预处理流程

```
taxi_raw/*.txt (原始 GPS)
       ↓
taxi_with_health_info.csv (合并后的完整数据)
       ↓
data_loader.py 只保留 [taxi_id, date_time, latitude, longitude]
       ↓
截取前 10,000 条用于训练/实验
```

**注**：`beijing_data.csv` 中的健康数据（Age, CVD_Risk 等）在当前迁移算法中**未被使用**，仅保留了纯物理轨迹信息。

---

## 5. 数据文件清单

| 文件路径 | 说明 |
|----------|------|
| `data/taxi_raw/*.txt` | 原始出租车 GPS 轨迹（T-Drive） |
| `data/taxi_with_health_info.csv` | 合并后的轨迹数据（~100万条） |
| `data/edge_server_locations.csv` | 600 个边缘服务器位置 |
| `data/beijing_data.csv` | 健康数据（未使用） |
| `data/taxi_points_with_edge_servers_.png` | 数据可视化图 |
