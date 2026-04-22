from collections import deque

import numpy as np

from core.microservice_dags import MICROSERVICE_DAGS


def get_entry_nodes(dag_info):
    """Return nodes with in-degree == 0 (DAG entry points)."""
    all_nodes = set(dag_info['nodes'].keys())
    nodes_with_incoming = set()
    for (src, dst) in dag_info['edges'].keys():
        nodes_with_incoming.add(dst)
    return list(all_nodes - nodes_with_incoming)


def topological_sort(dag_info):
    """BFS topological sort (Kahn's algorithm)."""
    all_nodes = list(dag_info['nodes'].keys())
    in_degree = {n: 0 for n in all_nodes}
    adj = {n: [] for n in all_nodes}
    for (src, dst) in dag_info['edges'].keys():
        adj[src].append(dst)
        in_degree[dst] += 1

    queue = deque([n for n in all_nodes if in_degree[n] == 0])
    sorted_nodes = []
    while queue:
        node = queue.popleft()
        sorted_nodes.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return sorted_nodes


def assign_dag_type():
    """Randomly assign a DAG type based on configured probabilities."""
    dag_names = list(MICROSERVICE_DAGS.keys())
    dag_probs = [MICROSERVICE_DAGS[d]['probability'] for d in dag_names]
    return np.random.choice(dag_names, p=dag_probs)


def initialize_dag_assignment(dag_type, nearest_server_id):
    """Deploy all microservice nodes of a DAG onto a single server."""
    dag_info = MICROSERVICE_DAGS[dag_type]
    return {node: nearest_server_id for node in dag_info['nodes']}
