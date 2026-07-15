# This script chooses a candidate using deterministic network calculations
# I will use this to compare LLMs later

from typing import List

import networkx as nx

Node = str

def path_latency_ms(graph: nx.Graph, path: List[Node]) -> float:
    total = 0.0
    for index in range(len(path) - 1):
        u = path[index]
        v = path[index + 1]
        total += float(graph[u][v]["latency_ms"])
    return total

def path_bottleneck_capacity_mbps(graph: nx.Graph, path: List[Node]) -> float:
    if len(path) < 2:
        return 0.0
    
    capacities = []

    for index in range(len(path) - 1):
        u = path[index]
        v = path[index + 1]
        capacities.append(
            float(graph[u][v]["capacity_mbps"])
        )
    return min(capacities)

def path_is_feasible(graph: nx.Graph, path: List[Node], demand_mbps: float) -> bool:
    if not path:
        return False
    
    bottleneck = path_bottleneck_capacity_mbps(graph, path)
    return demand_mbps <= bottleneck
