# This script picks one path from candidates for a flow.
# Then, it drops paths that don't have enough capacity for the demand.
# Also, it picks the lowest-latency path among those within the delay budget
# or among all feasible paths, if none of that fit the budget.

from typing import List

import networkx as nx

from path_tools import path_is_feasible, path_latency_ms

Node = str

class DeterministicPathSelector:
    """Chooses the lowest latency feasible candidate"""
    def select(
            self,
            graph: nx.Graph,
            candidates: List[List[Node]],
            demand_mbps: float,
            delay_budget_ms: float,
    ) -> List[Node]:
        feasible_candidates = []
        for path in candidates:
            if not path_is_feasible(graph, path, demand_mbps):
                continue
            latency = path_latency_ms(graph, path)

            feasible_candidates.append(
                {
                    "path": path,
                    "latency_ms": latency,
                    "within_delay_budget": (
                        latency <= delay_budget_ms
                    ),
                    "hop_count": len(path) - 1,
                }
            )
        
        if not feasible_candidates:
            return []
        
        within_budget = [item for item in feasible_candidates if item["within_delay_budget"]]
        available = (within_budget if within_budget else feasible_candidates)
        selected = min(
            available,
            key=lambda item: (
                item["latency_ms"],
                item["hop_count"],
            ),
        )

        return list(selected["path"])
