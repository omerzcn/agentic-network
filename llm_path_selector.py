# This script builds a prompt for each flow listing candidate paths with their
# precomputed latency and capacity. Then, it asks the LLM to pick one by index per
# flow, then maps the returned indices back to actual node paths.

from typing import Dict, List, Tuple

import networkx as nx

from llm_client import LLMClient
from path_tools import path_bottleneck_capacity_mbps, path_latency_ms

Node = str
FlowKey = Tuple[Node, Node]

class LLMPathSelector:
    def __init__(self, client: LLMClient):
        self.client = client
    
    def select(
            self,
            graph: nx.Graph,
            candidates_by_flow: Dict[FlowKey, List[List[Node]]],
            demands: Dict[FlowKey, float],
            delay_budget_ms: float,
    ) -> Dict[FlowKey, List[Node]]:
        if not candidates_by_flow:      # Shortcut: If there is nothing to ask, no API call
            return {}

        prompt = self._build_prompt(
            graph=graph,
            candidates_by_flow=candidates_by_flow,
            demands=demands,
            delay_budget_ms=delay_budget_ms,
        )
        selections = self.client.select_paths(prompt)

        if selections is None:
            return {}
        
        paths: Dict[FlowKey, List[Node]] = {}
        for flow, candidates in (candidates_by_flow.items()):
            src, dst = flow
            label = src + "->" + dst

            selected_index = selections.get(label) 
            
            if not isinstance(selected_index, int):
                continue

            if not 0 <= selected_index < len(candidates):
                continue

            paths[flow] = candidates[selected_index]
        
        return paths
    
    @staticmethod
    def _build_prompt(
        graph: nx.Graph,
        candidates_by_flow: Dict[FlowKey, List[List[Node]]],
        demands: Dict[FlowKey, float],
        delay_budget_ms: float,
    ) -> str:
        lines = [
            (
                "You are selecting network routes. "
                "For each flow, choose exactly one candidate "
                "path by integer index."
            ),
            (
                "Prefer a path that can carry the demand and "
                "stays within the delay budget."
            ),
            (
                "If multiple paths satisfy both conditions, "
                "choose the one with the lowest latency."
            ),
            (
                "Return only a valid JSON object such as "
                '{"A->B": 0, "C->D": 1}.'
            ),
            (
                "Do not return explanations, markdown, "
                "or paths."
            ),
            (
                "Delay budget: "
                + str(delay_budget_ms)
                + " ms"
            ),
        ]
        for flow, candidates in (candidates_by_flow.items()):
            src, dst = flow
            demand = demands[flow]

            lines.append("")
            lines.append(
                "Flow "
                + src
                + "->"
                + dst
                + ", demand="
                + str(round(demand, 2))
                + " Mbps"
            )

            for index, path in enumerate(candidates):
                latency = path_latency_ms(graph, path)
                bottleneck = path_bottleneck_capacity_mbps(graph, path)
                lines.append(
                    "["
                    + str(index)
                    + "] "
                    + " -> ".join(path)
                    + " | latency="
                    + str(round(latency, 2))
                    + " ms"
                    + " | bottleneck="
                    + str(round(bottleneck, 2))
                    + " Mbps"
                )
        return "\n".join(lines)
    
