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
            delay_budget_ms: Dict[FlowKey, float],
    ) -> Dict[FlowKey, List[Node]]:
        if not candidates_by_flow:      # Shortcut: If there is nothing to ask, no API call
            return {}

        prompt = self._build_prompt(
            graph=graph,
            candidates_by_flow=candidates_by_flow,
            demands=demands,
            delay_budget_ms=delay_budget_ms,
        )
        schema = self._build_schema(candidates_by_flow)
        selections = self.client.select_paths(prompt, schema=schema)

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

        # Debugging: flows the LLM never returned a usable pick for at all.
        missing = set(candidates_by_flow.keys()) - set(paths.keys())
        if missing:
            print(f"[LLMPathSelector] {len(missing)}/{len(candidates_by_flow)} flows got no usable "
                  f"selection from the LLM (dropped silently, never reach repair): {missing}")

        return paths

    @staticmethod
    def _build_schema(candidates_by_flow: Dict[FlowKey, List[List[Node]]]) -> dict:
        # A JSON Schema with every flow marked required. Ollama cannot finish
        # generating without producing a value for every required key
        properties = {}
        required = []

        for flow, candidates in candidates_by_flow.items():
            label = flow[0] + "->" + flow[1]
            properties[label] = {
                "type": "integer",
                "minimum": 0,
                "maximum": len(candidates) - 1,
            }
            required.append(label)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    @staticmethod
    def _build_prompt(
        graph: nx.Graph,
        candidates_by_flow: Dict[FlowKey, List[List[Node]]],
        demands: Dict[FlowKey, float],
        delay_budget_ms: Dict[FlowKey, float],
    ) -> str:
        lines = [
            (
                "You are selecting network routes. "
                "For each flow, choose exactly one candidate "
                "path by integer index."
            ),
            (
                "The 'available' number for each path is the capacity "
                "actually left on its tightest link right now, after "
                "other flows' current usage, not the link's total capacity."
            ),
            (
                "Each flow below has its own delay budget, they are not "
                "all the same. Prefer a path that can carry the demand and "
                "stays within that flow's own delay budget shown below."
            ),
            (
                "If multiple paths satisfy both conditions, "
                "choose the one with the lowest latency."
            ),
            (
                "Return only a valid JSON object such as "
                '{"A->B": 0, "C->D": 1}. '
            ),
            (
                "Do not return explanations, markdown, "
                "or paths."
            ),
        ]
        for flow, candidates in (candidates_by_flow.items()):
            src, dst = flow
            demand = demands[flow]
            budget = delay_budget_ms.get(flow, float("inf"))
            budget_str = "no limit" if budget == float("inf") else f"{round(budget, 2)} ms"

            lines.append("")
            lines.append(
                "Flow "
                + src
                + "->"
                + dst
                + ", demand="
                + str(round(demand, 2))
                + " Mbps, delay budget="
                + budget_str
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
                    + " | available="
                    + str(round(bottleneck, 2))
                    + " Mbps"
                )
        return "\n".join(lines)
    
