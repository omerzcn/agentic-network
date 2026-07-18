# This script builds a routable graph and candidate paths per flow, shared by both
# policy engines below. DeterministicPolicyEngine picks a path with fixed
# rules and LLMPolicyEngine asks an LLM to pick instead.

from typing import Dict, List, Tuple

import networkx as nx

from simulation import ControllerAPI, FlowId, TopologySnapshot

from candidate_generator import CandidateGenerator

from config import DELAY_BUDGET_MS
from deterministic_selector import DeterministicPathSelector
from path_tools import path_is_feasible

from llm_client import LLMClient
from llm_path_selector import LLMPathSelector

Node = str
FlowKey = Tuple[Node, Node]
Link = Tuple[Node, Node]

class BaseCandidatePolicy:
    def __init__(self, ctrl: ControllerAPI, candidates_per_flow: int = 3):
        self.ctrl = ctrl
        self.candidate_generator = CandidateGenerator(candidates_per_flow=candidates_per_flow)

    @staticmethod
    def _build_graph(topology: TopologySnapshot) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(topology.nodes)

        for link_id, attrs in topology.links.items():
            if not attrs.up:
                continue

            graph.add_edge(
                link_id.u,
                link_id.v,
                weight=attrs.weight,
                capacity_mbps=attrs.capacity_mbps,
                latency_ms=attrs.latency_ms,
                loss=attrs.loss,
            )

        return graph

    def _residual_capacities(self) -> Dict[Link, float]:
        # How much capacity is actually left per direction, not the link's total capacity.
        # This is what makes a policy capacity-aware
        return {
            link: data["capacity"] - data["util"]
            for link, data in self.ctrl.g.links.items()
            if data["up"]
        }

    def _residual_graph(self, residual: Dict[Link, float]) -> nx.DiGraph:
        # Every up link gets an edge, so a saturated link is correctly treated as infeasible
        # (by path_is_feasible) instead of crashing.
        graph = nx.DiGraph()
        graph.add_nodes_from(self.ctrl.g.nodes)

        for (u, v), left in residual.items():
            graph.add_edge(
                u, v,
                weight=self.ctrl.g.links[(u, v)]["weight"],
                latency_ms=self.ctrl.g.links[(u, v)]["latency"],
                capacity_mbps=left,
            )
        return graph

    @staticmethod
    def _reserve(residual: Dict[Link, float], path: List[Node], demand: float) -> None:
        for i in range(len(path) - 1):
            residual[(path[i], path[i + 1])] -= demand

    def _generate_all_candidates(self, graph: nx.Graph, demands: Dict[FlowKey, float]) -> Dict[FlowKey, List[List[Node]]]:
        candidates_by_flow = {}
        no_candidates = []

        for flow, demand in demands.items():
            if demand <= 0:
                continue

            src, dst = flow
            candidates = (
                self.candidate_generator.generate(graph, src, dst)
            )
            if candidates:
                candidates_by_flow[flow] = candidates
            else:
                no_candidates.append(flow)

        # Debugging: flows that never even reach the LLM, because the live
        # graph snapshot had zero paths between src and dst at perceive-time
        if no_candidates:
            print(f"[BaseCandidatePolicy] {len(no_candidates)}/{sum(1 for d in demands.values() if d > 0)} "
                  f"active flows had zero candidates in the live graph: {no_candidates}")

        return candidates_by_flow

    def _reuse_existing_paths(self, graph: nx.Graph, demands: Dict[FlowKey, float]) -> Tuple[Dict[FlowKey, List[Node]], Dict[FlowKey, float]]:
        flow_table = self.ctrl.get_flow_table_snapshot().flows

        reused: Dict[FlowKey, List[Node]] = {}
        remaining: Dict[FlowKey, float] = {}

        for flow, demand in demands.items():
            if demand <= 0:
                continue

            src, dst = flow
            entry = flow_table.get(FlowId(src=src, dst=dst))

            if entry is not None and entry.path:
                still_valid = self.ctrl.validate_path_logic(src, dst, entry.path)
                if still_valid and path_is_feasible(graph, entry.path, demand):
                    reused[flow] = list(entry.path)
                    continue

            remaining[flow] = demand

        return reused, remaining

class DeterministicPolicyEngine(BaseCandidatePolicy):
    def __init__(self, ctrl: ControllerAPI, candidates_per_flow: int = 3):
        super().__init__(ctrl, candidates_per_flow)
        self.selector = DeterministicPathSelector()

    def route_flows(self, demands: Dict[FlowKey, float]) -> Dict[FlowKey, List[Node]]:
        topology = self.ctrl.get_topology_snapshot()
        graph = self._build_graph(topology)

        candidates_by_flow = (
            self._generate_all_candidates(graph, demands)
        )

        selected_paths = {}

        for flow, candidates in (candidates_by_flow.items()):
            selected = self.selector.select(
                graph=graph,
                candidates=candidates,
                demand_mbps=demands[flow],
                delay_budget_ms=DELAY_BUDGET_MS,
            )
            if selected:
                selected_paths[flow] = selected

        return selected_paths

class LLMPolicyEngine(BaseCandidatePolicy):
    def __init__(self, ctrl: ControllerAPI, candidates_per_flow = 3):
        super().__init__(ctrl, candidates_per_flow)
        self.llm_client = LLMClient()
        self.selector = LLMPathSelector(self.llm_client)
    
    def route_flows(self, demands: Dict[FlowKey, float]) -> Dict[FlowKey, List[Node]]:
        topology = self.ctrl.get_topology_snapshot()
        graph = self._build_graph(topology)

        candidates_by_flow = (
            self._generate_all_candidates(graph, demands)
        )
        selected = self.selector.select(
                graph=graph,
                candidates_by_flow=candidates_by_flow,
                demands=demands,
                delay_budget_ms=DELAY_BUDGET_MS,
            )

        # Debugging: how many eligible flows did the LLM actually route?
        print(f"[LLMPolicyEngine] {len(selected)}/{len(candidates_by_flow)} flows routed by LLM")

        return selected
    
    def get_stats(self) -> dict:
        return self.llm_client.get_stats()
