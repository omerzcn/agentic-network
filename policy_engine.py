# This script builds a routable graph and candidate paths per flow, shared by the
# routing policies (RoutingAgent in agents.py, DeterministicGlobalTE in
# global_te_policy.py) built on top of BaseCandidatePolicy.

from typing import Dict, List, Optional, Tuple

import networkx as nx

from simulation import ControllerAPI, FlowId, TopologySnapshot, TrafficModel

from candidate_generator import CandidateGenerator

from path_tools import path_is_feasible, path_latency_ms

Node = str
FlowKey = Tuple[Node, Node]
Link = Tuple[Node, Node]

class BaseCandidatePolicy:
    def __init__(self, ctrl: ControllerAPI, candidates_per_flow: int = 3, traffic: Optional[TrafficModel] = None):
        self.ctrl = ctrl
        self.traffic = traffic
        self.candidate_generator = CandidateGenerator(candidates_per_flow=candidates_per_flow)

    def _latency_budget(self, src: Node, dst: Node) -> float:
        # Per-flow delay budget, replacing the old single DELAY_BUDGET_MS constant.
        if self.traffic is None:
            return float("inf")
        budget = self.traffic.get_max_latency_ms(src, dst)
        return budget if budget is not None else float("inf")

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
                self.candidate_generator.generate(graph, src, dst, weight_attr="latency_ms")
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
                within_budget = path_latency_ms(graph, entry.path) <= self._latency_budget(src, dst) + 1e-9
                if still_valid and within_budget and path_is_feasible(graph, entry.path, demand):
                    reused[flow] = list(entry.path)
                    continue

            remaining[flow] = demand

        return reused, remaining
