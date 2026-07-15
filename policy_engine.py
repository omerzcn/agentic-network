from typing import Dict, List, Tuple

import networkx as nx

from simulation import ControllerAPI, TopologySnapshot

from candidate_generator import CandidateGenerator

Node = str
FlowKey = Tuple[Node, Node]

class DeterministicPolicyEngine:
    """Simple routing policy to return one path for each active traffic demand."""

    def __init__(self, ctrl: ControllerAPI, candidates_per_flow: int = 3):
        self.ctrl = ctrl
        self.candidate_generator = CandidateGenerator(
            candidates_per_flow=candidates_per_flow
        )

    def route_flows(
        self,
        demands: Dict[FlowKey, float],
    ) -> Dict[FlowKey, List[Node]]:
        topology = self.ctrl.get_topology_snapshot()
        graph = self._build_graph(topology)

        paths: Dict[FlowKey, List[Node]] = {}

        for (src, dst), demand_mbps in demands.items():
            if demand_mbps <= 0:
                continue

            candidates = self.candidate_generator.generate(
                graph,
                src,
                dst,
            )

            #Debugging
            #print(f"[CandidateGenerator] " + src + "->" + dst + ": " + str(candidates))

            if candidates:
                paths[(src, dst)] = candidates[0]

        return paths

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

    @staticmethod
    def _shortest_path(
        graph: nx.Graph,
        src: Node,
        dst: Node,
    ) -> List[Node]:
        if src not in graph or dst not in graph:
            return []

        try:
            return nx.shortest_path(
                graph,
                src,
                dst,
                weight="weight",
            )
        except nx.NetworkXNoPath:
            return []
