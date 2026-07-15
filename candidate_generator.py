from typing import List

import networkx as nx

Node = str

class CandidateGenerator:
    def __init__(self, candidates_per_flow: int= 3):
        if candidates_per_flow < 1:
            raise ValueError("candidates_per_flow must be at least 1")
        
        self.candidates_per_flow = candidates_per_flow

    def generate(self, graph: nx.Graph, src: Node, dst: Node) -> List[List[Node]]:
        if src not in graph or dst not in graph:
            return []
        
        candidates: List[List[Node]] = []

        try:
            path_iterator = nx.shortest_simple_paths(
                graph,
                src,
                dst,
                weight="weight",
            )
            for path in path_iterator:
                candidates.append(list(path))

                if len(candidates) >= self.candidates_per_flow:
                    break
        
        except nx.NetworkXNoPath:
            return []

        return candidates
