# This script is a shared template for LangGraph workflow.
# LangGrah runs as a sequence of steps (nodes), select, validate and 
# each node reads that the previous node produced, so it can save its result.
# RoutingState is the shape of this shared data.

from typing import Dict, List, Tuple, TypedDict

import networkx as nx

Node = str
FlowKey = Tuple[Node, Node]

class RoutingState(TypedDict, total=False):
    demands: Dict[FlowKey, float]
    graph: nx.Graph
    candidates: Dict[FlowKey, List[List[Node]]]
    selected_paths: Dict[FlowKey, List[Node]]
    validated_paths: Dict[FlowKey, List[Node]]
    invalid_flows: List[FlowKey]
