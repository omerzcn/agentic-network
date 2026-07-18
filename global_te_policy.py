# This script is another deterministic routing policy for traffic engineering that fixes a problem:
# every other policy in this project checks a candidate path against a link's total capacity,
# so two different flows can both be told "yes, this path fits" for the same link,
# and only clash later when the simulator actually installs them.
# This policy instead tracks how much capacity is actually left after every flow it assigns,
# so each following flow sees an up-to-date view of the network, not the full capacity.

# Working Steps:
#   1. Give every flow its direct link first, if that link has enough capacity left
#      and is fast enough. Direct links use the least shared capacity, so grabbing them first
#      avoids wasting network-wide capacity on detours where a direct link would have worked fine.
#   2. For everything left over, handle the biggest demands first, generating fresh candidate paths
#      against whatever capacity actually remains, and reserving the capacity used by every pick immediately
#      so the next flow sees it as unavailable.
#   3. An already-installed path is offered as one candidate among others (not an automatic free pass),
#      so a flow keeps its existing route when it's still good, but isn't shielded from the same capacity
#      check everyone else goes through.

from typing import Dict, List, Optional, Tuple

from deterministic_selector import DeterministicPathSelector
from policy_engine import BaseCandidatePolicy
from simulation import ControllerAPI, FlowId, TrafficModel

Node = str
FlowKey = Tuple[Node, Node]

class DeterministicGlobalTE(BaseCandidatePolicy):
    def __init__(self, ctrl: ControllerAPI, candidates_per_flow: int = 8, traffic: Optional[TrafficModel] = None):
        super().__init__(ctrl, candidates_per_flow, traffic)
        self.selector = DeterministicPathSelector()

    def route_flows(self, demands: Dict[FlowKey, float]) -> Dict[FlowKey, List[Node]]:
        residual = self._residual_capacities()
        active = {flow: demand for flow, demand in demands.items() if demand > 0}

        assigned: Dict[FlowKey, List[Node]] = {}
        remaining: Dict[FlowKey, float] = {}

        # Step 1: reserve direct 1-hop links first.
        for flow, demand in active.items():
            src, dst = flow
            link = (src, dst)
            if (
                link in residual
                and residual[link] >= demand
                and self.ctrl.g.links[link]["latency"] <= self._latency_budget(src, dst)
            ):
                assigned[flow] = [src, dst]
                residual[link] -= demand
            else:
                remaining[flow] = demand

        # Step 2 and 3: largest demand first, always against the current residual view.
        for flow, demand in sorted(remaining.items(), key=lambda kv: -kv[1]):
            src, dst = flow
            graph = self._residual_graph(residual)

            candidates = []
            existing = self._existing_path(src, dst)
            if existing:
                candidates.append(existing)
            candidates += self.candidate_generator.generate(graph, src, dst, weight_attr="latency_ms")

            selected = self.selector.select(
                graph=graph,
                candidates=candidates,
                demand_mbps=demand,
                delay_budget_ms=self._latency_budget(src, dst),
            )
            if selected:
                assigned[flow] = selected
                self._reserve(residual, selected, demand)

        # Debugging: flows of this policy couldn't find a feasible candidate.
        # These get silently rescued by the simulator's heuristic fallback
        unrouted = len(active) - len(assigned)
        if unrouted:
            print(f"[DeterministicGlobalTE] {unrouted}/{len(active)} active flows had no "
                  f"feasible candidate among the {self.candidate_generator.candidates_per_flow} tried")

        return assigned

    def _existing_path(self, src: Node, dst: Node) -> Optional[List[Node]]:
        entry = self.ctrl.get_flow_table_snapshot().flows.get(FlowId(src=src, dst=dst))
        if entry is not None and entry.path and self.ctrl.validate_path_logic(src, dst, entry.path):
            return list(entry.path)
        return None

    def get_stats(self) -> dict:
        # No LLM calls happen in this policy. That's why, for interface compatibility with
        # RoutingAgent, these are saved to summary.json so that code works unchanged.
        return {"llm_calls": 0, "total_tokens": 0, "average_call_latency_s": 0.0}
