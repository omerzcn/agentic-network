# This is the LangGraph pipeline for the LLM-driven agent. It now shares the
# same residual-capacity tracking as DeterministicGlobalTE (global_te_policy.py):
# Instead of checking a candidate path against a link's total capacity, every
# step here checks against what's actually left after earlier steps' picks in
# this same decision, so the LLM is choosing based on real numbers instead of
# a static, already-out-of-date view of the network.

# Steps:
#   1. perceive             : snapshot how much capacity is really left on every link.
#   2. reuse                : keep already-installed paths that are still good, and reserve their capacity.
#   3. assign_direct        : give every remaining flow its direct link, if it still fits and is fast enough
#                             (same idea as DeterministicGlobalTE).
#   4. generate_candidates  : build path options for what's left, using the residual (not total) capacity view.
#   5. select               : ask the LLM to pick one candidate per remaining flow.
#   6. reconcile            : go through the LLM's picks largest-demand-first, checking each one against
#                             a running residual tracker. If a pick no longer fits (because an earlier, bigger flow
#                             already used that capacity), fall back to another one of the same candidates
#                             instead of dropping the flow. This also covers flows the LLM didn't return an answer.

from typing import Dict, List, Tuple

from langgraph.graph import END, START, StateGraph

from agent_state import RoutingState
from config import DELAY_BUDGET_MS
from deterministic_selector import DeterministicPathSelector
from llm_client import LLMClient
from llm_path_selector import LLMPathSelector
from path_tools import path_is_feasible
from policy_engine import BaseCandidatePolicy

Node = str
FlowKey = Tuple[Node, Node]

class RoutingAgent(BaseCandidatePolicy):
    def __init__(self, ctrl, candidates_per_flow: int = 3):
        super().__init__(ctrl, candidates_per_flow)

        self.llm_client = LLMClient()
        self.llm_selector = LLMPathSelector(self.llm_client)
        self.deterministic_selector = DeterministicPathSelector()
        self.graph = self._build_workflow()

    def _build_workflow(self):
        builder = StateGraph(RoutingState)

        builder.add_node("perceive", self._perceive)
        builder.add_node("reuse", self._reuse)
        builder.add_node("assign_direct", self._assign_direct)
        builder.add_node("generate_candidates", self._generate_candidates_node)
        builder.add_node("select", self._select)
        builder.add_node("reconcile", self._reconcile)

        builder.add_edge(START, "perceive")
        builder.add_edge("perceive", "reuse")
        builder.add_edge("reuse", "assign_direct")
        builder.add_edge("assign_direct", "generate_candidates")
        builder.add_edge("generate_candidates", "select")
        builder.add_edge("select", "reconcile")
        builder.add_edge("reconcile", END)

        return builder.compile()

    def route_flows(self, demands: Dict[FlowKey, float]) -> Dict[FlowKey, List[Node]]:
        result = self.graph.invoke(
            {
                "demands": demands,
            }
        )

        paths = {
            **result.get("reused_paths", {}),
            **result.get("direct_paths", {}),
            **result.get("validated_paths", {}),
        }

        # Final safety net: re-check every path against the live network
        # right before handing it to the simulator.
        safe_paths = {}
        rejected = 0

        for flow, path in paths.items():
            src, dst = flow
            demand = demands[flow]

            if not self.ctrl.validate_path_logic(src, dst, path):
                rejected += 1
                continue

            if not self.ctrl.validate_path_sla(path, demand):
                rejected += 1
                continue

            safe_paths[flow] = path

        if rejected:
            print(f"[RoutingAgent] final safety check: {rejected}/{len(paths)} paths "
                  f"rejected right before returning (network moved on since they were picked)")

        return safe_paths

    def _perceive(self, state: RoutingState) -> dict:
        residual = self._residual_capacities()
        return {
            "residual": residual,
        }

    def _reuse(self, state: RoutingState) -> dict:
        residual = dict(state["residual"])
        graph = self._residual_graph(residual)

        reused, remaining = self._reuse_existing_paths(
            graph=graph,
            demands=state["demands"],
        )
        for flow, path in reused.items():
            self._reserve(residual, path, state["demands"][flow])

        print(f"[RoutingAgent] reuse: {len(reused)} reused, {len(remaining)} remaining")

        return {
            "reused_paths": reused,
            "remaining_demands": remaining,
            "residual": residual,
        }

    def _assign_direct(self, state: RoutingState) -> dict:
        residual = dict(state["residual"])
        assigned: Dict[FlowKey, List[Node]] = {}
        remaining: Dict[FlowKey, float] = {}

        for flow, demand in state["remaining_demands"].items():
            src, dst = flow
            link = (src, dst)
            if (
                link in residual
                and residual[link] >= demand
                and self.ctrl.g.links[link]["latency"] <= DELAY_BUDGET_MS
            ):
                assigned[flow] = [src, dst]
                residual[link] -= demand
            else:
                remaining[flow] = demand

        print(f"[RoutingAgent] direct: {len(assigned)} direct-assigned, {len(remaining)} sent to LLM")

        return {
            "direct_paths": assigned,
            "remaining_demands": remaining,
            "residual": residual,
        }

    def _generate_candidates_node(self, state: RoutingState) -> dict:
        graph = self._residual_graph(state["residual"])
        candidates = self._generate_all_candidates(
            graph=graph,
            demands=state["remaining_demands"],
        )
        return {
            "candidates": candidates,
            "graph": graph,
        }

    def _select(self, state: RoutingState) -> dict:
        selected_paths = self.llm_selector.select(
            graph=state["graph"],
            candidates_by_flow=state["candidates"],
            demands=state["remaining_demands"],
            delay_budget_ms=DELAY_BUDGET_MS,
        )
        return {
            "selected_paths": selected_paths,
        }

    def _reconcile(self, state: RoutingState) -> dict:
        residual = dict(state["residual"])
        demands = state["demands"]
        candidates_by_flow = state["candidates"]

        validated: Dict[FlowKey, List[Node]] = {}
        kept_llm_pick = 0

        # Largest demand first: Every flow that got candidates goes through here,
        # whether or not the LLM actually returned a pick for it.
        order = sorted(candidates_by_flow.keys(), key=lambda flow: -demands[flow])

        for flow in order:
            demand = demands[flow]
            # Rebuilt every iteration: otherwise every flow would check
            # against the same snapshot instead of what earlier
            # flows in this same loop already reserved.
            graph = self._residual_graph(residual)
            llm_pick = state["selected_paths"].get(flow)

            # Trust the LLM's own choice if it's still capacity-feasible:
            # Only fall back to the deterministic selector (picking among the other candidates)
            # when the LLM's pick no longer fits or it didn't return one for this flow.
            if llm_pick and path_is_feasible(graph, llm_pick, demand):
                selected = llm_pick
                kept_llm_pick += 1
            else:
                selected = self.deterministic_selector.select(
                    graph=graph,
                    candidates=candidates_by_flow[flow],
                    demand_mbps=demand,
                    delay_budget_ms=DELAY_BUDGET_MS,
                )

            if selected:
                validated[flow] = selected
                self._reserve(residual, selected, demand)

        print(f"[RoutingAgent] reconcile: {len(validated)}/{len(candidates_by_flow)} flows routed "
              f"({kept_llm_pick} kept the LLM's own pick)")

        return {
            "validated_paths": validated,
        }

    def get_stats(self):
        return self.llm_client.get_stats()
