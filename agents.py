# Runs the same steps as LLMPolicyEngine (perceive, generate candidates,
# ask the LLM to select), but as an explicit LangGraph pipeline, plus a
# validate step that checks the LLM's picks before they're used.

from typing import Dict, List, Tuple

from langgraph.graph import END, START, StateGraph

from agent_state import RoutingState
from candidate_generator import CandidateGenerator
from config import DELAY_BUDGET_MS
from llm_client import LLMClient
from llm_path_selector import LLMPathSelector
from path_tools import path_is_feasible
from policy_engine import LLMPolicyEngine, BaseCandidatePolicy

from deterministic_selector import DeterministicPathSelector

Node = str
FlowKey = Tuple[Node, Node]

class RoutingAgent(BaseCandidatePolicy):
    def __init__(self, ctrl, candidates_per_flow: int = 3):
        super().__init__(ctrl, candidates_per_flow)

        self.llm_client = LLMClient()
        self.llm_selector = LLMPathSelector(self.llm_client)
        self.graph = self._build_workflow()
        self.deterministic_selector = DeterministicPathSelector()

    def _repair(self, state: RoutingState) -> dict:
        repaired = dict(state.get("validated_paths", {}))
        invalid_flows = state.get("invalid_flows", [])
        rescued = 0

        for flow in invalid_flows:
            candidates = state["candidates"].get(flow, [])
            selected = (
                self.deterministic_selector.select(
                graph=state["graph"],
                candidates=candidates,
                demand_mbps=state["demands"][flow],
                delay_budget_ms=DELAY_BUDGET_MS,
                )
            )
            if selected:
                repaired[flow] = selected
                rescued += 1

        # Debugging: how many invalid flows did the deterministic fallback rescue
        print(f"[RoutingAgent] repair: {rescued}/{len(invalid_flows)} invalid flows rescued")

        return {
            "validated_paths": repaired,
        }
    
    def _build_workflow(self):
        builder = StateGraph(RoutingState)

        builder.add_node("repair", self._repair)
        builder.add_node("perceive", self._perceive)
        builder.add_node("reuse", self._reuse)
        builder.add_node("generate_candidates", self._generate_candidates_node)
        builder.add_node("select", self._select)
        builder.add_node("validate", self._validate)
        builder.add_edge(START, "perceive")
        builder.add_edge("perceive", "reuse")
        builder.add_edge("reuse", "generate_candidates")
        builder.add_edge("generate_candidates", "select")
        builder.add_edge("select", "validate")
        builder.add_conditional_edges(
            "validate",
            self._after_validation,
            {
                "repair": "repair",
                "finish": END,
            },
        )
        builder.add_edge("repair", END)

        return builder.compile()

    def route_flows(self, demands: Dict[FlowKey, float]) -> Dict[FlowKey, List[Node]]:
        result = self.graph.invoke(
            {
                "demands": demands,
            }
        )
        return {
            **result.get("reused_paths", {}),
            **result.get("validated_paths", {}),
        }

    def _perceive(self, state: RoutingState) -> dict:
        topology = self.ctrl.get_topology_snapshot()
        graph = self._build_graph(topology)
        return {
            "graph": graph,
        }

    def _reuse(self, state: RoutingState) -> dict:
        reused, remaining = self._reuse_existing_paths(
            graph=state["graph"],
            demands=state["demands"],
        )

        # Debugging: how many flows were reused vs sent on to the LLM
        print(f"[RoutingAgent] reuse: {len(reused)} reused, {len(remaining)} sent to LLM")

        return {
            "reused_paths": reused,
            "remaining_demands": remaining,
        }

    def _generate_candidates_node(self, state: RoutingState) -> dict:
        candidates = self._generate_all_candidates(
            graph=state["graph"],
            demands=state["remaining_demands"],
        )
        return {
            "candidates": candidates,
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
    
    def _validate(self, state: RoutingState) -> dict:
        validated = {}
        invalid_flows = []
        logic_fail = 0
        capacity_fail = 0

        for flow, path in state["selected_paths"].items():
            src, dst = flow
            demand = state["demands"][flow]

            logic_ok = (self.ctrl.validate_path_logic(src, dst, path))

            capacity_ok = path_is_feasible(state["graph"], path, demand)

            if not logic_ok:
                logic_fail += 1
            if not capacity_ok:
                capacity_fail += 1

            if logic_ok and capacity_ok:
                validated[flow] = path
            else:
                invalid_flows.append(flow)

        # Debugging: dividing invalid reasons for stale and live link mismatches against capacity shortfall 
        print(
            f"[RoutingAgent] validate: {len(validated)} ok, {len(invalid_flows)} invalid "
            f"(failed logic/live-link: {logic_fail}, failed capacity: {capacity_fail})"
        )

        return {
            "validated_paths": validated,
            "invalid_flows": invalid_flows,
        }

    def _after_validation(self, state: RoutingState) -> str:
        if state.get("invalid_flows"):
            return "repair"
        return "finish"
    
    def get_stats(self):
        return self.llm_client.get_stats()
