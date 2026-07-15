from typing import Any, Dict

def compute_sla_metrics(ctrl, metrics: Dict[str, Any], delay_budget_ms: float) -> Dict[str, Any]:
    """SLA Violations:
    Bandwidth -> A demand was dropped specifically for that reason, not flow is over-capacity.
    Delay -> An installed path whose total link latency exceeds the configured budget.
    """

    bandwidth_violating_flow_list = [
        {"src": d["src"], "dst": d["dst"], "demand": d["demand"], "path": " -> ".join(d.get("path", []))}
        for d in metrics["dropped_demands"] if d["reason"] == "insufficient_capacity"
    ]
    bandwidth_violating_flows = len(bandwidth_violating_flow_list)
    bandwidth_dropped_mbps = sum(d["demand"] for d in bandwidth_violating_flow_list)

    topology = ctrl.get_topology_snapshot()
    latency_lookup: Dict[tuple, float] = {}
    for link_id, attrs in topology.links.items():
        latency_lookup[(link_id.u, link_id.v)] = attrs.latency_ms
        latency_lookup[(link_id.v, link_id.u)] = attrs.latency_ms

    delay_violating_flow_list = []
    total_delay_excess_ms = 0.0
    for entry in metrics["accepted_demands"]:
        path = entry["path"]
        path_latency_ms = sum(
            latency_lookup[(path[i], path[i + 1])] for i in range(len(path) - 1)
        )
        if path_latency_ms > delay_budget_ms:
            excess = path_latency_ms - delay_budget_ms
            total_delay_excess_ms += excess
            delay_violating_flow_list.append({
                "src": entry["src"], "dst": entry["dst"],
                "latency_ms": path_latency_ms, "delay_excess_ms": excess,
            })
    delay_violating_flows = len(delay_violating_flow_list)

    return {
        "bandwidth_violation": 1 if bandwidth_violating_flows else 0,
        "bandwidth_violating_flows": bandwidth_violating_flows,
        "bandwidth_dropped_mbps": bandwidth_dropped_mbps,
        "bandwidth_violating_flow_list": bandwidth_violating_flow_list,
        "delay_violation": 1 if delay_violating_flows else 0,
        "delay_violating_flows": delay_violating_flows,
        "total_delay_excess_ms": total_delay_excess_ms,
        "delay_violating_flow_list": delay_violating_flow_list,
    }
