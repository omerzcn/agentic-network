import json
import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from agents import RoutingAgent
from config import *

from helpers import create_random_network, create_random_traffic_pattern

from simulation import NetworkGraph, TrafficModel, ControllerAPI, Simulator

from metrics_logger import compute_sla_metrics

from global_te_policy import DeterministicGlobalTE

if __name__ == "__main__":

    # Set random seeds for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        if verbosity_level >= 1:
            print(f"Random seed set to: {random_seed}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results will be saved to: {RESULTS_DIR}")

    # Create network
    g = NetworkGraph()
    create_random_network(g, seed=random_seed, **network_args)

    # (Optional) visualize
    g.draw_to_file(f"{RESULTS_DIR}/network.png", random_seed)
    g.save_to_text_file(f"{RESULTS_DIR}/network.txt")

    # Create traffic model
    base_demands = create_random_traffic_pattern(g, seed=random_seed, base_demand=base_demand, start_t_max=num_steps)
    traffic = TrafficModel(base_demands, **traffic_args)

    # Setup controller
    ctrl = ControllerAPI(g)
    if POLICY_MODE == "global_te":
        llm_agent = DeterministicGlobalTE(ctrl, candidates_per_flow=8)
    else:
        llm_agent = RoutingAgent(ctrl, candidates_per_flow=8)
    print(f"Policy for the 'agentic' slot: {POLICY_MODE}")

    # Collect metrics
    all_history = {}
    all_history_per_demand = {}
    aggregate_acceptance = {}
    aggregate_sla = {}

    with Simulator(g, traffic, ctrl, llm_agent, verbosity=verbosity_level) as simulator:
        for algo in algorithms:

            # Reseed so that every algorithm sees the same sequence of link failures and noise,
            # not whatever is left over from the previous algorithm's run.
            if random_seed is not None:
                random.seed(random_seed)
                np.random.seed(random_seed)

            simulator.reset_agentic_state()
            ctrl.reset()
            g.reset_link_states()

            history = {k: [] for k in metric_names}
            history_per_demand = pd.DataFrame(
                columns=[(src, dst) for src in g.nodes for dst in g.nodes],
                index=range(num_steps),
            )
            drop_reason_counts = Counter()
            drop_reason_mbps = Counter()
            sla_totals = Counter()
            total_accepted_mbps = 0.0
            total_offered_mbps = 0.0
            for t in tqdm(range(num_steps)):
                _, metrics = simulator.step(algo)
                for k in metric_names:
                    history[k].append(metrics.get(k, 0.0))
                for item in metrics["accepted_demands"]:
                    history_per_demand.at[t, (item["src"], item["dst"])] = item["demand"]
                for item in metrics["dropped_demands"]:
                    drop_reason_counts[item["reason"]] += 1
                    drop_reason_mbps[item["reason"]] += item["demand"]

                sla = compute_sla_metrics(ctrl, metrics, DELAY_BUDGET_MS)
                sla_totals["bandwidth_violation_steps"] += sla["bandwidth_violation"]
                sla_totals["bandwidth_violating_flows"] += sla["bandwidth_violating_flows"]
                sla_totals["bandwidth_dropped_mbps"] += sla["bandwidth_dropped_mbps"]
                sla_totals["delay_violation_steps"] += sla["delay_violation"]
                sla_totals["delay_violating_flows"] += sla["delay_violating_flows"]
                sla_totals["total_delay_excess_ms"] += sla["total_delay_excess_ms"]

                # Weighted by Mbps (not per-step average), so busier steps count more
                total_accepted_mbps += metrics["accepted"]
                total_offered_mbps += metrics["total"]

            all_history[algo] = history
            all_history_per_demand[algo] = history_per_demand

            acceptance_ratio = total_accepted_mbps / total_offered_mbps if total_offered_mbps > 0 else 0.0
            aggregate_acceptance[algo] = acceptance_ratio
            aggregate_sla[algo] = dict(sla_totals)

            print(f"[{algo}] drop reasons (count): {dict(drop_reason_counts)}")
            print(f"[{algo}] drop reasons (Mbps): {dict(drop_reason_mbps)}")
            print(f"[{algo}] SLA violations (of {num_steps} steps): {dict(sla_totals)}")
            print(f"[{algo}] aggregate acceptance ratio: {acceptance_ratio:.4f} "
                  f"({total_accepted_mbps:.1f}/{total_offered_mbps:.1f} Mbps)")

        print("\nAggregate acceptance ratio by algorithm:")
        for algo, ratio in aggregate_acceptance.items():
            print(f"  {algo}: {ratio:.4f}")

    # Plot metrics
    fig, axes = plt.subplots(1, len(metric_names), figsize=(12, 4), sharex=True)
    axes = axes.flatten()

    for ax, name in zip(axes, metric_names):
        for algo in algorithms:
            ax.plot(all_history[algo][name], label=algo, color=colors[algo], linestyle=linestyles[algo])
        ax.set_title(name)
        ax.set_xlabel("time step")
        ax.set_ylabel(name)
        if name == "loss_rate":
            ax.set_ylim([0, 1])
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/results.png", dpi=150)
    plt.close(fig)

    for algo in algorithms:
        all_history_per_demand[algo].to_csv(f"{RESULTS_DIR}/all_history_{algo}.csv")

    print(f"Simulation completed. Metric results were saved to {RESULTS_DIR}/results.png.")

    # summary.json
    summary = {
        "llm_backend": LLM_BACKEND,
        "llm_model": LLM_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
        "random_seed": random_seed,
        "num_steps": num_steps,
        "delay_budget_ms": DELAY_BUDGET_MS,
        "algo_delays": algo_delays,
        "pdr_by_algorithm": aggregate_acceptance,
        "sla_violations_by_algorithm": aggregate_sla,
        "llm_agent_stats": llm_agent.get_stats(),
    }
    with open(f"{RESULTS_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {RESULTS_DIR}/summary.json")
