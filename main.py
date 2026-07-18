import json
import os
import random
from collections import Counter

# matplot crushed my laptop by trying to oopen a tool in background.
# So, this prevent opening a new window, just saves pictures to files.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from agents import RoutingAgent
from config import *

from helpers import create_random_network, create_random_traffic_pattern

from simulation import NetworkGraph, TrafficModel, ControllerAPI, Simulator

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
    base_demands = create_random_traffic_pattern(
        g,
        seed=random_seed,
        base_demand=base_demand,
        start_t_max=num_steps,
        **traffic_pattern_args,
    )
    traffic = TrafficModel(base_demands, **traffic_args)

    # Setup controller
    ctrl = ControllerAPI(g)
    if POLICY_MODE == "global_te":
        llm_agent = DeterministicGlobalTE(ctrl, candidates_per_flow=8, traffic=traffic)
    else:
        llm_agent = RoutingAgent(ctrl, candidates_per_flow=CANDIDATES_PER_FLOW, traffic=traffic)
        llm_agent.llm_client.warm_up()
    print(f"Policy for the 'agentic' slot: {POLICY_MODE}")

    # Collect metrics
    all_history = {}
    all_history_per_demand = {}
    aggregate_acceptance = {}
    aggregate_sla = {}

    request_outcome_keys = [
        ("Accepted", "accepted_request_count", "#4c78a8"),
        ("Rejected: delay/SLA", "delay_dropped_request_count", "#f58518"),
        ("Rejected: no valid path", "no_valid_path_request_count", "#e45756"),
        ("Rejected: other", "other_dropped_request_count", "#72b7b2"),
    ]
    request_outcome_totals = {
        algo: {key: 0 for _, key, _ in request_outcome_keys}
        for algo in algorithms
    }

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
                for _, key, _ in request_outcome_keys:
                    request_outcome_totals[algo][key] += metrics.get(key, 0)

                # Per-flow latency SLA is now enforced inside compute_metrics itself
                # (a demand whose path misses its own max_latency_ms is dropped with
                # reason "latency_requirement_not_met"), so these are just straight
                # accumulations of the fields it already returns.
                sla_totals["active_request_count"] += metrics["active_request_count"]
                sla_totals["accepted_request_count"] += metrics["accepted_request_count"]
                sla_totals["delay_dropped_request_count"] += metrics["delay_dropped_request_count"]
                sla_totals["no_valid_path_request_count"] += metrics["no_valid_path_request_count"]
                sla_totals["other_dropped_request_count"] += metrics["other_dropped_request_count"]
                sla_totals["delay_dropped_mbps"] += metrics["latency_dropped"]

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

    # Plot aggregate request outcomes as one stacked bar per algorithm.
    fig_bar, ax_bar = plt.subplots(figsize=(max(8, 1.4 * len(algorithms) + 3), 5))
    x = np.arange(len(algorithms))
    bottom = np.zeros(len(algorithms))

    for label, key, color in request_outcome_keys:
        values = np.array([request_outcome_totals[algo][key] for algo in algorithms])
        ax_bar.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values

    ax_bar.set_title("Request outcomes by algorithm")
    ax_bar.set_xlabel("algorithm")
    ax_bar.set_ylabel("number of active requests")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(algorithms, rotation=20, ha="right")
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.legend()
    fig_bar.tight_layout()
    fig_bar.savefig(f"{RESULTS_DIR}/request_outcomes_stacked_bar.png", dpi=150)
    plt.close(fig_bar)

    request_outcomes = pd.DataFrame.from_dict(request_outcome_totals, orient="index")
    request_outcomes.to_csv(f"{RESULTS_DIR}/request_outcomes_by_algorithm.csv")

    for algo in algorithms:
        all_history_per_demand[algo].to_csv(f"{RESULTS_DIR}/all_history_{algo}.csv")

    print(f"Simulation completed. Metric results were saved to {RESULTS_DIR}/results.png.")
    print(f"Request outcome bar plot was saved to {RESULTS_DIR}/request_outcomes_stacked_bar.png.")

    # summary.json
    summary = {
        "llm_backend": LLM_BACKEND,
        "llm_model": LLM_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
        "random_seed": random_seed,
        "num_steps": num_steps,
        "latency_requirement_range_ms": [
            traffic_pattern_args["latency_requirement_min_ms"],
            traffic_pattern_args["latency_requirement_max_ms"],
        ],
        "algo_delays": algo_delays,
        "pdr_by_algorithm": aggregate_acceptance,
        "sla_violations_by_algorithm": aggregate_sla,
        "llm_agent_stats": llm_agent.get_stats(),
    }
    with open(f"{RESULTS_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {RESULTS_DIR}/summary.json")
