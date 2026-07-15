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

if __name__ == "__main__":

    # Set random seeds for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        if verbosity_level >= 1:
            print(f"Random seed set to: {random_seed}")

    os.makedirs("results", exist_ok=True)

    # Create network
    g = NetworkGraph()
    create_random_network(g, seed=random_seed, **network_args)

    # (Optional) visualize
    g.draw_to_file("results/network.png", random_seed)
    g.save_to_text_file("results/network.txt")

    # Create traffic model
    base_demands = create_random_traffic_pattern(g, seed=random_seed, base_demand=base_demand, start_t_max=num_steps)
    traffic = TrafficModel(base_demands, **traffic_args)

    # Setup controller + agent
    ctrl = ControllerAPI(g)
    llm_agent = RoutingAgent(ctrl, candidates_per_flow=8)

    # Collect metrics
    all_history = {}
    all_history_per_demand = {}

    with Simulator(g, traffic, ctrl, llm_agent, verbosity=verbosity_level) as simulator:
        for algo in algorithms:

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
            for t in tqdm(range(num_steps)):
                _, metrics = simulator.step(algo)
                for k in metric_names:
                    history[k].append(metrics.get(k, 0.0))
                for item in metrics["accepted_demands"]:
                    history_per_demand.at[t, (item["src"], item["dst"])] = item["demand"]
                for item in metrics["dropped_demands"]:
                    drop_reason_counts[item["reason"]] += 1
                    drop_reason_mbps[item["reason"]] += item["demand"]

            all_history[algo] = history
            all_history_per_demand[algo] = history_per_demand

            print(f"[{algo}] drop reasons (count): {dict(drop_reason_counts)}")
            print(f"[{algo}] drop reasons (Mbps): {dict(drop_reason_mbps)}")

    # Plot metrics
    fig, axes = plt.subplots(1, len(metric_names), figsize=(12, 4), sharex=True)
    axes = axes.flatten()

    for ax, name in zip(axes, metric_names):
        for algo in algorithms:
            ax.plot(all_history[algo][name], label=algo, color=colors[algo])
        ax.set_title(name)
        ax.set_xlabel("time step")
        ax.set_ylabel(name)
        if name == "loss_rate":
            ax.set_ylim([0, 1])
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    fig.savefig("results/results.png", dpi=150)
    plt.close(fig)

    for algo in algorithms:
        all_history_per_demand[algo].to_csv(f"results/all_history_{algo}.csv")

    print("Simulation completed. Metric results were saved to results.png.")
