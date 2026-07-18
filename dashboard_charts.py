# Functions for the dashboard

from typing import Dict, Tuple

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

def render_centered_table_html(df: pd.DataFrame) -> str:
    # Overview table in html format
    html_df = df.rename(columns=lambda c: c.replace("\n", "<br>"))
    table_html = html_df.to_html(index=False, escape=False, classes="centered-table", border=0)
    style = (
        "<style>\n"
        ".centered-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }\n"
        ".centered-table th, .centered-table td { text-align: center !important; padding: 8px 14px; }\n"
        ".centered-table th { background-color: #2c3e50; color: white; font-weight: 600; }\n"
        ".centered-table tr:nth-child(even) { background-color: rgba(128, 128, 128, 0.12); }\n"
        "</style>\n"
    )
    return style + table_html


ALGO_COLORS = {
    "heuristic_no_delay": "#1f77b4",
    "heuristic_low_delay": "#ff7f0e",
    "heuristic_high_delay": "#d62728",
    "agentic": "#2ca02c",
}

# Congestion colors
_UTIL_LOW, _UTIL_MID, _UTIL_HIGH = "#2ca02c", "#f0ad4e", "#d62728"

def _util_color(ratio: float) -> str:
    if ratio >= 0.9:
        return _UTIL_HIGH
    if ratio >= 0.6:
        return _UTIL_MID
    return _UTIL_LOW

def build_comparison_chart(all_summaries: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    algo_cols = ["heuristic_no_delay", "heuristic_low_delay", "heuristic_high_delay", "agentic"]
    labels = {
        "heuristic_no_delay": "Heuristic (no delay)",
        "heuristic_low_delay": "Heuristic (low delay)",
        "heuristic_high_delay": "Heuristic (high delay)",
        "agentic": "Agentic (LLM)",
    }
    for algo in algo_cols:
        fig.add_trace(go.Bar(
            name=labels[algo],
            x=all_summaries["model"],
            y=all_summaries[algo],
            marker_color=ALGO_COLORS[algo],
        ))
    fig.update_layout(
        barmode="group",
        title="Packet Delivery Ratio by model and algorithm",
        yaxis_title="PDR",
        yaxis_tickformat=".0%",
        legend_title="Algorithm",
    )
    return fig

def build_kpi_bars(summary: dict) -> go.Figure:
    pdr = summary.get("pdr_by_algorithm", {})
    algos = list(pdr.keys())
    fig = go.Figure(go.Bar(
        x=algos,
        y=[pdr[a] for a in algos],
        marker_color=[ALGO_COLORS.get(a, "#888") for a in algos],
    ))
    fig.update_layout(
        title="Packet Delivery Ratio by algorithm",
        yaxis_title="PDR",
        yaxis_tickformat=".0%",
    )
    return fig

def build_violation_duration_bars(summary: dict) -> go.Figure:
    sla = summary.get("sla_violations_by_algorithm", {})
    algos = list(sla.keys())
    num_steps = summary.get("num_steps", 1)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Bandwidth violation",
        x=algos,
        y=[sla[a].get("bandwidth_violation_duration_steps", 0) for a in algos],
        marker_color="#f58518",
    ))
    fig.add_trace(go.Bar(
        name="Delay violation",
        x=algos,
        y=[sla[a].get("delay_violation_duration_steps", 0) for a in algos],
        marker_color="#e45756",
    ))
    fig.update_layout(
        barmode="group",
        title=f"SLA violation duration by algorithm (of {num_steps} simulated steps)",
        yaxis_title="Steps with >=1 violation",
    )
    return fig

def build_utilization_timeseries(step_metrics: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for algo in step_metrics["algo"].unique():
        sub = step_metrics[step_metrics["algo"] == algo]
        fig.add_trace(go.Scatter(
            x=sub["step"], y=sub["acceptance_rate"],
            mode="lines", name=algo,
            line=dict(color=ALGO_COLORS.get(algo, "#888")),
        ))
    fig.update_layout(
        title="Acceptance rate over time",
        xaxis_title="simulated step",
        yaxis_title="acceptance rate",
        yaxis_tickformat=".0%",
        legend_title="Algorithm",
    )
    return fig

def compute_layout(links: pd.DataFrame, seed: int) -> Dict[str, Tuple[float, float]]:
    graph = nx.Graph()
    for _, row in links.iterrows():
        graph.add_edge(row["u"], row["v"])
    return nx.spring_layout(graph, seed=seed, k=2, iterations=200)

def build_topology_figure(
        links: pd.DataFrame,
        link_utilization: pd.DataFrame,
        pos: Dict[str, Tuple[float, float]],
        algo: str,
        step: int,
) -> go.Figure:
    step_util = link_utilization[(link_utilization["algo"] == algo) & (link_utilization["step"] == step)]

    # Utilization is asymmetric 
    util_by_undirected: Dict[Tuple[str, str], float] = {}
    up_by_undirected: Dict[Tuple[str, str], bool] = {}
    for _, row in step_util.iterrows():
        key = tuple(sorted((row["u"], row["v"])))
        util_by_undirected[key] = util_by_undirected.get(key, 0.0) + row["utilization"]
        up_by_undirected[key] = up_by_undirected.get(key, True) and bool(row["up"])

    fig = go.Figure()

    for _, link in links.iterrows():
        key = (link["u"], link["v"])
        x0, y0 = pos[link["u"]]
        x1, y1 = pos[link["v"]]
        up = up_by_undirected.get(key, True)
        util = util_by_undirected.get(key, 0.0)
        ratio = min(util / link["capacity"], 1.5) if link["capacity"] > 0 else 0.0

        if not up:
            line = dict(color="#999999", width=2, dash="dash")
            hover = f"{link['u']}-{link['v']} (DOWN)"
        else:
            line = dict(color=_util_color(ratio), width=2 + 6 * min(ratio, 1.0))
            hover = f"{link['u']}-{link['v']}<br>utilization: {util:.1f}/{link['capacity']:.1f} Mbps ({ratio:.0%})"

        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=line, hoverinfo="text", text=hover, showlegend=False,
        ))

    node_x = [pos[n][0] for n in pos]
    node_y = [pos[n][1] for n in pos]
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=list(pos.keys()), textposition="top center",
        marker=dict(size=18, color="#4c78a8"),
        showlegend=False, hoverinfo="text",
    ))

    fig.update_layout(
        title=f"Network topology - {algo}, step {step}",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig
