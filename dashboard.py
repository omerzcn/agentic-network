# Evaluation dashboard 

import streamlit as st

import compare_results
import dashboard_charts as dc
import dashboard_data as dd
from config import algorithms as ALL_ALGORITHMS

st.set_page_config(page_title="Agentic Routing - Evaluation Dashboard", layout="wide")
st.title("Agentic Routing - Evaluation Dashboard")

models = dd.load_available_models()
if not models:
    st.error("No results found under results/<label>/summary.json. Run main.py or run_models.py first.")
    st.stop()

with st.sidebar:
    st.header("Selection")
    selected_model = st.selectbox("Model", models)
    selected_algo = st.selectbox("Algorithm", ALL_ALGORITHMS)

summary = dd.load_summary(selected_model)

tab_overview, tab_kpis, tab_ai, tab_topology = st.tabs(
    ["Overview", "Network KPIs", "AI Model Metrics", "Network Topology"]
)

with tab_overview:
    st.subheader("Packet Delivery Ratio across all models")
    all_summaries = dd.load_all_summaries()
    st.plotly_chart(dc.build_comparison_chart(all_summaries), width="stretch")

    ranked = all_summaries.sort_values("agentic", ascending=False, na_position="last")
    display_df = compare_results.format_display_df(ranked)
    st.markdown(dc.render_centered_table_html(display_df), unsafe_allow_html=True)

with tab_kpis:
    st.subheader(f"Network KPIs - {selected_model}")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(dc.build_kpi_bars(summary), width="stretch")
    with col2:
        st.plotly_chart(dc.build_violation_duration_bars(summary), width="stretch")

    step_metrics = dd.load_step_metrics(selected_model)
    st.plotly_chart(dc.build_utilization_timeseries(step_metrics), width="stretch")

with tab_ai:
    st.markdown(f"## {summary.get('llm_model', selected_model)}")
    st.caption(f"Backend: `{summary.get('llm_backend', '-')}`  ·  Temperature: {summary.get('llm_temperature', '-')}")

    stats = summary.get("llm_agent_stats", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Flow decisions", f"{stats.get('total_flows_decided', 0):,}")
    kept, reconciled = stats.get("total_kept_llm_pick"), stats.get("total_reconciled")
    kept_pct = f"{kept / reconciled:.1%}" if reconciled else "-"
    c2.metric("Kept LLM's own pick", kept_pct)
    c3.metric("Total tokens", f"{stats.get('total_tokens', 0):,}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg call latency (s)", f"{stats.get('average_call_latency_s', 0):.2f}")
    c2.metric("LLM calls (HTTP round-trips)", stats.get("llm_calls", 0))
    c3.empty()

    st.divider()
    st.markdown("**Local deployment resource usage**")
    if "avg_cpu_percent" in stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg CPU", f"{stats['avg_cpu_percent']:.0f}%")
        c2.metric("Avg memory", f"{stats['avg_memory_mb'] / 1024:.1f} GB")
        c3.metric("Peak memory", f"{stats['peak_memory_mb'] / 1024:.1f} GB")
        if "avg_gpu_percent" in stats:
            c4.metric("Avg GPU", f"{stats['avg_gpu_percent']:.0f}%")
        else:
            c4.metric("Avg GPU", "N/A")
        st.caption("CPU/memory are system-wide samples taken around each local call "
                   "(the model runs in a separate Ollama server process, not this script).")
    else:
        st.info(f"No local deployment data in this run (backend: {summary.get('llm_backend', '-')}).")

with tab_topology:
    st.subheader(f"Network Topology - {selected_model} / {selected_algo}")
    links = dd.load_network_links(selected_model)
    util = dd.load_link_utilization(selected_model)
    drops = dd.load_drop_events(selected_model)

    num_steps = summary.get("num_steps", 1)
    step = st.slider("Step", 0, num_steps - 1, min(60, num_steps - 1))

    pos = dc.compute_layout(links, seed=summary.get("random_seed", 0))
    fig = dc.build_topology_figure(links, util, pos, selected_algo, step)

    col_graph, col_events = st.columns([2, 1])
    with col_graph:
        st.plotly_chart(fig, width="stretch")
        st.caption("Edge color/width = utilization at the selected step "
                   "going red as it approaches capacity; "
                   "gray means link down.")
    with col_events:
        st.markdown("**SLA violation / drop events at this step**")
        # demand<=0 rows are flows that haven't started yet, not real drops.
        # So they're excluded from the table.
        step_drops = drops[
            (drops["algo"] == selected_algo) & (drops["step"] == step) & (drops["demand"] > 1e-12)
        ]
        if step_drops.empty:
            st.caption("No dropped flows at this step.")
        else:
            st.dataframe(
                step_drops[["src", "dst", "demand", "reason"]],
                width="stretch", hide_index=True,
            )
