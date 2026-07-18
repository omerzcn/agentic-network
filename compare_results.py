# This scripts saves a comparison table as a png file using summary.json for 4 models

import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

ALGO_COLUMNS = ["heuristic_no_delay", "heuristic_low_delay", "heuristic_high_delay", "agentic"]

def load_summaries(results_dir: str = "results") -> pd.DataFrame:
    rows = []

    for summary_path in sorted(glob.glob(os.path.join(results_dir, "*", "summary.json"))):
        label = os.path.basename(os.path.dirname(summary_path))

        with open(summary_path) as f:
            data = json.load(f)

        pdr = data.get("pdr_by_algorithm", {})
        stats = data.get("llm_agent_stats", {})

        row = {
            "model": data.get("llm_model"),
        }
        for algo in ALGO_COLUMNS:
            row[algo] = pdr.get(algo)

        # Flow decisions is the real work-done number
        row["flow_decisions"] = stats.get("total_flows_decided")
        total_reconciled = stats.get("total_reconciled")
        kept = stats.get("total_kept_llm_pick")
        row["kept_llm_pick_ratio"] = (kept / total_reconciled) if total_reconciled else None
        row["avg_call_latency_s"] = stats.get("average_call_latency_s")

        rows.append(row)

    return pd.DataFrame(rows)

def format_display_df(df: pd.DataFrame) -> pd.DataFrame:
    # Shared formatting for static PNG table and the dashboard's overview
    display_df = df.rename(columns={
        "model": "Model",
        "heuristic_no_delay": "Heuristic\n(no delay)",
        "heuristic_low_delay": "Heuristic\n(low delay)",
        "heuristic_high_delay": "Heuristic\n(high delay)",
        "agentic": "Agentic\n(LLM)",
        "flow_decisions": "Flow\ndecisions",
        "kept_llm_pick_ratio": "Kept LLM\npick %",
        "avg_call_latency_s": "Avg latency\n(s)",
    })

    for col in ["Heuristic\n(no delay)", "Heuristic\n(low delay)", "Heuristic\n(high delay)", "Agentic\n(LLM)", "Kept LLM\npick %"]:
        display_df[col] = display_df[col].map(lambda v: f"{v:.2%}" if pd.notnull(v) else "-")
    display_df["Avg latency\n(s)"] = display_df["Avg latency\n(s)"].map(lambda v: f"{v:.2f}" if pd.notnull(v) else "-")
    display_df["Flow\ndecisions"] = display_df["Flow\ndecisions"].map(lambda v: f"{int(v)}" if pd.notnull(v) else "-")
    return display_df

def render_table_image(df: pd.DataFrame, out_path: str) -> None:
    display_df = format_display_df(df)

    n_rows, n_cols = display_df.shape
    fig, ax = plt.subplots(figsize=(1.6 * n_cols, 0.6 * (n_rows + 1)))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    table.auto_set_column_width(col=list(range(n_cols)))

    best_agentic_col = list(display_df.columns).index("Agentic\n(LLM)")
    for col in range(n_cols):
        cell = table[0, col]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", weight="bold")

    for row in range(1, n_rows + 1):
        for col in range(n_cols):
            cell = table[row, col]
            cell.set_facecolor("#f0f3f6" if row % 2 == 0 else "white")
            if col == best_agentic_col:
                cell.set_text_props(weight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    df = load_summaries()
    df = df.sort_values("agentic", ascending=False, na_position="last")

    out_path = "results/comparison_table.png"
    render_table_image(df, out_path)
    print(f"Saved to {out_path}")
