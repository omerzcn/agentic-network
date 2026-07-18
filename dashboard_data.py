# This script loads data for the dashboard from files in results folder

import glob
import json
import os

import pandas as pd
import streamlit as st

import compare_results

RESULTS_DIR = "results"

@st.cache_data
def load_available_models() -> list:
    labels = []
    for summary_path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*", "summary.json"))):
        labels.append(os.path.basename(os.path.dirname(summary_path)))
    return labels


@st.cache_data
def load_summary(label: str) -> dict:
    with open(os.path.join(RESULTS_DIR, label, "summary.json")) as f:
        return json.load(f)

@st.cache_data
def load_network_links(label: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, label, "network_links.csv"))

@st.cache_data
def load_link_utilization(label: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, label, "link_utilization.csv"))

@st.cache_data
def load_drop_events(label: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, label, "drop_events.csv"))

@st.cache_data
def load_step_metrics(label: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, label, "step_metrics.csv"))

@st.cache_data
def load_all_summaries() -> pd.DataFrame:
    # Reuses compare_results.py's logic directly
    return compare_results.load_summaries(RESULTS_DIR)
