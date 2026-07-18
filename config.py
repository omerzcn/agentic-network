import os
from dotenv import load_dotenv

num_steps = 120
algorithms = ["heuristic_no_delay", "heuristic_low_delay", "heuristic_high_delay", "agentic"]

# For quick debugging runs: ALGORITHMS=agentic python3 main.py
_algorithms_override = os.getenv("ALGORITHMS")
if _algorithms_override:
    algorithms = [a.strip() for a in _algorithms_override.split(",") if a.strip()]

colors = {
    "heuristic_no_delay": "blue",
    "heuristic_low_delay": "orange",
    "heuristic_high_delay": "red",
    "agentic": "green",
}

# Added: heuristic_no_delay and heuristic_low_delay created identical curves,
# so adding different line styles enables that each is visible perfectly.
linestyles = {
    "heuristic_no_delay": "-",
    "heuristic_low_delay": "--",
    "heuristic_high_delay": "-.",
    "agentic": "--",
}

algo_delays = {
    "heuristic_no_delay": 0.0,
    "heuristic_low_delay": 0.5,
    "heuristic_high_delay": 10.0,
}

metric_names = ["total", "acceptance_rate"]

random_seed = 50

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_MY_API_KEY")
LLM_BACKEND = os.getenv("LLM_BACKEND", "openrouter")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")

# Keeping Ollama loaded in memory between calls
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

LLM_TEMPERATURE = 0

# llama struggle with long prompts, so keeping the candidate list smaller
CANDIDATES_PER_FLOW = int(os.getenv("CANDIDATES_PER_FLOW", "4" if LLM_BACKEND == "ollama" else "8"))

# Which policy plugs into the "agentic" slot: "llm" or "global_te" (DeterministicGlobalTE, no LLM involved).
POLICY_MODE = os.getenv("POLICY_MODE", "llm")

RESULTS_LABEL = os.getenv("RESULTS_LABEL", "global_te" if POLICY_MODE == "global_te" else LLM_MODEL.replace("/", "_"))
RESULTS_DIR = os.path.join("results", RESULTS_LABEL)

verbosity_level = 1  # 0: no print, 1: few prints (important messages), 2: more prints (detailed)

network_args = {
    "num_nodes": 15,
    "link_prob": 0.5,
    "capacity_min": 20,
    "capacity_max": 80,
    "latency_min": 1,
    "latency_max": 10,
    "loss_min": 0.0,
    "loss_max": 0.02,
    "fail_p_min": 0.001,
    "fail_p_max": 0.01,
}

base_demand = 8.0

traffic_pattern_args = {
    "latency_requirement_min_ms": 4.0,
    "latency_requirement_max_ms": 8.0,
}

traffic_args = {
    "seasonal_amplitude": 0.0,
    "seasonal_frequency": 24.0,
    "noise_power": 0.0,
}
