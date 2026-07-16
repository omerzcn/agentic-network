import os
from dotenv import load_dotenv

num_steps = 100
algorithms = ["heuristic_no_delay", "heuristic_low_delay", "heuristic_high_delay", "agentic"]

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

DELAY_BUDGET_MS = 15

random_seed = 42

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_MY_API_KEY")
LLM_BACKEND = os.getenv("LLM_BACKEND", "openrouter")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
LLM_TEMPERATURE = 0

RESULTS_LABEL = os.getenv("RESULTS_LABEL", LLM_MODEL.replace("/", "_"))
RESULTS_DIR = os.path.join("results", RESULTS_LABEL)

verbosity_level = 1  # 0: no print, 1: few prints (important messages), 2: more prints (detailed)

network_args = {
    "num_nodes": 10,
    "link_prob": 1.0,
    "capacity_min": 10,
    "capacity_max": 20,
    "latency_min": 1,
    "latency_max": 10,
    "loss_min": 0.0,
    "loss_max": 0.1,
    "fail_p_min": 0.0,
    "fail_p_max": 0.1,
}

base_demand = 10.0

traffic_args = {
    "seasonal_amplitude": 0.0,
    "seasonal_frequency": 24.0,
    "noise_power": 0.0,
}
