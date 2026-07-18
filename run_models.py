# This script runs main.py once per LLM model listed in RUNS

import os
import subprocess
import sys

RUNS = [
    #{"backend": "openrouter", "model": "openai/gpt-4o-mini", "label": "gpt4o_mini"},
    #{"backend": "openrouter", "model": "anthropic/claude-haiku-4.5", "label": "claude-haiku-4.5"},
    #{"backend": "openrouter", "model": "deepseek/deepseek-v4-pro", "label": "deepseek"},
    {"backend": "openrouter", "model": "google/gemini-3.1-flash-lite", "label": "gemini"},
    #{"backend": "ollama", "model": "llama3.2", "label": "llama3_2_local"},
]

def run_model(backend: str, model: str, label: str) -> bool:
    env = os.environ.copy()
    env["LLM_BACKEND"] = backend
    env["LLM_MODEL"] = model
    env["RESULTS_LABEL"] = label

    print(f"\n=== Running {label} (backend={backend}, model={model}) ===")
    result = subprocess.run([sys.executable, "main.py"], env=env)
    return result.returncode == 0

if __name__ == "__main__":
    outcomes = {}
    for run in RUNS:
        outcomes[run["label"]] = run_model(**run)

    print("\n=== Batch summary ===")
    for label, ok in outcomes.items():
        print(f"  {label}: {'OK' if ok else 'FAILED'}")
