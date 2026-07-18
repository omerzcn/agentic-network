# This script sends a routing prompt to an LLM 
# and parses the JSON reply, and tracks call count, tokens and latency stats.

import json
import re
import time
from typing import Dict, Optional, Tuple
import requests

from config import LLM_BACKEND, LLM_MODEL, LLM_TEMPERATURE, OLLAMA_KEEP_ALIVE, OLLAMA_URL, OPENROUTER_API_KEY, OPENROUTER_URL

class LLMClient:
    def __init__(self):
        self.call_count = 0
        self.total_tokens = 0
        self.total_latency_s = 0

    def warm_up(self) -> None:
        # Loading Ollama into memory on first request, because it takes long time with no GPU
        if LLM_BACKEND != "ollama":
            return

        print(f"[LLMClient] Warming up local model '{LLM_MODEL}' (loading into memory)...")
        start = time.perf_counter()
        try:
            self._call_ollama('Reply with exactly this JSON: {"status": "ok"}', timeout=300)
        except requests.RequestException as exc:
            print(f"[LLMClient] Warm-up failed: {exc}")
            return
        print(f"[LLMClient] Warm-up done in {time.perf_counter() - start:.1f}s")

    def select_paths(self, prompt: str, schema: Optional[dict] = None) -> Optional[Dict[str, int]]:
        start = time.perf_counter()
        token_count = 0
        try:
            if LLM_BACKEND == "ollama":
                content, token_count = self._call_ollama(prompt, schema=schema)
            elif LLM_BACKEND == "openrouter":
                content, token_count = self._call_openrouter(prompt)
            else:
                raise ValueError("Unvalid LLM: " + str(LLM_BACKEND))
        except requests.RequestException as exc:
            print(f"[LLMClient] Request Failed: " + str(exc))
            return None
        finally:
            self.call_count += 1
            self.total_latency_s += (time.perf_counter() - start)
            self.total_tokens += token_count

        try:
            extracted = self._extract_json(content)
            parsed = json.loads(extracted)
        except (json.JSONDecodeError, TypeError):
            print(f"[LLMClient] Invalid JSON response: " + str(content))
            return None
        
        result: Dict[str, int] = {}

        for key, value in parsed.items():
            if isinstance(key, str) and isinstance(value, int):
                result[key] = value

        return result
    
    @staticmethod
    def _extract_json(content: str) -> str:
        match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            content,
            re.DOTALL,
        )
        if match:
            return match.group(1)
        return content.strip()
    
    def _call_openrouter(self, prompt: str) -> Tuple[str, int]:
        if not OPENROUTER_API_KEY:
            raise RuntimeError("API Key Missing!")
        headers = {
            "Authorization": ("Bearer " + OPENROUTER_API_KEY),
            "Content-Type": "application/json",
        }
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": LLM_TEMPERATURE,
        }
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {}).get("total_tokens", 0)
        return content, int(usage)
    
    def _call_ollama(self, prompt: str, timeout: int = 180, schema: Optional[dict] = None) -> Tuple[str, int]:
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            # A schema with required keys forces the local model to produce a
            # value for every one of them
            "format": schema if schema is not None else "json",
            "options": {"temperature": LLM_TEMPERATURE, "num_predict": -1},
        }
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data["message"]["content"]
        token_count = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
        return content, int(token_count)
    
    def get_stats(self) -> dict:
        average_latency = 0.0
        if self.call_count:
            average_latency = (self.total_latency_s / self.call_count)
        return {
            "llm_calls": self.call_count,
            "total_tokens": self.total_tokens,
            "average_call_latency_s": (average_latency), 
        }
    

