from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from harness.client import OllamaClient


class BaseBenchmark(ABC):
    name: str = "base"

    def __init__(self, client: OllamaClient, config: dict):
        self.client = client
        self.config = config
        self.judge_client = config.get("judge_client", client)  # separate client for LLM-as-judge

    @abstractmethod
    def load_samples(self) -> list[dict]:
        """Return list of sample dicts with at minimum 'id' and 'prompt' keys."""

    @abstractmethod
    def score(self, sample: dict, response: str) -> dict:
        """Return scoring dict with at minimum 'passed' (bool) and 'score' (float 0-1)."""

    def format_prompt(self, sample: dict) -> str:
        return sample["prompt"]

    # Set to True in subclasses where extended reasoning is the point (e.g. philosophical).
    # For all others, /no_think suppresses qwen3's internal thinking chain which can
    # burn thousands of tokens per sample and make runs impractically slow.
    allow_thinking: bool = False

    def system_prompt(self) -> str | None:
        return None

    def _effective_system(self, model: str = "") -> str | None:
        base = self.system_prompt()
        # /no_think is a qwen3-specific tag to suppress extended thinking chains.
        # Don't add it for other models — it confuses them and produces worse output.
        is_qwen3 = "qwen3" in model.lower()
        if self.allow_thinking or not is_qwen3:
            return base
        return f"{base}\n/no_think" if base else "/no_think"

    def run(self, model: str, n_samples: int = None, on_sample=None) -> list[dict]:
        samples = self.load_samples()
        if n_samples:
            samples = samples[:n_samples]

        results = []
        for i, sample in enumerate(samples):
            prompt = self.format_prompt(sample)
            response = self.client.complete(
                model=model,
                prompt=prompt,
                system=self._effective_system(model),
                max_tokens=self.config.get("max_tokens", 2048),
            )
            scoring = self.score(sample, response["content"]) if not response["error"] else {"passed": False, "score": 0.0}
            result = {
                "id": sample.get("id", ""),
                "model": model,
                "benchmark": self.name,
                "ts": datetime.now().isoformat(timespec="seconds"),
                "prompt": prompt[:500],
                "response": response["content"][:1000],
                "elapsed": response["elapsed"],
                "tok_per_sec": response["tok_per_sec"],
                "error": response["error"],
                **scoring,
            }
            results.append(result)
            if on_sample:
                on_sample(i + 1, len(samples), result)
        return results
