import random
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from harness.client import OllamaClient


class MemorySwapAbort(Exception):
    """Raised when a model is causing memory thrashing / swap. The caller
    should abort the current model and skip remaining benchmarks for it."""
    def __init__(self, message: str, partial_results: list[dict] | None = None):
        super().__init__(message)
        self.partial_results = partial_results or []


def _median(values: list[float]) -> float:
    clean = sorted(v for v in values if v is not None and v > 0)
    if not clean:
        return 0.0
    n = len(clean)
    return clean[n // 2] if n % 2 else (clean[n // 2 - 1] + clean[n // 2]) / 2


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
    def score(self, sample: dict, response: str, tool_calls: list[dict] | None = None) -> dict:
        """Return scoring dict with at minimum 'passed' (bool) and 'score' (float 0-1)."""

    def format_prompt(self, sample: dict) -> str:
        return sample["prompt"]

    def format_tools(self, sample: dict) -> list[dict] | None:
        """Override to supply OpenAI-style function schemas for a sample.

        None (the default) means no `tools` are sent — every existing
        text-scoring benchmark leaves this alone.
        """
        return None

    # Set in subclasses whose samples span groups that must all be represented
    # (MMLU spans academic subjects). Without stratification, taking the first N
    # of a concatenated list draws every sample from whichever group happens to
    # come first — MMLU at n=20 was 20/20 abstract algebra, despite five
    # subjects being configured.
    stratify_key: str | None = None

    def select_samples(self, samples: list[dict], n: int | None) -> list[dict]:
        """Pick n samples deterministically but representatively.

        Seeded, so runs stay reproducible and every model sees an identical set;
        shuffled, so the sample is not a fixed slice of whatever order the
        dataset shipped in; stratified when the benchmark spans groups, so n=20
        means 4 from each of 5 subjects rather than 20 from one.
        """
        if not n or n >= len(samples):
            return samples
        rng = random.Random(self.config.get("sample_seed", 42))

        if not self.stratify_key:
            pool = list(samples)
            rng.shuffle(pool)
            return pool[:n]

        groups: dict[Any, list[dict]] = {}
        for s in samples:
            groups.setdefault(s.get(self.stratify_key), []).append(s)
        for g in groups.values():
            rng.shuffle(g)

        # Round-robin so the groups stay balanced at any n, and any remainder
        # is spread across the earliest groups rather than dumped on one.
        keys = sorted(groups, key=lambda k: (k is None, k))
        picked: list[dict] = []
        i = 0
        while len(picked) < n and any(groups[k] for k in keys):
            bucket = groups[keys[i % len(keys)]]
            if bucket:
                picked.append(bucket.pop())
            i += 1
        return picked[:n]

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

    def run(self, model: str, n_samples: int = None, on_sample=None, ctx: int | None = None) -> list[dict]:
        samples = self.select_samples(self.load_samples(), n_samples)
        actual_n = len(samples)

        guard_cfg = self.config.get("memory_guard", {})
        # Thinking is an explicit, recorded dimension: it changes results
        # substantially, so it must never vary implicitly with the Ollama
        # version. Config wins; otherwise the benchmark's own default (only
        # philosophical wants extended reasoning).
        think = self.config.get("think")
        if think is None:
            think = self.allow_thinking

        # Thinking spends its budget before the answer starts, so it needs a
        # bigger cap or the run measures exhaustion instead of capability.
        max_tokens = self.config.get("max_tokens", 4096)
        if think:
            max_tokens = self.config.get("think_max_tokens", max(max_tokens, 16384))

        results = []
        for i, sample in enumerate(samples):
            prompt = self.format_prompt(sample)
            # Streaming so TTFT/prefill is separable from decode, and so the
            # watchdog can abort a thrashing call mid-flight instead of after it
            # completes. tok_per_sec stays end-to-end (prefill included) for
            # continuity with older results; decode_tps is the generation rate
            # comparable to published tok/s figures.
            response = self.client.complete_native(
                model=model,
                prompt=prompt,
                system=self._effective_system(model),
                max_tokens=max_tokens,
                ctx=ctx,
                guard_cfg=guard_cfg,
                think=think,
                tools=self.format_tools(sample),
            )

            elapsed = response["elapsed"]

            # A hard abort means the OS corroborated real memory pressure: the
            # model does not fit, so the caller skips its remaining benchmarks.
            # A soft abort costs only this sample and the run continues.
            if response.get("aborted") and response.get("abort_hard"):
                swap_err = response["aborted"]
                for j in range(i, actual_n):
                    result = {
                        "id": sample.get("id", ""),
                        "model": model,
                        "benchmark": self.name,
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "prompt": prompt[:500],
                        "response": "",
                        "elapsed": 0,
                        "tok_per_sec": 0,
                        "error": swap_err,
                        "n_samples": actual_n,
                        "passed": False,
                        "score": 0.0,
                    }
                    results.append(result)
                    if on_sample:
                        on_sample(j + 1, actual_n, result)
                raise MemorySwapAbort(swap_err, partial_results=results)

            scoring = self.score(sample, response["content"], tool_calls=response.get("tool_calls")) if not response["error"] else {"passed": False, "score": 0.0}
            result = {
                "id": sample.get("id", ""),
                "model": model,
                "benchmark": self.name,
                "ts": datetime.now().isoformat(timespec="seconds"),
                "prompt": prompt[:500],
                "response": response["content"][:1000],
                "elapsed": elapsed,
                "tok_per_sec": response["tok_per_sec"],
                "ttft": response.get("ttft"),
                "decode_tps": response.get("decode_tps"),
                "prompt_tokens": response.get("prompt_tokens"),
                "completion_tokens": response.get("completion_tokens"),
                "think": response.get("think"),
                "reasoning_chars": len(response.get("reasoning") or ""),
                "error": response["error"],
                "n_samples": actual_n,
                **scoring,
            }
            results.append(result)
            if on_sample:
                on_sample(i + 1, len(samples), result)
        return results
