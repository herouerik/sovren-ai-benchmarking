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

    def _check_memory_swap(self, i: int, elapsed: float, calibration_times: list[float],
                           guard_cfg: dict, n_remaining: int, model: str, bench_name: str) -> str | None:
        """Return an error message if the system is swapping, else None."""
        if not guard_cfg.get("enabled", True):
            return None
        max_baseline = guard_cfg.get("max_baseline_seconds", 30)
        cal_samples = guard_cfg.get("calibration_samples", 3)
        threshold = guard_cfg.get("spike_threshold", 5.0)

        # Only skip early detection for thinking-heavy benchmarks (e.g. philosophical
        # where extended reasoning can legitimately take >30s per sample).
        if not self.allow_thinking:
            # Early detection: if a non-first sample exceeds max_baseline the model is
            # likely swapping. (Sample 0 includes model loading time so it gets a pass.)
            if i > 0 and elapsed > max_baseline:
                remaining = n_remaining - i - 1
                return (f"swap_abort: sample {i+1}/{n_remaining} took {elapsed:.1f}s "
                        f"(exceeds max_baseline_seconds={max_baseline}), "
                        f"{remaining} samples skipped")

        if i < cal_samples:
            return None  # still calibrating

        baseline = _median(calibration_times[:cal_samples])
        if baseline <= 0:
            return None

        # Check if calibration baseline is already abnormal — model barely fits
        if i == cal_samples and baseline > max_baseline:
            remaining = n_remaining - i - 1
            return (f"swap_abort: calibration baseline {baseline:.1f}s exceeds "
                    f"max_baseline_seconds={max_baseline}, "
                    f"model too large for available memory, {remaining} samples skipped")

        # Check for latency spike — system is swapping
        if elapsed > threshold * baseline:
            remaining = n_remaining - i - 1
            return (f"swap_abort: sample {i+1}/{n_remaining} took {elapsed:.1f}s "
                    f"({(elapsed/baseline):.1f}× baseline {baseline:.2f}s), "
                    f"{remaining} samples skipped")

        return None

    def run(self, model: str, n_samples: int = None, on_sample=None, ctx: int | None = None) -> list[dict]:
        samples = self.load_samples()
        if n_samples:
            samples = samples[:n_samples]
        actual_n = len(samples)

        guard_cfg = self.config.get("memory_guard", {})
        calibration_times: list[float] = []

        results = []
        for i, sample in enumerate(samples):
            prompt = self.format_prompt(sample)
            response = self.client.complete(
                model=model,
                prompt=prompt,
                system=self._effective_system(model),
                max_tokens=self.config.get("max_tokens", 2048),
                ctx=ctx,
            )

            elapsed = response["elapsed"]
            calibration_times.append(elapsed)

            swap_err = self._check_memory_swap(i, elapsed, calibration_times, guard_cfg,
                                                actual_n, model, self.name)

            if swap_err:
                # Fill remaining samples with swap-abort records
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

            scoring = self.score(sample, response["content"]) if not response["error"] else {"passed": False, "score": 0.0}
            result = {
                "id": sample.get("id", ""),
                "model": model,
                "benchmark": self.name,
                "ts": datetime.now().isoformat(timespec="seconds"),
                "prompt": prompt[:500],
                "response": response["content"][:1000],
                "elapsed": elapsed,
                "tok_per_sec": response["tok_per_sec"],
                "error": response["error"],
                "n_samples": actual_n,
                **scoring,
            }
            results.append(result)
            if on_sample:
                on_sample(i + 1, len(samples), result)
        return results
