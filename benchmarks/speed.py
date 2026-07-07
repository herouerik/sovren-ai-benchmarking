"""Speed benchmark: measures TTFT, decode TPS, and prefill throughput.

Unlike accuracy benchmarks, this has no ground truth. Each probe is run
n_runs times and the median is reported, to reduce cold-start noise.

Probes:
  ttft_baseline   — short prompt, 1-word answer.  Isolates TTFT with minimal prefill.
  decode_throughput — short prompt, long forced output.  Isolates decode token rate.
  prefill_speed   — long prompt (~350 tok), short answer.  Isolates prefill/encode speed.
  realistic_task  — medium prompt + medium output, typical of an agentic code call.
"""

from benchmarks.base import BaseBenchmark

# ~350-token context for the prefill probe
_PREFILL_CONTEXT = """\
Here is a Python module for a thread-safe LRU cache:

```python
import threading
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: int) -> int:
        with self._lock:
            if key not in self.cache:
                return -1
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: int, value: int) -> None:
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def peek(self, key: int) -> int:
        with self._lock:
            return self.cache.get(key, -1)

    def resize(self, new_capacity: int) -> None:
        with self._lock:
            self.capacity = new_capacity
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def keys(self) -> list:
        with self._lock:
            return list(self.cache.keys())

    def clear(self) -> None:
        with self._lock:
            self.cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self.cache)

    def __contains__(self, key: int) -> bool:
        with self._lock:
            return key in self.cache
```

In one sentence only, what is the time complexity of the `put` method?"""

PROBES = [
    {
        "id": "ttft_baseline",
        "label": "TTFT baseline",
        "description": "Short prompt, 1-word answer. Minimises prefill to isolate time-to-first-token.",
        "prompt": "Reply with a single word — yes or no. Are dolphins mammals?",
        "max_tokens": 5,
    },
    {
        "id": "decode_throughput",
        "label": "Decode throughput",
        "description": "Short prompt, long forced output. Minimises prefill to isolate raw decode token rate.",
        "prompt": "Count from 1 to 200, one number per line. Begin with 1.",
        "max_tokens": 600,
    },
    {
        "id": "prefill_speed",
        "label": "Prefill speed",
        "description": "Long prompt (~350 tokens), one-sentence answer. Isolates how fast the model encodes input context.",
        "prompt": _PREFILL_CONTEXT,
        "max_tokens": 30,
    },
    {
        "id": "realistic_task",
        "label": "Realistic agentic call",
        "description": "Medium prompt + medium output, representative of a typical coding-assistant turn.",
        "prompt": "Write a Python function `top_k_frequent(nums: list[int], k: int) -> list[int]` that returns the k most frequent elements. Include a docstring and type hints. No explanation needed, just the function.",
        "max_tokens": 350,
    },
]


def _median(values: list[float]) -> float | None:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    n = len(clean)
    return clean[n // 2] if n % 2 else (clean[n // 2 - 1] + clean[n // 2]) / 2


class SpeedBenchmark(BaseBenchmark):
    name = "speed"

    def load_samples(self) -> list[dict]:
        return PROBES

    def system_prompt(self) -> str | None:
        return None

    def score(self, sample: dict, response: str) -> dict:
        # No accuracy signal — just check the model responded at all
        return {"passed": bool(response.strip()), "score": 1.0 if response.strip() else 0.0}

    def run(self, model: str, n_samples: int = None, on_sample=None, ctx: int | None = None) -> list[dict]:
        probes = self.load_samples()
        n_runs = self.config.get("n_runs", 3)
        results = []

        for probe in probes:
            runs = []
            for _ in range(n_runs):
                resp = self.client.complete_streaming(
                    model=model,
                    prompt=probe["prompt"],
                    system=None,
                    max_tokens=probe.get("max_tokens", 200),
                    temperature=0.0,
                    ctx=ctx,
                )
                if not resp["error"] and resp["ttft"] is not None:
                    runs.append(resp)

            if not runs:
                results.append({
                    "id": probe["id"], "model": model, "benchmark": "speed",
                    "prompt": probe["prompt"][:300], "response": "",
                    "elapsed": 0, "tok_per_sec": 0, "error": "all runs failed",
                    "passed": False, "score": 0.0,
                    "probe": probe["id"], "label": probe["label"],
                    "ttft": None, "decode_tps": None,
                    "prompt_tokens": None, "completion_tokens": None, "n_runs": 0,
                })
                continue

            results.append({
                "id": probe["id"],
                "model": model,
                "benchmark": "speed",
                "prompt": probe["prompt"][:300],
                "response": runs[-1]["content"][:600],
                "elapsed": _median([r["elapsed"] for r in runs]),
                "tok_per_sec": _median([r["tok_per_sec"] for r in runs]),
                "error": None,
                "passed": True,
                "score": 1.0,
                # Speed-specific fields
                "probe": probe["id"],
                "label": probe["label"],
                "description": probe["description"],
                "ttft": _median([r["ttft"] for r in runs]),
                "decode_tps": _median([r["decode_tps"] for r in runs]),
                "prompt_tokens": _median([r["prompt_tokens"] for r in runs]),
                "completion_tokens": _median([r["completion_tokens"] for r in runs]),
                "n_runs": len(runs),
            })

        return results
