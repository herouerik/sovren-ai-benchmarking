import ast
import json
from huggingface_hub import hf_hub_download

from benchmarks.base import BaseBenchmark
from benchmarks.bfcl import _to_openai_tool
from benchmarks.bfcl_multi_turn_envs import CLASS_REGISTRY

# Phase 2 of the BFCL build (docs/ROADMAP.md §E): multi-turn, state-based
# scoring. Long-Context is implemented first — it is the subcategory that
# actually stresses what this GPU pool's large-context models were bisected
# for; Base/Missing-Functions/Missing-Parameters share this exact engine and
# only need their own BENCHMARK_REGISTRY entry (same class, different
# category constant) once this is validated.
_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
_CATEGORY = "multi_turn_long_context"

_FUNC_DOC_FILES = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}

# A turn legitimately needs several sequential tool calls (e.g. cd, then mv,
# then grep). This is a runaway cap against a model that loops on a bad call,
# not a target — 20 calls is generous for any single BFCL turn.
_MAX_CALLS_PER_TURN = 20

SYSTEM = ("You are a function-calling assistant with access to a stateful environment "
          "across multiple tools. Call the function(s) needed to satisfy each request. "
          "When you have finished the requested action, respond with plain text "
          "confirming what was done — do not call a function unless one is needed.")

_func_doc_cache: dict[str, list[dict]] = {}


def _load_func_doc(class_name: str) -> list[dict]:
    if class_name not in _func_doc_cache:
        path = hf_hub_download(_REPO, f"multi_turn_func_doc/{_FUNC_DOC_FILES[class_name]}",
                                repo_type="dataset")
        with open(path) as f:
            _func_doc_cache[class_name] = [json.loads(l) for l in f if l.strip()]
    return _func_doc_cache[class_name]


def _parse_call(call_str: str) -> tuple[str, dict]:
    """Parse a BFCL ground-truth call string, e.g. "cd(folder='document')",
    into (function_name, kwargs). ast-based rather than eval() since this
    runs against every ground-truth record on every benchmark run."""
    node = ast.parse(call_str.strip(), mode="eval").body
    if not isinstance(node, ast.Call):
        raise ValueError(f"not a call expression: {call_str}")
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    return node.func.id, kwargs


def _instantiate(involved_classes: list[str], initial_config: dict) -> dict:
    instances = {}
    for class_name in involved_classes:
        inst = CLASS_REGISTRY[class_name]()
        if hasattr(inst, "_load_scenario"):
            inst._load_scenario(initial_config.get(class_name, {}), long_context=True)
        instances[class_name] = inst
    return instances


def _find_owner(instances: dict, func_name: str):
    for inst in instances.values():
        if hasattr(inst, func_name):
            return inst
    return None


def _execute(instances: dict, func_name: str, kwargs: dict):
    owner = _find_owner(instances, func_name)
    if owner is None:
        return {"error": f"no available function named '{func_name}'"}
    try:
        return getattr(owner, func_name)(**kwargs)
    except Exception as e:
        return {"error": str(e)}


def _attr_equal(x, y) -> bool:
    # 4 of the 8 vendored classes store a seeded random.Random instance as
    # state (self._random), used for scenario values like generated IDs and
    # sensor readings. random.Random has no __eq__, so two separately
    # instantiated objects compare unequal by identity even when seeded
    # identically and consumed in the exact same sequence — comparing
    # .getstate() instead checks what's actually observable: did the two
    # runs draw the same number of random values in the same order.
    import random
    if isinstance(x, random.Random) and isinstance(y, random.Random):
        return x.getstate() == y.getstate()
    return x == y


def _state_diff(a: dict, b: dict) -> dict:
    """{class_name: {attr: (model_value, ground_truth_value)}} for every
    mismatching attribute — empty dict means the states match. Reimplementation
    choice, documented same as benchmarks/bfcl.py: compares full instance
    __dict__ rather than the official checker's per-class relevant-attribute
    subset. Stricter in some edge cases, but never silently permissive."""
    diff = {}
    for class_name in a:
        va, vb = vars(a[class_name]), vars(b[class_name])
        mismatches = {k: (va.get(k), vb.get(k)) for k in va.keys() | vb.keys()
                      if k not in va or k not in vb or not _attr_equal(va[k], vb[k])}
        if mismatches:
            diff[class_name] = mismatches
    return diff


class BFCLMultiTurnLongContextBenchmark(BaseBenchmark):
    """Multi-turn function calling under injected long-context state padding.

    Cannot use BaseBenchmark.run()'s single-shot loop — each sample is a
    sequence of turns against a live, mutating environment, and scoring
    compares final environment state rather than parsing one text response.
    """
    name = "bfcl_multi_turn_long_context"

    def load_samples(self) -> list[dict]:
        q_path = hf_hub_download(_REPO, f"BFCL_v3_{_CATEGORY}.json", repo_type="dataset")
        a_path = hf_hub_download(_REPO, f"possible_answer/BFCL_v3_{_CATEGORY}.json", repo_type="dataset")
        with open(q_path) as f:
            questions = {row["id"]: row for row in (json.loads(l) for l in f if l.strip())}
        with open(a_path) as f:
            answers = {row["id"]: row for row in (json.loads(l) for l in f if l.strip())}

        samples = []
        for qid, q in questions.items():
            a = answers.get(qid)
            if a is None:
                continue
            samples.append({
                "id": qid,
                "turns": q["question"],
                "initial_config": q["initial_config"],
                "involved_classes": q["involved_classes"],
                "ground_truth": a["ground_truth"],
            })
        return samples

    def format_prompt(self, sample: dict) -> str:
        # Unused — run() is overridden and never calls complete_native().
        return sample["turns"][0][0]["content"]

    def score(self, sample: dict, response: str, tool_calls: list[dict] | None = None) -> dict:
        # Unused for the same reason; scoring happens inline in run() since it
        # needs the live instances, not just one response.
        raise NotImplementedError

    def run(self, model: str, n_samples: int = None, on_sample=None, ctx: int | None = None) -> list[dict]:
        import time
        from datetime import datetime

        samples = self.select_samples(self.load_samples(), n_samples)
        guard_cfg = self.config.get("memory_guard", {})
        max_tokens = self.config.get("max_tokens", 4096)

        results = []
        for i, sample in enumerate(samples):
            gt_instances = _instantiate(sample["involved_classes"], sample["initial_config"])
            for turn_calls in sample["ground_truth"]:
                for call_str in turn_calls:
                    func_name, kwargs = _parse_call(call_str)
                    _execute(gt_instances, func_name, kwargs)

            model_instances = _instantiate(sample["involved_classes"], sample["initial_config"])
            tools = [_to_openai_tool(fn) for cls in sample["involved_classes"]
                     for fn in _load_func_doc(cls)]
            messages = [{"role": "system", "content": SYSTEM}]

            start = time.perf_counter()
            total_calls = 0
            error = None
            for turn in sample["turns"]:
                messages.extend(turn)
                for _ in range(_MAX_CALLS_PER_TURN):
                    response = self.client.complete_chat(
                        model=model, messages=messages, tools=tools,
                        max_tokens=max_tokens, ctx=ctx, guard_cfg=guard_cfg,
                    )
                    if response["error"]:
                        error = response["error"]
                        break
                    calls = response.get("tool_calls") or []
                    if not calls:
                        messages.append({"role": "assistant", "content": response["content"]})
                        break
                    messages.append({"role": "assistant", "content": response["content"],
                                      "tool_calls": calls})
                    for tc in calls:
                        func_name = tc["function"]["name"]
                        kwargs = tc["function"].get("arguments") or {}
                        result = _execute(model_instances, func_name, kwargs)
                        messages.append({"role": "tool", "content": json.dumps(result, default=str)})
                        total_calls += 1
                else:
                    error = f"exceeded {_MAX_CALLS_PER_TURN} tool calls in one turn"
                if error:
                    break

            elapsed = time.perf_counter() - start
            diff = {} if error else _state_diff(model_instances, gt_instances)
            passed = error is None and not diff
            predicted_calls = [
                f"{tc['function']['name']}({tc['function'].get('arguments') or {}})"
                for m in messages if m.get("role") == "assistant"
                for tc in (m.get("tool_calls") or [])
            ]
            result = {
                "id": sample["id"],
                "model": model,
                "benchmark": self.name,
                "ts": datetime.now().isoformat(timespec="seconds"),
                "prompt": sample["turns"][0][0]["content"][:500],
                "response": "",
                "elapsed": elapsed,
                "tok_per_sec": 0,
                "error": error,
                "n_samples": len(samples),
                "passed": passed,
                "score": float(passed),
                "num_turns": len(sample["turns"]),
                "num_tool_calls": total_calls,
                "involved_classes": sample["involved_classes"],
                "predicted_calls": predicted_calls,
                "expected_calls": [c for turn in sample["ground_truth"] for c in turn],
                "state_diff": diff,
            }
            results.append(result)
            if on_sample:
                on_sample(i + 1, len(samples), result)
        return results
