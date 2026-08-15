import json
from huggingface_hub import hf_hub_download
from benchmarks.base import BaseBenchmark

# BFCL's own eval package (bfcl-eval) pulls in sglang/vllm/cuda-bindings as
# hard dependencies just to reach its dataset + checker utilities — several
# GB for a benchmarking harness that otherwise has no ML runtime deps at all.
# The dataset itself is plain JSON on HF, and the checker logic (AST-style
# argument comparison against a possible-answer set) is a few hundred lines
# we can own directly, the same way benchmarks/sql.py implements its own
# execution/scoring instead of depending on Spider's reference harness.
_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

# Non-live, single-turn AST categories only (Phase 1 of the ROADMAP.md §E
# scope). Live/exec/multi-turn categories are separate benchmark classes.
_CATEGORIES = ["simple", "multiple", "parallel", "parallel_multiple"]

SYSTEM = ("You are a function-calling assistant. Given a user request and a set of "
          "available functions, call the function(s) needed to satisfy the request. "
          "Only call a function when one is actually needed.")


def _bfcl_type_to_json_schema_type(bfcl_type: str) -> str:
    # BFCL's function schemas use "dict"/"any" where JSON Schema (and Ollama's
    # /api/chat tools param) expects "object"/omitting the type entirely.
    return {"dict": "object", "any": "string"}.get(bfcl_type, bfcl_type)


def _convert_parameters(params: dict) -> dict:
    converted = dict(params)
    converted["type"] = _bfcl_type_to_json_schema_type(params.get("type", "object"))
    props = {}
    for name, spec in (params.get("properties") or {}).items():
        spec = dict(spec)
        if "type" in spec:
            spec["type"] = _bfcl_type_to_json_schema_type(spec["type"])
        props[name] = spec
    converted["properties"] = props
    return converted


def _to_openai_tool(fn: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": _convert_parameters(fn.get("parameters", {})),
        },
    }


def _load_category(category: str) -> list[dict]:
    q_path = hf_hub_download(_REPO, f"BFCL_v3_{category}.json", repo_type="dataset")
    a_path = hf_hub_download(_REPO, f"possible_answer/BFCL_v3_{category}.json", repo_type="dataset")
    with open(q_path) as f:
        questions = {row["id"]: row for row in (json.loads(l) for l in f if l.strip())}
    with open(a_path) as f:
        answers = {row["id"]: row for row in (json.loads(l) for l in f if l.strip())}

    samples = []
    for qid, q in questions.items():
        a = answers.get(qid)
        if a is None:
            continue
        # BFCL's schema nests turns for multi-turn reuse; single-turn categories
        # are always one turn with one user message: question[0][0].
        turn = q["question"][0]
        user_msg = next((m["content"] for m in turn if m["role"] == "user"), "")
        samples.append({
            "id": qid,
            "category": category,
            "prompt": user_msg,
            "tools": [_to_openai_tool(fn) for fn in q["function"]],
            "ground_truth": a["ground_truth"],
        })
    return samples


def _values_match(predicted, acceptable: list) -> bool:
    if predicted in acceptable:
        return True
    # Cross int/float/str type coercion — BFCL's acceptable-value sets are
    # authored somewhat loosely on type (e.g. "10" vs 10), and a model
    # producing a same-value-different-type argument should not fail on that
    # basis alone; the semantic value is what's being tested.
    return str(predicted) in [str(v) for v in acceptable]


def _call_matches_ground_truth(call: dict, gt_entry: dict) -> bool:
    """gt_entry: {func_name: {param_name: [acceptable values]}}, single call."""
    if len(gt_entry) != 1:
        return False
    expected_name, expected_params = next(iter(gt_entry.items()))
    if call.get("name") != expected_name:
        return False
    args = call.get("arguments") or {}
    for param, acceptable in expected_params.items():
        if param not in args:
            # Optional/defaultable param: acceptable list contains "" or None
            # to mean "omitting this is fine".
            if not any(v in ("", None) for v in acceptable):
                return False
            continue
        if not _values_match(args[param], acceptable):
            return False
    return True


class BFCLBenchmark(BaseBenchmark):
    """Single-turn AST categories: simple, multiple, parallel, parallel_multiple.

    Non-live only — BFCL's "live" categories are community-contributed and
    noisier; the curated non-live set is the stable, comparable baseline.
    Scoring is a local reimplementation of BFCL's AST-match logic (function
    name + every ground-truth parameter's value drawn from an acceptable-value
    set), not the official bfcl_eval checker — see the module docstring for why.
    """
    name = "bfcl"
    stratify_key = "category"

    def load_samples(self) -> list[dict]:
        samples = []
        for category in _CATEGORIES:
            samples.extend(_load_category(category))
        return samples

    def system_prompt(self) -> str:
        return SYSTEM

    def format_tools(self, sample: dict) -> list[dict]:
        return sample["tools"]

    def score(self, sample: dict, response: str, tool_calls: list[dict] | None = None) -> dict:
        calls = [
            {"name": tc["function"]["name"], "arguments": tc["function"].get("arguments") or {}}
            for tc in (tool_calls or [])
        ]
        ground_truth = sample["ground_truth"]

        # Greedy multiset match: each expected call must be satisfied by a
        # distinct predicted call (parallel categories expect >1 calls, order
        # does not matter, but count and each call's arguments do).
        remaining = list(calls)
        matched = 0
        for gt_entry in ground_truth:
            for i, call in enumerate(remaining):
                if _call_matches_ground_truth(call, gt_entry):
                    matched += 1
                    remaining.pop(i)
                    break

        passed = matched == len(ground_truth) and len(calls) == len(ground_truth)
        return {
            "passed": passed,
            "score": float(passed),
            "category": sample["category"],
            "predicted_calls": calls,
            "expected_calls": ground_truth,
        }
