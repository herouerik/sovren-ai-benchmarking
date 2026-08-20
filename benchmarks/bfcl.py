import json
import re
from pathlib import Path

from benchmarks.base import BaseBenchmark

# BFCL v4. Two things worth knowing about where this data comes from:
#
# The checker logic (AST-style argument comparison against a possible-answer
# set) is a few hundred lines we own directly, the same way benchmarks/sql.py
# implements its own execution/scoring instead of depending on Spider's
# reference harness. The original reason was that bfcl-eval pulled in
# sglang/vllm/cuda-bindings — no longer true as of 2026.3.23, whose wheel is
# 1.9MB with no CUDA, but it still drags in four vendor LLM SDKs this harness
# never calls, so the local checker stays.
#
# The dataset, however, is no longer on HF: that repo still carries only the
# BFCL_v3_* files, and v4 ships exclusively inside the bfcl-eval wheel. It is
# extracted to data/bfcl_v4/ by prefetch_datasets.download_bfcl_v4().
#
# For the four AST categories below, v4 is v3 with four corrected ground
# truths and two corrected question texts out of 1000 items — verified by an
# ID-aligned diff of every field. The fixes are real bugs: simple_363's
# ground truth omitted the "restaurant_search." namespace, so a model calling
# the function correctly was scored wrong; parallel_multiple_141 expected the
# month "Febuary". None of the corrected items fall inside the seed-42 n=100
# draw, so the v3 -> v4 move leaves every score already in results/ intact.
_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bfcl_v4"

# v4 renamed the Python AST set to "simple_python" (it split out simple_java
# and simple_javascript). The category name is kept as "simple" so the
# stratify buckets, and therefore the seeded sample selection, are unchanged.
_CATEGORY_FILES = {
    "simple": "simple_python",
    "multiple": "multiple",
    "parallel": "parallel",
    "parallel_multiple": "parallel_multiple",
    "irrelevance": "irrelevance",
}

# Non-live, single-turn AST categories only (Phase 1 of the ROADMAP.md §E
# scope). Live/exec/multi-turn categories are separate benchmark classes.
# `irrelevance` is scored the other way round and lives in its own benchmark
# (BFCLIrrelevanceBenchmark) rather than being folded in here — mixing it
# into this score would silently redefine a metric already published for
# seven models at n=100.
_CATEGORIES = ["simple", "multiple", "parallel", "parallel_multiple"]

_ID_NUM = re.compile(r"(\d+)$")


def _read(rel: str) -> list[dict]:
    path = _DATA_DIR / rel
    if not path.exists():
        raise FileNotFoundError(
            f"BFCL v4 data missing: {path}. Run `python prefetch_datasets.py` "
            f"(it extracts the v4 dataset from the bfcl-eval wheel).")
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _sort_key(row: dict) -> int:
    """Numeric ID order, independent of how the upstream file happens to be laid out.

    The seeded sample selection in BaseBenchmark.select_samples shuffles each
    stratum in arrival order, so a reordered upstream file silently changes
    which items every model is scored on. Both v3 and v4 ship ascending
    already; sorting makes that a guarantee rather than an accident.
    """
    m = _ID_NUM.search(row["id"])
    return int(m.group(1)) if m else 0


SYSTEM = ("You are a function-calling assistant. Given a user request and a set of "
          "available functions, call the function(s) needed to satisfy the request. "
          "Only call a function when one is actually needed.")


def _bfcl_type_to_json_schema_type(bfcl_type: str) -> str:
    # BFCL's function schemas use "dict"/"any"/"float"/"tuple" where JSON
    # Schema (and Ollama's /api/chat tools param, which compiles this into a
    # constrained-generation grammar) expects "object"/"string"/"number"/
    # "array". An unmapped type isn't just ignored — some backends (qwen3.8's
    # GGUF pull, not gemma4 or qwen3.6 on the same request) hard-reject the
    # whole call with an HTTP 400 "Unrecognized schema", which the streaming
    # client then silently turned into a fake empty-but-successful response
    # (see harness/client.py's status-check fix) rather than a visible error.
    # That one unmapped type ("float") was responsible for a 26-point bfcl
    # gap on qwen3.8 alone, affecting ~23% of samples across all 4 categories.
    return {"dict": "object", "any": "string", "float": "number", "tuple": "array"}.get(bfcl_type, bfcl_type)


def _convert_parameters(params: dict) -> dict:
    converted = dict(params)
    converted["type"] = _bfcl_type_to_json_schema_type(params.get("type", "object"))
    props = {}
    for name, spec in (params.get("properties") or {}).items():
        spec = dict(spec)
        if "type" in spec:
            spec["type"] = _bfcl_type_to_json_schema_type(spec["type"])
        if "items" in spec and isinstance(spec["items"], dict) and "type" in spec["items"]:
            spec["items"] = dict(spec["items"])
            spec["items"]["type"] = _bfcl_type_to_json_schema_type(spec["items"]["type"])
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


def _canonical_id(raw: str) -> str:
    """simple_python_7 -> simple_7.

    Keeps per-sample record IDs comparable against the runs already in
    results/, which were collected under the v3 naming.
    """
    return raw.replace("simple_python_", "simple_")


def _load_category(category: str) -> list[dict]:
    stem = _CATEGORY_FILES[category]
    rows = sorted(_read(f"BFCL_v4_{stem}.json"), key=_sort_key)
    questions = {_canonical_id(r["id"]): r for r in rows}
    # irrelevance has no possible_answer file: the correct behaviour is to call
    # nothing, so there is no ground-truth call to compare against.
    answers = {}
    if category != "irrelevance":
        answers = {_canonical_id(r["id"]): r
                   for r in _read(f"possible_answer/BFCL_v4_{stem}.json")}

    samples = []
    for qid, q in questions.items():
        a = answers.get(qid)
        if a is None and category != "irrelevance":
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
            "ground_truth": a["ground_truth"] if a else [],
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


IRRELEVANCE_SYSTEM = (
    "You are a function-calling assistant. Given a user request and a set of "
    "available functions, call the function(s) needed to satisfy the request. "
    "If none of the available functions can satisfy the request, do not call "
    "any function — answer in plain text instead."
)


class BFCLIrrelevanceBenchmark(BaseBenchmark):
    """Non-live irrelevance: 240 requests no available function can satisfy.

    The inverse of every other benchmark in this suite — a pass means the
    model called *nothing*. Nothing else here penalises over-calling, so a
    model that fires a tool at every prompt scores identically to one with
    judgment. Needle 2 is the worked example: asked for a joke about
    databases with only a weather tool in scope, it emitted a call.

    Kept separate from BFCLBenchmark rather than added as a fifth category:
    folding it in would change what the published `bfcl` number means for the
    seven models already measured at n=100.
    """
    name = "bfcl_irrelevance"

    def load_samples(self) -> list[dict]:
        return _load_category("irrelevance")

    def system_prompt(self) -> str:
        # Deliberately spells out the opt-out. Without it the prompt is a
        # trick question: a model with only one irrelevant tool in scope has
        # no way to know that declining is an allowed move, and the score
        # measures prompt wording rather than judgment.
        return IRRELEVANCE_SYSTEM

    def format_tools(self, sample: dict) -> list[dict]:
        return sample["tools"]

    def score(self, sample: dict, response: str, tool_calls: list[dict] | None = None) -> dict:
        calls = [
            {"name": tc["function"]["name"], "arguments": tc["function"].get("arguments") or {}}
            for tc in (tool_calls or [])
        ]
        passed = not calls
        return {
            "passed": passed,
            "score": float(passed),
            "predicted_calls": calls,
            # No expected_calls key with content: the expectation is the empty
            # set, and emitting [] here keeps the record shape identical to
            # BFCLBenchmark's so both render in the same dashboard column.
            "expected_calls": [],
        }
