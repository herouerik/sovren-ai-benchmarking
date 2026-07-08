import re
import textwrap
from datasets import load_dataset
from benchmarks.base import BaseBenchmark
from harness.sandbox import execute_python


SYSTEM = "You are an expert Python programmer. Write clean, correct Python code. Return ONLY the code with no explanation or markdown fences."


def extract_code(response: str) -> str:
    # Strip markdown fences if present
    match = re.search(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def normalize_completion_indent(code: str) -> str:
    """Re-indent a function body (no def line) to consistent 4-space levels.

    Completion models like codestral often use inconsistent indentation
    (e.g. 1-space outer, 7-space inner). We reconstruct levels with a stack
    that tracks seen indent values, then emit 4*level spaces per line.
    """
    lines = code.expandtabs(4).splitlines()
    non_empty = [(len(l) - len(l.lstrip()), l.lstrip()) for l in lines if l.strip()]
    if not non_empty:
        return code

    level_stack = [(non_empty[0][0], 0)]
    level_map = {non_empty[0][0]: 0}

    for indent, _ in non_empty[1:]:
        top_indent, top_level = level_stack[-1]
        if indent > top_indent:
            new_level = top_level + 1
            level_stack.append((indent, new_level))
            level_map[indent] = new_level
        elif indent < top_indent:
            while len(level_stack) > 1 and level_stack[-1][0] > indent:
                level_stack.pop()
            nearest_indent, nearest_level = min(level_stack, key=lambda x: abs(x[0] - indent))
            level_map.setdefault(indent, nearest_level)
            if not any(e[0] == indent for e in level_stack):
                level_stack.append((indent, level_map[indent]))

    result = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            result.append("")
            continue
        indent = len(line) - len(stripped)
        level = level_map.get(indent, 0)
        result.append("    " * (level + 1) + stripped)  # +1: body is one level inside def

    return "\n".join(result)


class HumanEvalBenchmark(BaseBenchmark):
    name = "humaneval"

    def load_samples(self) -> list[dict]:
        ds = load_dataset("openai/openai_humaneval", split="test")
        return [
            {
                "id": row["task_id"],
                "prompt": row["prompt"],
                "entry_point": row["entry_point"],
                "test_code": row["test"] + f"\ncheck({row['entry_point']})",
                "canonical": row["canonical_solution"],
            }
            for row in ds
        ]

    def system_prompt(self) -> str:
        return SYSTEM

    def score(self, sample: dict, response: str) -> dict:
        code = extract_code(response)
        entry = sample.get("entry_point", "")
        # Completion-style models (e.g. codestral) return only the function body
        # without the def line and often with inconsistent indentation.
        if entry and f"def {entry}" not in code:
            body = normalize_completion_indent(code)
            code = sample["prompt"] + "\n" + body
        result = execute_python(
            code,
            sample["test_code"],
            timeout=self.config.get("execution_timeout", 10),
        )
        return {
            "passed": result["passed"],
            "score": float(result["passed"]),
            "exec_error": result.get("stderr", "").strip()[:200] or result.get("error"),
            "stderr": result.get("stderr", "")[:500],
        }


class MBPPBenchmark(BaseBenchmark):
    name = "mbpp"

    def load_samples(self) -> list[dict]:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        samples = []
        for row in ds:
            # Extract function signature from the reference solution so the
            # model knows what function name and arguments the tests expect.
            # Without this the model guesses wrong names and scores 0%.
            code = row["code"]
            sig = code.split(":")[0] + ":"  # e.g. "def remove_Occ(s,ch):"
            prompt = f"{row['prompt']}\n\n# Define function\n{sig}"
            test_code = "\n".join(row["test_list"])
            test_setup = "\n".join(row["test_imports"]) if row["test_imports"] else ""
            if test_setup:
                test_code = f"{test_setup}\n{test_code}"
            samples.append({
                "id": f"mbpp_{row['task_id']}",
                "prompt": prompt,
                "test_code": test_code,
            })
        return samples

    def system_prompt(self) -> str:
        return SYSTEM

    def score(self, sample: dict, response: str) -> dict:
        code = extract_code(response)
        result = execute_python(
            code,
            sample["test_code"],
            timeout=self.config.get("execution_timeout", 10),
        )
        return {
            "passed": result["passed"],
            "score": float(result["passed"]),
            "exec_error": result.get("error"),
            "stderr": result.get("stderr", "")[:500],
        }
