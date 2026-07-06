import re
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


class HumanEvalBenchmark(BaseBenchmark):
    name = "humaneval"

    def load_samples(self) -> list[dict]:
        ds = load_dataset("openai/openai_humaneval", split="test")
        return [
            {
                "id": row["task_id"],
                "prompt": row["prompt"],
                "test_code": row["test"] + f"\ncheck({row['entry_point']})",
                "canonical": row["canonical_solution"],
            }
            for row in ds
        ]

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


class MBPPBenchmark(BaseBenchmark):
    name = "mbpp"

    def load_samples(self) -> list[dict]:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        samples = []
        for row in ds:
            test_code = "\n".join(row["test_list"])
            samples.append({
                "id": f"mbpp_{row['task_id']}",
                "prompt": row["text"],
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
