import re
from datasets import load_dataset
from benchmarks.base import BaseBenchmark


SYSTEM = "You are a math tutor. Solve the problem step by step. At the very end of your response, write the final numeric answer on its own line prefixed with '####'."


def extract_number(text: str) -> str | None:
    # GSM8K ground truth format: #### 42
    match = re.search(r'####\s*([\d,]+)', text)
    if match:
        return match.group(1).replace(",", "").strip()
    # Fallback: last number in response
    numbers = re.findall(r'\b\d[\d,]*\b', text)
    return numbers[-1].replace(",", "") if numbers else None


class GSM8KBenchmark(BaseBenchmark):
    name = "gsm8k"

    def load_samples(self) -> list[dict]:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [
            {"id": f"gsm8k_{i}", "prompt": row["question"], "answer": extract_number(row["answer"])}
            for i, row in enumerate(ds)
        ]

    def system_prompt(self) -> str:
        return SYSTEM

    def score(self, sample: dict, response: str) -> dict:
        predicted = extract_number(response)
        passed = predicted is not None and predicted == sample["answer"]
        return {"passed": passed, "score": float(passed), "predicted": predicted, "expected": sample["answer"]}
