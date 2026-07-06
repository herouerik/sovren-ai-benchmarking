import re
from datasets import load_dataset
from benchmarks.base import BaseBenchmark


CHOICES = ["A", "B", "C", "D"]

SYSTEM = "You are a knowledgeable assistant. Answer multiple choice questions by responding with only the letter of the correct answer (A, B, C, or D)."


class MMLUBenchmark(BaseBenchmark):
    name = "mmlu"

    def load_samples(self) -> list[dict]:
        subjects = self.config.get("subjects", ["abstract_algebra"])
        samples = []
        for subject in subjects:
            ds = load_dataset("cais/mmlu", subject, split="test")
            for i, row in enumerate(ds):
                choices_str = "\n".join(f"{CHOICES[j]}. {row['choices'][j]}" for j in range(len(row["choices"])))
                samples.append({
                    "id": f"mmlu_{subject}_{i}",
                    "subject": subject,
                    "prompt": f"Question: {row['question']}\n\n{choices_str}",
                    "answer": CHOICES[row["answer"]],
                })
        return samples

    def system_prompt(self) -> str:
        return SYSTEM

    def score(self, sample: dict, response: str) -> dict:
        # Extract first A/B/C/D from response
        match = re.search(r'\b([ABCD])\b', response.strip().upper())
        predicted = match.group(1) if match else ""
        passed = predicted == sample["answer"]
        return {"passed": passed, "score": float(passed), "predicted": predicted, "expected": sample["answer"]}


class ARCBenchmark(BaseBenchmark):
    name = "arc"

    def load_samples(self) -> list[dict]:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        samples = []
        for i, row in enumerate(ds):
            labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            choices_str = "\n".join(f"{labels[j]}. {texts[j]}" for j in range(len(labels)))
            samples.append({
                "id": f"arc_{i}",
                "prompt": f"Question: {row['question']}\n\n{choices_str}",
                "answer": row["answerKey"],
            })
        return samples

    def system_prompt(self) -> str:
        return SYSTEM

    def score(self, sample: dict, response: str) -> dict:
        match = re.search(r'\b([ABCD1234])\b', response.strip().upper())
        predicted = match.group(1) if match else ""
        passed = predicted == sample["answer"].upper()
        return {"passed": passed, "score": float(passed), "predicted": predicted, "expected": sample["answer"]}
