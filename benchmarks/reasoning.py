import re
from datasets import load_dataset
from benchmarks.base import BaseBenchmark


CHOICES = ["A", "B", "C", "D"]

SYSTEM = "You are a knowledgeable assistant. Answer multiple choice questions by responding with only the letter of the correct answer (A, B, C, or D)."

# Explicit answer declarations, tried before falling back to a bare letter.
_DECLARED = [
    re.compile(r"\\boxed\{\s*\*{0,2}([A-D1-4])\b", re.I),
    re.compile(r"\b(?:answer|option|choice)\b[^A-D1-4\n]{0,20}?\*{0,2}([A-D1-4])\b", re.I),
    re.compile(r"\*\*\s*([A-D1-4])\s*\*\*"),
]


def extract_choice(response: str, valid: str = "ABCD") -> str:
    """Pull the model's chosen letter out of a free-form response.

    A terse "B" is unambiguous, but models often reason at length and restate
    the options ("A. 3\nB. 4..."), so taking the *first* letter in the text
    picks up the question's own option labels and scores a correct answer as
    wrong. Prefer an explicit declaration, then the *last* standalone letter,
    which is where a conclusion lands. Identical to first-match on terse
    answers, so previously-scored terse responses are unaffected.
    """
    if not response:
        return ""
    text = response.strip()
    pattern = f"[{valid}]"
    for rx in _DECLARED:
        found = rx.findall(text)
        if found:
            candidate = found[-1].upper()
            if candidate in valid:
                return candidate
    matches = re.findall(rf"\b({pattern})\b", text.upper())
    return matches[-1] if matches else ""


class MMLUBenchmark(BaseBenchmark):
    name = "mmlu"
    stratify_key = "subject"   # balance across the configured subjects

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

    def score(self, sample: dict, response: str, tool_calls: list[dict] | None = None) -> dict:
        predicted = extract_choice(response, "ABCD")
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

    def score(self, sample: dict, response: str, tool_calls: list[dict] | None = None) -> dict:
        predicted = extract_choice(response, "ABCD1234")
        passed = predicted == sample["answer"].upper()
        return {"passed": passed, "score": float(passed), "predicted": predicted, "expected": sample["answer"]}
