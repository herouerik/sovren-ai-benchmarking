from benchmarks.base import BaseBenchmark
from harness.judge import llm_judge, llm_judge_ensemble


PROMPTS = [
    {"id": "ph_01", "prompt": "Is free will compatible with a deterministic universe? Defend your position with concrete arguments."},
    {"id": "ph_02", "prompt": "Can a society be truly just if it tolerates significant inequality? Where does your reasoning lead?"},
    {"id": "ph_03", "prompt": "Is there an objective moral truth, or is ethics always culturally constructed? Explore the strongest case for each side."},
    {"id": "ph_04", "prompt": "If you could know the exact moment of your death, would it be rational to want to know? What does this reveal about how we value time?"},
    {"id": "ph_05", "prompt": "Does technological progress make humanity more or less free? Ground your answer in specific examples."},
    {"id": "ph_06", "prompt": "Is it ever morally permissible to lie? Construct the strongest argument you can for and against."},
    {"id": "ph_07", "prompt": "What obligations, if any, do the living have toward future generations who do not yet exist?"},
    {"id": "ph_08", "prompt": "Can machines ever truly understand meaning, or only simulate understanding? What hangs on the answer?"},
    {"id": "ph_09", "prompt": "Is suffering necessary for a meaningful life? Explore the question without dismissing either answer too quickly."},
    {"id": "ph_10", "prompt": "What is the relationship between knowledge and power? Is ignorance ever preferable?"},
]


class PhilosophicalBenchmark(BaseBenchmark):
    name = "philosophical"
    allow_thinking = True

    def load_samples(self) -> list[dict]:
        return PROMPTS

    def system_prompt(self) -> str:
        return "You are a thoughtful philosopher. Engage deeply and honestly with the question."

    def score(self, sample: dict, response: str) -> dict:
        criteria = self.config.get("judge_rubric", [
            "Depth of reasoning (1-5)",
            "Coherence and logical consistency (1-5)",
            "Acknowledgment of multiple perspectives (1-5)",
            "Originality of insight (1-5)",
            "Clarity of expression (1-5)",
        ])
        judge_model = self.config.get("judge_model", "llama3.1:8b")
        use_ensemble = self.config.get("use_ensemble", False)

        if use_ensemble:
            models = self.config.get("ensemble_models", [])
            result = llm_judge_ensemble(
                client=self.judge_client,
                models=models,
                question=sample["prompt"],
                response=response,
                criteria=criteria,
            )
        else:
            result = llm_judge(
                client=self.judge_client,
                judge_model=judge_model,
                question=sample["prompt"],
                response=response,
                criteria=criteria,
            )
        mean = result.get("mean_score", 0.0)
        return {
            "passed": mean >= 3.5,
            "score": mean / 5.0,
            "mean_score": mean,
            "judge_scores": result.get("scores", {}),
            "judge_reasoning": result.get("reasoning", ""),
            "judge_error": result.get("error"),
        }

    def run(self, model: str, n_samples: int = None, on_sample=None) -> list[dict]:
        self.config["judge_model"] = self.config.get("judge_model", "llama3.1:8b")
        return super().run(model, n_samples, on_sample=on_sample)
