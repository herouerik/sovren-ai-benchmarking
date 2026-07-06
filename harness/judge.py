import re
from harness.client import OllamaClient


JUDGE_SYSTEM = """You are an impartial evaluator of AI-generated responses.
Score the response on each criterion from 1 (poor) to 5 (excellent).
Respond ONLY with a JSON object like: {"scores": {"criterion": score, ...}, "reasoning": "brief explanation"}"""


def llm_judge(client: OllamaClient, judge_model: str, question: str, response: str, criteria: list[str]) -> dict:
    criteria_str = "\n".join(f"- {c}" for c in criteria)
    prompt = f"""Question asked to the AI:
{question}

AI's response:
{response}

Score this response on these criteria (1-5 each):
{criteria_str}

Return only the JSON object."""

    result = client.complete(judge_model, prompt, system=JUDGE_SYSTEM, temperature=0.0, max_tokens=512)

    if result["error"]:
        return {"scores": {}, "reasoning": "", "error": result["error"]}

    try:
        # Extract JSON even if model wraps it in markdown
        content = result["content"]
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            import json
            parsed = json.loads(match.group())
            return {
                "scores": parsed.get("scores", {}),
                "reasoning": parsed.get("reasoning", ""),
                "mean_score": sum(parsed.get("scores", {}).values()) / max(len(parsed.get("scores", {})), 1),
                "error": None,
            }
    except Exception as e:
        pass

    return {"scores": {}, "reasoning": content, "mean_score": 0.0, "error": "parse_failed"}
