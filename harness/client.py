import time
from openai import OpenAI


class OllamaClient:
    def __init__(self, base_url: str, api_key: str = "ollama", timeout: int = 120):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def complete(self, model: str, prompt: str, system: str = None, max_tokens: int = 2048, temperature: float = 0.0) -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            elapsed = time.time() - start
            content = response.choices[0].message.content or ""
            tokens = response.usage.completion_tokens if response.usage else 0
            return {
                "content": content,
                "elapsed": elapsed,
                "tokens": tokens,
                "tok_per_sec": tokens / elapsed if elapsed > 0 else 0,
                "error": None,
            }
        except Exception as e:
            return {
                "content": "",
                "elapsed": time.time() - start,
                "tokens": 0,
                "tok_per_sec": 0,
                "error": str(e),
            }
