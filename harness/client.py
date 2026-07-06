import time
from openai import OpenAI


class OllamaClient:
    def __init__(self, base_url: str, api_key: str = "ollama", timeout: int = 120):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def complete_streaming(self, model: str, prompt: str, system: str = None, max_tokens: int = 2048, temperature: float = 0.0) -> dict:
        """Streaming completion. Records TTFT and decode TPS separately from prefill."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        ttft = None
        parts = []
        prompt_tokens = 0
        completion_tokens = 0

        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens or 0
                    completion_tokens = chunk.usage.completion_tokens or 0
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    parts.append(delta.content)

            end = time.perf_counter()
            total_elapsed = end - start
            decode_elapsed = max(total_elapsed - (ttft or 0), 1e-6)
            content = "".join(parts)
            if completion_tokens == 0:
                completion_tokens = max(1, len(content) // 4)

            return {
                "content": content,
                "ttft": ttft,
                "elapsed": total_elapsed,
                "decode_elapsed": decode_elapsed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tok_per_sec": completion_tokens / total_elapsed if total_elapsed > 0 else 0,
                "decode_tps": completion_tokens / decode_elapsed,
                "error": None,
            }
        except Exception as e:
            end = time.perf_counter()
            return {
                "content": "", "ttft": None, "elapsed": end - start,
                "decode_elapsed": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "tok_per_sec": 0, "decode_tps": 0, "error": str(e),
            }

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
