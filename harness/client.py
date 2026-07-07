import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from openai import OpenAI


def resolve_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} patterns with environment variable values."""
    def _replace(m: re.Match) -> str:
        var = m.group(1)
        val = os.environ.get(var)
        if val is None:
            raise ValueError(f"Environment variable ${var} is not set (required for {m.string!r})")
        return val
    return re.sub(r'\$\{(\w+)\}', _replace, value) if isinstance(value, str) else value


class OllamaClient:
    def __init__(self, base_url: str, api_key: str = "ollama", timeout: int = 120):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def _build_ctx_kwargs(self, ctx: int | None, kwargs: dict) -> dict:
        if ctx is None:
            return kwargs
        extra = kwargs.get("extra_body", {})
        extra["options"] = {**(extra.get("options", {})), "num_ctx": ctx}
        kwargs["extra_body"] = extra
        return kwargs

    def complete_streaming(self, model: str, prompt: str, system: str = None, max_tokens: int = 2048, temperature: float = 0.0, ctx: int | None = None) -> dict:
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
            kwargs = self._build_ctx_kwargs(ctx, {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                "stream_options": {"include_usage": True},
            })
            stream = self.client.chat.completions.create(**kwargs)
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

    def complete(self, model: str, prompt: str, system: str = None, max_tokens: int = 2048, temperature: float = 0.0, ctx: int | None = None) -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        try:
            kwargs = self._build_ctx_kwargs(ctx, {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            })
            response = self.client.chat.completions.create(**kwargs)
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


def build_client(cfg: dict) -> OllamaClient:
    """Build an OllamaClient from a config dict (supports env-var substitution)."""
    raw_key = cfg.get("api_key", "ollama")
    api_key = resolve_env_vars(raw_key) if raw_key else raw_key
    return OllamaClient(
        base_url=resolve_env_vars(cfg.get("base_url", "http://localhost:11434/v1")),
        api_key=api_key,
        timeout=cfg.get("timeout", 120),
    )


# ── OpenCode CLI judge client ──────────────────────────────────────────────────


def _find_opencode() -> str:
    """Locate the opencode binary."""
    path = shutil.which("opencode")
    if not path:
        for candidate in [
            Path.home() / ".opencode" / "bin" / "opencode",
            Path.home() / ".local" / "bin" / "opencode",
        ]:
            if candidate.exists():
                return str(candidate)
    return path or "opencode"


class OpenCodeClient:
    """Wraps the opencode CLI subprocess to make LLM calls (free, no API key needed)."""

    def __init__(self, model: str = "opencode/deepseek-v4-flash-free", timeout: int = 120):
        self.bin = _find_opencode()
        self.model = model
        self.timeout = timeout

    def complete(self, model: str = None, prompt: str = "", system: str = None,
                 max_tokens: int = 2048, temperature: float = 0.0) -> dict:
        """Call opencode CLI as subprocess and return completion."""
        start = time.time()
        # Prepend system instruction to prompt (opencode CLI has no --system flag)
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        try:
            cmd = [self.bin, "run", "--model", model or self.model,
                   "--format", "json", full_prompt]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            elapsed = time.time() - start
            if result.returncode != 0:
                return {"content": "", "elapsed": elapsed, "tokens": 0,
                        "tok_per_sec": 0, "error": result.stderr[:500]}

            # Parse newline-delimited JSON events from stdout
            content_parts = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    if event.get("type") == "error":
                        err_msg = str(event.get("error", {}).get("data", {}).get("message", "unknown"))
                        return {"content": "", "elapsed": elapsed, "tokens": 0,
                                "tok_per_sec": 0, "error": err_msg}
                    if event.get("type") == "text":
                        text = event.get("part", {}).get("text", "")
                        if text:
                            content_parts.append(text)

            content = "".join(content_parts).strip()
            if not content:
                return {"content": "", "elapsed": elapsed, "tokens": 0,
                        "tok_per_sec": 0, "error": "empty response"}
            tokens = len(content) // 4
            return {
                "content": content,
                "elapsed": elapsed,
                "tokens": tokens,
                "tok_per_sec": tokens / elapsed if elapsed > 0 else 0,
                "error": None,
            }
        except subprocess.TimeoutExpired:
            return {"content": "", "elapsed": self.timeout, "tokens": 0,
                    "tok_per_sec": 0, "error": f"timeout ({self.timeout}s)"}
        except FileNotFoundError:
            return {"content": "", "elapsed": 0, "tokens": 0,
                    "tok_per_sec": 0, "error": "opencode binary not found in PATH"}
        except Exception as e:
            elapsed = time.time() - start
            return {"content": "", "elapsed": elapsed, "tokens": 0,
                    "tok_per_sec": 0, "error": str(e)}
