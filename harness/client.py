import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from openai import OpenAI

from harness.guard import StreamWatchdog


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
        # Ollama's native endpoint, derived from the OpenAI-compat base_url.
        # Benchmark inference goes here rather than /v1 because /v1 silently
        # ignores `think`, and thinking materially changes benchmark results —
        # it has to be an explicit, recorded dimension, not an implicit default.
        # Native also reports authoritative prefill/decode timings and supports
        # keep_alive for deterministic unloading between models.
        self.native_url = base_url.rstrip("/").removesuffix("/v1")
        self.timeout = timeout
        self._thinking_cache: dict[str, bool] = {}

    def supports_thinking(self, model: str) -> bool:
        """Whether the model declares a thinking capability (cached per model).

        Sending `think` to a model without it is a hard error from Ollama
        ('does not support thinking') and yields an empty response — which then
        scores as a total failure rather than an unsupported request.
        """
        if model in self._thinking_cache:
            return self._thinking_cache[model]
        import httpx
        supported = False
        try:
            r = httpx.post(f"{self.native_url}/api/show", json={"model": model}, timeout=15)
            supported = "thinking" in (r.json().get("capabilities") or [])
        except Exception:
            pass
        self._thinking_cache[model] = supported
        return supported

    def unload(self, model: str) -> bool:
        """Ask Ollama to evict a model immediately (keep_alive=0).

        Called between models so two large models are never resident at once,
        which is what drove the machine into swap during earlier runs. This is
        client-side and portable — unlike OLLAMA_MAX_LOADED_MODELS, which is a
        server-side env var and silently does nothing when set on the client.
        """
        import httpx
        try:
            httpx.post(f"{self.native_url}/api/generate",
                       json={"model": model, "keep_alive": 0}, timeout=30)
            return True
        except Exception:
            return False

    def complete_native(self, model: str, prompt: str, system: str = None,
                        max_tokens: int = 2048, temperature: float = 0.0,
                        ctx: int | None = None, guard_cfg: dict | None = None,
                        think: bool | None = None, keep_alive=None,
                        tools: list[dict] | None = None) -> dict:
        """Single-turn streaming completion over Ollama's native /api/chat.

        Builds a one-shot messages list from prompt/system and delegates to
        `complete_chat`, which holds the actual streaming/parsing logic shared
        with multi-turn callers (see its docstring).
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.complete_chat(model, messages, max_tokens=max_tokens,
                                   temperature=temperature, ctx=ctx, guard_cfg=guard_cfg,
                                   think=think, keep_alive=keep_alive, tools=tools)

    def complete_chat(self, model: str, messages: list[dict],
                      max_tokens: int = 2048, temperature: float = 0.0,
                      ctx: int | None = None, guard_cfg: dict | None = None,
                      think: bool | None = None, keep_alive=None,
                      tools: list[dict] | None = None) -> dict:
        """Streaming completion over Ollama's native /api/chat for an arbitrary
        message history (multi-turn tool-call loops, assistant/tool messages
        already appended by the caller) — `complete_native` is the single-turn
        convenience wrapper around this for every other benchmark.

        Separates thinking from the answer: `reasoning` holds chain-of-thought,
        `content` holds only what gets scored. Both count as liveness for the
        watchdog, so a long thinking phase is not mistaken for a stalled call.

        `tools`, when given, is passed through verbatim (OpenAI-style function
        schemas) and any `message.tool_calls` the model emits are collected
        into the returned `tool_calls` list. Ollama emits a tool call as one
        complete structured chunk rather than token-streaming it, but calls
        are accumulated across chunks regardless in case a model splits them.
        """
        import httpx

        options = {"temperature": temperature, "num_predict": max_tokens}
        if ctx:
            options["num_ctx"] = ctx
        body = {"model": model, "messages": messages, "stream": True, "options": options}
        if tools:
            body["tools"] = tools
        # Only send `think` to models that support it. Asking a non-thinking
        # model to think is a hard error and returns nothing, which would be
        # scored as a failed answer rather than an unsupported request.
        if think is not None and self.supports_thinking(model):
            body["think"] = think
        else:
            think = False if think else think
        if keep_alive is not None:
            body["keep_alive"] = keep_alive

        watchdog = StreamWatchdog(guard_cfg)
        watchdog.begin()
        start = time.perf_counter()
        ttft = None
        parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[dict] = []
        final: dict = {}

        # A read timeout is the backstop for the watchdog: closing the response
        # from another thread does not reliably unblock a blocked iter_lines(),
        # which previously deadlocked the run. The timeout guarantees the reader
        # wakes up regardless, and the loop also checks the trip flag itself.
        # The read timeout must cover the *longest* legitimate silence, which is
        # prefill on a cold or long-context request — not just the inter-token
        # stall. Sizing it to token_stall alone killed samples whose first token
        # simply took a while, scoring a capable model zero. The watchdog, not
        # this timeout, is what decides a call is unhealthy.
        g = guard_cfg or {}
        quiet = max(float(g.get("token_stall_seconds", 20.0)),
                    float(g.get("ttft_ceiling_seconds", 300.0)))
        timeouts = httpx.Timeout(connect=30.0, read=quiet + 30.0, write=30.0, pool=30.0)
        try:
            with httpx.stream("POST", f"{self.native_url}/api/chat", json=body,
                              timeout=timeouts) as resp:
                watchdog.arm(on_trip=resp.close)
                try:
                    for line in resp.iter_lines():
                        if watchdog.tripped:
                            break
                        if not line:
                            continue
                        chunk = json.loads(line)
                        msg = chunk.get("message") or {}
                        thinking = msg.get("thinking")
                        content = msg.get("content")
                        chunk_tool_calls = msg.get("tool_calls")
                        if thinking or content or chunk_tool_calls:
                            watchdog.on_token()
                        if thinking:
                            reasoning_parts.append(thinking)
                        if content:
                            if ttft is None:
                                ttft = time.perf_counter() - start
                            parts.append(content)
                        if chunk_tool_calls:
                            if ttft is None:
                                ttft = time.perf_counter() - start
                            tool_calls.extend(chunk_tool_calls)
                        if chunk.get("done"):
                            final = chunk
                            # Generation is over. Disarm before the stream is
                            # drained and closed, otherwise the watchdog keeps
                            # scoring an idle stream: with no further tokens the
                            # trailing window empties and reads as a decode
                            # collapse, or the silence eventually trips the
                            # stall detector. Both scored real answers as swap
                            # aborts on short-output benchmarks (BFCL tool
                            # calls) while the machine had zero swap activity.
                            # The guard's job ends when the model stops
                            # generating; waiting on the server is not thrash.
                            watchdog.stop()
                except Exception:
                    if not watchdog.tripped:
                        raise
                finally:
                    watchdog.stop()
        except Exception as e:
            watchdog.stop()
            return {
                "content": "", "reasoning": "", "tool_calls": [], "ttft": None,
                "elapsed": time.perf_counter() - start, "decode_elapsed": 0,
                "prompt_tokens": 0, "completion_tokens": 0,
                "tok_per_sec": 0, "decode_tps": 0, "think": think,
                "aborted": watchdog.reason, "abort_hard": watchdog.hard,
                "error": watchdog.reason or str(e),
            }

        total_elapsed = time.perf_counter() - start
        content = "".join(parts)
        completion_tokens = final.get("eval_count") or 0
        prompt_tokens = final.get("prompt_eval_count") or 0
        # Server-reported decode duration is authoritative — it excludes prefill
        # and queueing, so it is the number comparable to published tok/s.
        eval_ns = final.get("eval_duration") or 0
        if completion_tokens and eval_ns:
            decode_elapsed = eval_ns / 1e9
            decode_tps = completion_tokens / decode_elapsed
        else:
            decode_elapsed = max(total_elapsed - (ttft or 0), 1e-6)
            completion_tokens = completion_tokens or max(1, len(content) // 4)
            decode_tps = completion_tokens / decode_elapsed

        return {
            "content": content,
            "reasoning": "".join(reasoning_parts),
            "tool_calls": tool_calls,
            "ttft": ttft,
            "elapsed": total_elapsed,
            "decode_elapsed": decode_elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tok_per_sec": completion_tokens / total_elapsed if total_elapsed > 0 else 0,
            "decode_tps": decode_tps,
            "think": think,
            "aborted": watchdog.reason,
            "abort_hard": watchdog.hard,
            "error": watchdog.reason,
        }

    def _build_ctx_kwargs(self, ctx: int | None, kwargs: dict) -> dict:
        if ctx is None:
            return kwargs
        extra = kwargs.get("extra_body", {})
        extra["options"] = {**(extra.get("options", {})), "num_ctx": ctx}
        kwargs["extra_body"] = extra
        return kwargs

    def complete_streaming(self, model: str, prompt: str, system: str = None, max_tokens: int = 2048, temperature: float = 0.0, ctx: int | None = None, guard_cfg: dict | None = None) -> dict:
        """Streaming completion. Records TTFT and decode TPS separately from prefill.

        A StreamWatchdog aborts the call mid-flight if the generation stalls or
        the machine starts thrashing, so a bad sample costs seconds rather than
        running effectively forever. On abort the partial content is kept and
        `aborted` carries the reason.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        watchdog = StreamWatchdog(guard_cfg)
        watchdog.begin()
        start = time.perf_counter()
        ttft = None
        parts = []
        reasoning_parts = []
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
            watchdog.arm(on_trip=stream.close)
            try:
                for chunk in stream:
                    if chunk.usage:
                        prompt_tokens = chunk.usage.prompt_tokens or 0
                        completion_tokens = chunk.usage.completion_tokens or 0
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if not delta:
                        continue
                    # Thinking models stream chain-of-thought in `reasoning`,
                    # separate from the answer in `content`. Both count as
                    # liveness — otherwise a long thinking phase looks like
                    # prefill silence and trips the watchdog — but only
                    # `content` is the answer that gets scored.
                    reasoning_delta = (delta.model_extra or {}).get("reasoning")
                    if delta.content or reasoning_delta:
                        watchdog.on_token()
                    if reasoning_delta:
                        reasoning_parts.append(reasoning_delta)
                    if delta.content:
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        parts.append(delta.content)
            except Exception:
                # A trip closes the stream underneath us; that raise is expected.
                if not watchdog.tripped:
                    raise
            finally:
                watchdog.stop()

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
                "reasoning": "".join(reasoning_parts),
                "tok_per_sec": completion_tokens / total_elapsed if total_elapsed > 0 else 0,
                "decode_tps": completion_tokens / decode_elapsed,
                "aborted": watchdog.reason,
                "abort_hard": watchdog.hard,
                "error": watchdog.reason,
            }
        except Exception as e:
            end = time.perf_counter()
            watchdog.stop()
            return {
                "content": "", "ttft": None, "elapsed": end - start,
                "decode_elapsed": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "tok_per_sec": 0, "decode_tps": 0,
                "aborted": watchdog.reason, "abort_hard": watchdog.hard,
                "error": watchdog.reason or str(e),
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
