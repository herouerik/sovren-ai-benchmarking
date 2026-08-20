#!/usr/bin/env python3
"""Ollama-native API shim for Cactus Needle (`pip install cactus-needle`).

Needle is not an Ollama model: it ships as a Python package around a
dependency-free C++ engine, and its own `--serve` mode binds a *static*
tools.json at startup — unusable for BFCL, where every sample carries its
own function schemas. Rather than add a second client path to the harness,
this exposes Needle behind the subset of Ollama's native API that
`harness/client.py` and `run_benchmark.collect_model_info` actually call:

    POST /api/chat      streaming NDJSON, `tools` in, `message.tool_calls` out
    POST /api/show      model metadata for the results' model_info block
    GET  /api/tags      on-disk size (collect_model_info reads size from here)
    POST /api/generate  keep_alive=0 unload no-op, so client.unload() succeeds

Everything downstream — the swap watchdog, timing capture, incremental
merge, dashboard — then works untouched.

Two Needle-specific details this has to get right:

1. State leaks across calls. A `Needle` instance carries context between
   `.complete()` calls, and one unrelated prior query is enough to turn a
   correct tool call into a refusal (reproduced: cold -> `call`, after an
   unrelated prior -> `respond` with no calls, after `.reset()` -> `call`
   again). Every request resets before generating, so samples stay
   independent the way the rest of the harness assumes.

2. Grammar compile is per-tool-set, not per-token. `needle_init` compiles a
   byte-level grammar from the declared schemas (~3.2s on an M4) and BFCL
   hands us a different tool set every sample. That cost is prefill-like,
   not decode, so it is reported as prompt_eval_duration and kept out of
   eval_duration — otherwise Needle's decode rate would read ~10x too low
   for reasons that have nothing to do with the model. Instances are cached
   by tool-set so a repeated schema pays it once.

Usage:
    python tools/needle_shim.py --port 11499
    python run_benchmark.py --config config-needle.yaml
"""
import argparse
import hashlib
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import needle

MODEL_NAME = "needle2"

# Reported in /api/show and /api/tags. Needle has no GGUF metadata to read,
# so these come from the model card (45M params, CQ2-bit, 14MB binary) and
# are the same fields collect_model_info would otherwise pull from Ollama.
# CQ2 is trained end-to-end, not a post-hoc quantization of a larger model.
CARD = {
    "params": "45M",
    "parameter_count": 45_000_000,
    "quantization_level": "CQ2",
    "size_bytes": 14 * 1024 * 1024,
    "official_name": "Needle 2",
    "architecture": "needle2",
    "context_length": 4096,
}

_agents: dict[str, "needle.Needle"] = {}


def _agent(tools: list[dict], system: str | None) -> tuple["needle.Needle", float]:
    """Needle instance for this tool-set, plus the grammar-compile seconds.

    Cached on the schema digest: `needle_init` compiles a grammar from the
    schemas, and paying that per sample would misattribute setup cost to
    generation.
    """
    schemas = [
        t["function"] if t.get("type") == "function" and "function" in t else t
        for t in (tools or [])
    ]
    key = hashlib.sha256(
        json.dumps([schemas, system or ""], sort_keys=True).encode()
    ).hexdigest()
    if key in _agents:
        return _agents[key], 0.0
    t0 = time.perf_counter()
    agent = needle.Needle(tools=schemas, system=system or None)
    compile_s = time.perf_counter() - t0
    _agents[key] = agent
    return agent, compile_s


def _last_user_text(messages: list[dict]) -> tuple[str, str | None]:
    """Flatten the harness's one-shot messages list into Needle's text input.

    Needle takes a single query string plus an optional system block set at
    init, not a role-tagged history. Every benchmark that reaches this shim
    is single-turn, so the last user message is the query; any earlier
    assistant/tool turns would need Needle's own `run` loop and are refused
    rather than silently dropped.
    """
    system = next((m["content"] for m in messages if m.get("role") == "system"), None)
    users = [m for m in messages if m.get("role") == "user"]
    if len(messages) - len(users) - (1 if system else 0) > 0:
        raise ValueError("needle_shim handles single-turn requests only")
    return (users[-1]["content"] if users else ""), system


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet; the harness prints progress
        pass

    def _json(self, obj: dict, status: int = 200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path.rstrip("/") == "/api/tags":
            return self._json({"models": [{
                "name": MODEL_NAME,
                "model": MODEL_NAME,
                "size": CARD["size_bytes"],
                "details": {"quantization_level": CARD["quantization_level"],
                            "family": CARD["architecture"],
                            "parameter_size": CARD["params"]},
            }]})
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.rstrip("/")
        try:
            if path == "/api/show":
                return self._show()
            if path == "/api/generate":
                # Only ever called as client.unload()'s keep_alive=0. Needle's
                # whole footprint is ~140MB peak, so there is nothing to evict;
                # answering 200 keeps the harness's unload-between-models step
                # from logging a spurious failure.
                return self._json({"done": True})
            if path == "/api/chat":
                return self._chat()
        except Exception as e:
            # Surface as a real HTTP error: client.py checks status_code and
            # raises, which is what makes a broken sample visible instead of
            # scoring as "the model chose not to answer".
            return self._json({"error": str(e)}, 500)
        self._json({"error": "not found"}, 404)

    def _show(self):
        self._json({
            # No "thinking" capability: supports_thinking() reads this, and
            # Needle's `reasoning` field is a post-hoc justification string,
            # not a chain-of-thought budget the harness can vary.
            "capabilities": ["tools"],
            "details": {"quantization_level": CARD["quantization_level"],
                        "family": CARD["architecture"],
                        "parameter_size": CARD["params"]},
            "model_info": {
                "general.architecture": CARD["architecture"],
                "general.basename": CARD["official_name"],
                "general.parameter_count": CARD["parameter_count"],
                f"{CARD['architecture']}.context_length": CARD["context_length"],
            },
        })

    def _chat(self):
        req = self._body()
        query, system = _last_user_text(req.get("messages") or [])
        max_new = int((req.get("options") or {}).get("num_predict") or 256)
        agent, compile_s = _agent(req.get("tools") or [], system)

        # Mandatory: see module docstring note 1.
        agent.reset()
        t0 = time.perf_counter()
        result = agent.complete(query, max_new_tokens=max_new)
        gen_s = time.perf_counter() - t0

        calls = [{"function": {"name": c.get("name"),
                               "arguments": c.get("arguments") or {}}}
                 for c in (result.get("function_calls") or [])]
        # Needle emits a decision, not prose. `reason`/`reasoning` is its own
        # justification text; it goes to `content` only when no call was made,
        # so a tool-calling sample is never scored against explanatory prose.
        content = "" if calls else (result.get("reason") or result.get("reasoning") or "")

        # Needle reports its own rates; derive counts from them so the numbers
        # in the results are the engine's, not this shim's wall-clock guesses.
        decode_tps = float(result.get("decode_tps") or 0.0)
        prefill_tps = float(result.get("prefill_tps") or 0.0)
        eval_count = max(1, round(decode_tps * gen_s)) if decode_tps else 1
        eval_ns = int((eval_count / decode_tps) * 1e9) if decode_tps else int(gen_s * 1e9)
        # Prefill window is generation minus decode, and deliberately excludes
        # grammar compile: folding compile into prompt_eval_duration and then
        # multiplying by prefill_tps invents thousands of prompt tokens that
        # were never in the prompt (observed 7920 for a two-sentence request).
        # Compile is reported on its own key instead.
        prefill_s = max(gen_s - eval_ns / 1e9, 0.0)
        prompt_count = max(1, round(prefill_tps * prefill_s)) if prefill_tps else 1
        prompt_ns = int(prefill_s * 1e9)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for chunk in (
            {"model": MODEL_NAME, "message": {"role": "assistant", "content": content,
                                              **({"tool_calls": calls} if calls else {})},
             "done": False},
            {"model": MODEL_NAME, "message": {"role": "assistant", "content": ""},
             "done": True, "done_reason": "stop",
             "eval_count": eval_count, "eval_duration": eval_ns,
             "prompt_eval_count": prompt_count, "prompt_eval_duration": prompt_ns,
             "total_duration": int((gen_s + compile_s) * 1e9),
             "needle_grammar_compile_ns": int(compile_s * 1e9),
             # Not an Ollama field. Needle is the only model here that reports
             # its own peak RSS, and at 45M params it is the whole point.
             "needle_peak_ram_mb": result.get("peak_ram_mb"),
             "needle_confidence": result.get("confidence")},
        ):
            line = (json.dumps(chunk) + "\n").encode()
            self.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11499)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print(f"needle shim ({MODEL_NAME}) on http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
