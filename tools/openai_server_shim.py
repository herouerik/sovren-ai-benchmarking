#!/usr/bin/env python3
"""Ollama-native API shim for an OpenAI-compatible server (llama.cpp `llama-server`).

`harness/client.py` talks to Ollama's native /api/chat, not /v1, so a model
Ollama cannot load yet (new GGUF architecture) can't be benchmarked by pointing
`base_url` at llama-server directly — every call 404s. Rather than add a second
client path, this exposes the upstream server behind the subset of the native
API the harness calls, the same approach as tools/needle_shim.py:

    POST /api/chat      streaming NDJSON in, translated from /v1/chat/completions SSE
    POST /api/show      model metadata for the results' model_info block
    GET  /api/tags      on-disk size (collect_model_info reads size from here)
    POST /api/generate  keep_alive=0 unload no-op, so client.unload() succeeds

Timings come from llama-server's own `timings` block, so decode tok/s is the
server's number, not the shim's wall clock.

Sampling: --temperature/--top-p/--top-k override what the harness sends, for
benchmarking a model at its vendor-recommended settings as a separate row.

Thinking: the shim advertises no `thinking` capability, so the harness never
sends `think` and the model is benchmarked in whatever mode its chat template
imposes. Flint's template always reasons; `reasoning_content` is surfaced as
`message.thinking` so it still counts as watchdog liveness.

Usage:
    llama-server -m model.gguf --alias NAME --port 8090 -c 262144 -ngl 99 --jinja
    python tools/openai_server_shim.py --upstream http://127.0.0.1:8090 --model NAME \\
        --gguf model.gguf --port 8091
    # config: ollama.base_url: http://localhost:8091/v1
"""
import argparse
import json
import os
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

UPSTREAM = ""
MODEL_NAME = ""
SIZE_BYTES = 0
CARD: dict = {}
SAMPLING: dict = {}


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """Ollama keeps tool-call arguments as objects and links tool results by
    position; the OpenAI schema wants JSON-string arguments and call ids."""
    out = []
    pending_ids: list[str] = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            pending_ids = []
            for c in m["tool_calls"]:
                call_id = c.get("id") or uuid.uuid4().hex[:24]
                pending_ids.append(call_id)
                fn = c["function"]
                args = fn.get("arguments")
                calls.append({"id": call_id, "type": "function", "function": {
                    "name": fn["name"],
                    "arguments": args if isinstance(args, str) else json.dumps(args or {})}})
            m["tool_calls"] = calls
        if m.get("role") == "tool":
            m.pop("tool_name", None)
            m["tool_call_id"] = pending_ids.pop(0) if pending_ids else uuid.uuid4().hex[:24]
        m.pop("thinking", None)
        out.append(m)
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
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
                "name": MODEL_NAME, "model": MODEL_NAME, "size": SIZE_BYTES,
                "details": {"quantization_level": CARD["quantization_level"],
                            "family": CARD["architecture"],
                            "parameter_size": CARD["params"]}}]})
        if self.path.rstrip("/") == "/api/ps":
            return self._json({"models": [{"name": MODEL_NAME, "model": MODEL_NAME,
                                            "size": SIZE_BYTES}]})
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.rstrip("/")
        if path == "/api/show":
            self._body()
            return self._json({
                "capabilities": ["tools"],
                "details": {"quantization_level": CARD["quantization_level"],
                            "family": CARD["architecture"],
                            "parameter_size": CARD["params"]},
                "model_info": {
                    "general.architecture": CARD["architecture"],
                    "general.parameter_count": CARD["parameter_count"],
                    f"{CARD['architecture']}.context_length": CARD["context_length"]}})
        if path == "/api/generate":
            self._body()
            return self._json({"model": MODEL_NAME, "done": True, "done_reason": "unload"})
        if path == "/api/chat":
            return self._chat()
        self._json({"error": "not found"}, 404)

    def _chat(self):
        req = self._body()
        opts = req.get("options") or {}
        body = {
            "model": MODEL_NAME,
            "messages": _to_openai_messages(req.get("messages") or []),
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": opts.get("temperature", 0.0),
            "max_tokens": opts.get("num_predict", 2048),
        }
        # A vendor-recommended sampler replaces the harness's temperature 0, which
        # is a different (greedy) operating point for models tuned to sample.
        body.update(SAMPLING)
        if "seed" in opts:
            body["seed"] = opts["seed"]
        if req.get("tools"):
            body["tools"] = req["tools"]

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def emit(chunk: dict):
            line = (json.dumps(chunk) + "\n").encode()
            self.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n")
            self.wfile.flush()

        calls: dict[int, dict] = {}
        timings: dict = {}
        finish = "stop"
        with httpx.stream("POST", f"{UPSTREAM}/v1/chat/completions", json=body,
                          timeout=httpx.Timeout(30.0, read=1800.0)) as resp:
            if resp.status_code >= 400:
                resp.read()
                emit({"error": {"message": resp.text[:300]}, "done": True})
                self.wfile.write(b"0\r\n\r\n")
                return
            for line in resp.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                d = json.loads(line[6:])
                timings = d.get("timings") or timings
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    msg = {"role": "assistant", "content": delta.get("content") or ""}
                    if delta.get("reasoning_content"):
                        msg["thinking"] = delta["reasoning_content"]
                    for tc in delta.get("tool_calls") or []:
                        slot = calls.setdefault(tc.get("index", 0), {"name": "", "args": ""})
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["args"] += fn.get("arguments") or ""
                    if msg["content"] or msg.get("thinking"):
                        emit({"model": MODEL_NAME, "message": msg, "done": False})

        if calls:
            tool_calls = [{"function": {"name": c["name"],
                                        "arguments": json.loads(c["args"] or "{}")}}
                          for _, c in sorted(calls.items())]
            emit({"model": MODEL_NAME, "done": False,
                  "message": {"role": "assistant", "content": "", "tool_calls": tool_calls}})
        emit({"model": MODEL_NAME, "message": {"role": "assistant", "content": ""},
              "done": True, "done_reason": "tool_calls" if calls else finish,
              "eval_count": timings.get("predicted_n", 0),
              "eval_duration": int(timings.get("predicted_ms", 0) * 1e6),
              "prompt_eval_count": timings.get("prompt_n", 0),
              "prompt_eval_duration": int(timings.get("prompt_ms", 0) * 1e6)})
        self.wfile.write(b"0\r\n\r\n")


def main():
    global UPSTREAM, MODEL_NAME, SIZE_BYTES, CARD, SAMPLING
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--model", required=True, help="name reported to the harness; must match llama-server --alias")
    ap.add_argument("--gguf", required=True, help="model file, for the size reported in /api/tags")
    ap.add_argument("--architecture", required=True)
    ap.add_argument("--params", required=True, help="e.g. 39.8B")
    ap.add_argument("--parameter-count", type=int, required=True)
    ap.add_argument("--quantization", required=True)
    ap.add_argument("--context-length", type=int, required=True)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    UPSTREAM = args.upstream.rstrip("/")
    MODEL_NAME = args.model
    SIZE_BYTES = os.path.getsize(args.gguf)
    CARD = {"architecture": args.architecture, "params": args.params,
            "parameter_count": args.parameter_count,
            "quantization_level": args.quantization, "context_length": args.context_length}
    SAMPLING = {k: v for k, v in (("temperature", args.temperature), ("top_p", args.top_p),
                                  ("top_k", args.top_k)) if v is not None}
    print(f"openai-server shim ({MODEL_NAME}) on http://{args.host}:{args.port} -> {UPSTREAM}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
