#!/usr/bin/env python3
"""A fake OpenAI-compatible model server for testing the agent pipeline without a GPU.

It speaks the same /v1/chat/completions API that vLLM serves, and follows a fixed script:
    1st call -> run_command("git status --short | head -5")
    2nd call -> run_command(<small edit to README>)   so the patch is non-empty
    3rd call -> submit_patch()
    then     -> a short final text answer
Every request is appended to --log (JSONL), which shows exactly what the real model would receive:
system prompt, the harness's task message, tool definitions and tool results.

If a request carries "return_token_ids": true (kaggle/common/capture_proxy.py adds it), the reply
also has vLLM's extra fields: prompt_token_ids and choices[0].token_ids. The ids are fake, but the
completion ends the way Gemma 4's does: 50 ('<|tool_response>') after a tool call, 106 ('<turn|>')
after a text answer, so the capture-to-training-data path can be tested without a GPU.

Usage:
    python scripts/mock_llm.py --port 8001 --log /tmp/mock_requests.jsonl
    MODEL_URL=http://127.0.0.1:8001/v1 python kaggle/agent_eval/agent_eval.py ...
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOOL_CALL_STOP, TEXT_STOP = 50, 106  # Gemma 4 stop tokens: '<|tool_response>' and '<turn|>'

SCRIPT = [
    ("run_command", {"command": "git status --short | head -5"}),
    ("run_command", {"command": "echo '<!-- mock edit -->' >> README.md && git diff --stat"}),
    ("submit_patch", {}),
]


def reply(request: dict) -> dict:
    tool_results = sum(1 for m in request.get("messages", []) if m.get("role") == "tool")
    message: dict = {"role": "assistant", "content": None}
    if tool_results < len(SCRIPT):
        name, args = SCRIPT[tool_results]
        message["content"] = f"Step {tool_results + 1}: calling {name}."
        message["tool_calls"] = [{
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]
        finish = "tool_calls"
    else:
        message["content"] = "Mock agent finished: made a small README edit and submitted it."
        finish = "stop"
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.get("model", "mock"),
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }
    if request.get("return_token_ids"):
        # Fake but well-formed: 2 = <bos>, then one id per message; the completion ends with the stop
        # token vLLM would have sampled (it is not shown in the text, but it is in token_ids).
        response["prompt_token_ids"] = [2] + [1000 + i for i in range(len(request.get("messages", [])))]
        stop = TOOL_CALL_STOP if finish == "tool_calls" else TEXT_STOP
        response["choices"][0]["token_ids"] = [3000 + tool_results, 3100, stop]
    return response


def make_handler(log_path: str | None):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - http.server naming
            if self.path.rstrip("/").endswith("/models"):
                self._send({"object": "list", "data": [{"id": "mock", "object": "model"}]})
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self):  # noqa: N802
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if log_path:
                with open(log_path, "a") as fh:
                    fh.write(json.dumps(request) + "\n")
            if request.get("stream"):
                self._send({"error": "streaming not supported by the mock"}, 400)
                return
            self._send(reply(request))

        def log_message(self, *args):  # keep the console quiet
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--log", help="append every request here as JSONL")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.log))
    print(f"mock model listening on http://127.0.0.1:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
