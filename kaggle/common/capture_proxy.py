"""Recording reverse proxy between the agent's model client (LiteLLM) and vLLM, for training data.

Why: fine-tuning text must be exactly what the served model saw and sampled. The agent traces only
hold the parsed messages; vLLM can also return the token ids it rendered (prompt_token_ids) and
sampled (choices[i].token_ids) when asked with "return_token_ids": true. This proxy adds that flag
to every chat-completion request, appends one JSON line per call to
<out_dir>/<variant>/<instance_id>.jsonl.gz, and strips the id fields again so the agent receives
exactly the response it would get without the proxy.

Each record: seq, t_start, t_end, variant, instance_id, kind (agent / summarizer / other),
http_status, error (response text when status != 200), retried_without_token_ids,
first_user_sha1, request, response, prompt_token_ids, completion_token_ids (one list per choice).

Attribution: the notebook sets TAG["variant"] and TAG["instance_id"] before each task (tasks run
one at a time). The proxy reads TAG (and assigns seq) when a call ARRIVES, not when vLLM answers:
if a task hits the harness's session timeout, its in-flight call is cancelled on the agent's side
but vLLM still finishes it, possibly after the next task has started; that late record must stay
in the task that sent it. first_user_sha1 is a weaker hint for re-attribution (ADK compaction can
replace the first user message).

Fail-open rule: recording must never break a run. If the upstream rejects return_token_ids (HTTP
400), the request is resent without it and the ids are recorded as null; if parsing the response
or writing the record fails, the agent still gets the upstream's answer; if the proxy itself fails
before the call reaches the model, the agent's request is forwarded untouched. An unreachable
upstream becomes an HTTP 502, which LiteLLM treats like any other server error.

Stdlib only, and no `from __future__` imports: scripts/kaggle_build.py pastes this file into the
middle of a notebook, where such an import would be a syntax error.
"""

import gzip
import hashlib
import http.client
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TAG = {"variant": "unknown", "instance_id": "unknown", "seq": 0}
_LOCK = threading.Lock()
# Hop-by-hop / length headers that the proxy must set itself rather than copy through.
_DROP_HEADERS = ("host", "content-length", "connection", "accept-encoding", "transfer-encoding")
_SUMMARY_PREFIX = "The following is a conversation history between a user and an AI agent"


def _call_kind(body: dict) -> str:
    """'summarizer' for ADK's event-compaction calls (google/adk/apps/llm_event_summarizer.py), else agent/other."""
    msgs = body.get("messages") or []
    if not body.get("tools") and len(msgs) == 1 and msgs[0].get("role") == "user":
        c = msgs[0].get("content")
        text = c if isinstance(c, str) else " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict))
        if text.startswith(_SUMMARY_PREFIX):
            return "summarizer"
    return "agent" if body.get("tools") else "other"


def split_token_ids(data: bytes) -> tuple[dict, list | None, list | None]:
    """Split a vLLM chat-completion response into (response without ids, prompt ids, completion ids).

    The ids get their own record fields, so the recorded response is exactly what the agent sees.
    Completion ids are one list per choice; None when the upstream sent none (flag not honoured).
    """
    response = json.loads(data)
    prompt_ids = response.pop("prompt_token_ids", None)
    choices = [dict(ch) for ch in response.get("choices") or []]
    completion_ids = [ch.pop("token_ids", None) for ch in choices]
    response["choices"] = choices
    return response, prompt_ids, completion_ids if any(completion_ids) else None


def make_capture_handler(upstream_host: str, upstream_port: int, out_dir: Path, timeout: float = 1800):
    # token_ids_supported turns False once vLLM has rejected the flag and accepted the same request
    # without it: stop asking. warned: print each kind of proxy failure once, not once per call.
    state = {"token_ids_supported": True, "warned": set()}

    def warn_once(what: str) -> None:
        if what not in state["warned"]:
            state["warned"].add(what)
            print(f"capture proxy: {what} failed (the run continues; warning shown once):\n"
                  + traceback.format_exc(), flush=True)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive, as LiteLLM's httpx client expects

        def log_message(self, *args):
            pass

        def _forward(self, method: str, raw: bytes) -> tuple[int, str, bytes]:
            """Send the request upstream; a connection failure becomes a 502 instead of an exception."""
            headers = {k: v for k, v in self.headers.items() if k.lower() not in _DROP_HEADERS}
            headers["Content-Length"] = str(len(raw))
            try:
                conn = http.client.HTTPConnection(upstream_host, upstream_port, timeout=timeout)
                try:
                    conn.request(method, self.path, body=raw if method != "GET" else None, headers=headers)
                    resp = conn.getresponse()
                    return resp.status, resp.getheader("Content-Type", "application/json"), resp.read()
                finally:
                    conn.close()
            except Exception as err:  # upstream down or timed out: answer like a gateway would
                data = json.dumps({"error": {"message": f"capture proxy: upstream error: {err!r}"}}).encode()
                return 502, "application/json", data

        def _reply(self, status: int, ctype: str, data: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802 - http.server naming (/v1/models, /health)
            self._reply(*self._forward("GET", b""))

        def do_POST(self):  # noqa: N802
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            try:
                reply = self._capture(raw)
            except Exception:
                # A bug before the call reached the model: forward the agent's bytes untouched.
                # (_capture never raises once the upstream has answered, so nothing is sent twice.)
                warn_once("capturing a request")
                reply = self._forward("POST", raw)
            self._reply(*reply)

        def _capture(self, raw: bytes) -> tuple[int, str, bytes]:
            try:
                body = json.loads(raw)
            except ValueError:
                body = None
            if (not isinstance(body, dict) or body.get("stream")
                    or not self.path.rstrip("/").endswith("/chat/completions")):
                # Streaming is off in ADK unless RunConfig.streaming_mode=SSE; pass anything else through.
                return self._forward("POST", raw)

            # Snapshot the task tag now, while the sending task is still the current one (see module
            # docstring: a timed-out task's call can be answered after the next task has begun).
            with _LOCK:
                TAG["seq"] += 1
                tag = {"seq": TAG["seq"], "variant": TAG["variant"], "instance_id": TAG["instance_id"]}
            added = state["token_ids_supported"] and "return_token_ids" not in body
            t0 = time.time()
            if added:
                status, ctype, data = self._forward("POST", json.dumps(dict(body, return_token_ids=True)).encode())
            else:
                status, ctype, data = self._forward("POST", raw)
            retried = False
            if added and status == 400:
                # Maybe this vLLM does not know the flag. Resend the agent's own bytes so the run goes
                # on; 400s for other reasons (context overflow) just come back with the same error.
                first_error = data
                status, ctype, data = self._forward("POST", raw)
                retried, added = True, False
                if status == 200:
                    state["token_ids_supported"] = False
                    print("capture proxy: upstream rejected return_token_ids "
                          f"({first_error[:200]!r}); recording without token ids from now on", flush=True)
            t1 = time.time()

            out, response, prompt_ids, completion_ids = data, None, None, None
            if status == 200:
                try:
                    response, prompt_ids, completion_ids = split_token_ids(data)
                    if added:  # the agent did not ask for ids: give it the response without them
                        out = json.dumps(response).encode()
                except Exception:
                    warn_once("parsing a response")  # the agent still gets the upstream's bytes
            try:
                self._record(tag, body, status, data, response, prompt_ids, completion_ids, retried, t0, t1)
            except Exception:
                warn_once("writing a record")
            return status, ctype, out

        def _record(self, tag: dict, body: dict, status: int, data: bytes, response: dict | None,
                    prompt_ids: list | None, completion_ids: list | None, retried: bool,
                    t0: float, t1: float) -> None:
            """Append one JSON line to <out_dir>/<variant>/<instance_id>.jsonl.gz.

            tag is the TAG snapshot taken when the call arrived. seq therefore follows send order;
            lines in a file can be out of seq order (a slow call written late), so readers sort by seq.
            """
            msgs = body.get("messages") or []
            first_user = next((m for m in msgs if isinstance(m, dict) and m.get("role") == "user"), {})
            fu = first_user.get("content")
            fu = fu if isinstance(fu, str) else json.dumps(fu)
            seq, variant, iid = tag["seq"], tag["variant"], tag["instance_id"]
            rec = {
                "seq": seq, "t_start": t0, "t_end": t1, "variant": variant, "instance_id": iid,
                "kind": _call_kind(body), "http_status": status,
                "error": None if status == 200 else data.decode("utf-8", errors="replace")[:4000],
                "retried_without_token_ids": retried,
                "first_user_sha1": hashlib.sha1(fu.encode()).hexdigest(),
                "request": body,                         # exactly what LiteLLM sent
                "response": response,                    # what the agent received (None on errors)
                "prompt_token_ids": prompt_ids,          # exactly what vLLM rendered
                "completion_token_ids": completion_ids,  # exactly what the model sampled
            }
            path = out_dir / str(variant) / f"{iid}.jsonl.gz"
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            with _LOCK:
                path.parent.mkdir(parents=True, exist_ok=True)
                # Append mode adds a new gzip member per call; gzip readers treat the members as one stream.
                with gzip.open(path, "at", encoding="utf-8") as f:
                    f.write(line)

    return Handler


def start_proxy(listen_port: int, upstream_host: str, upstream_port: int, out_dir: str | Path) -> ThreadingHTTPServer:
    """Serve the proxy on 127.0.0.1:<listen_port> in a daemon thread (port 0 = any free port).

    The port actually used is server.server_address[1].
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    handler = make_capture_handler(upstream_host, upstream_port, out_dir)
    server = ThreadingHTTPServer(("127.0.0.1", listen_port), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="capture-proxy").start()
    return server
