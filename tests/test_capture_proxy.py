"""Tests for kaggle/common/capture_proxy.py against a tiny fake vLLM, plus the CAPTURE / ADAPTER_SRC
helpers in kaggle/agent_eval/agent_eval.py. Run with: python -m pytest -q tests/test_capture_proxy.py"""

import gzip
import inspect
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle" / "common"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "kaggle" / "agent_eval"))

import agent_eval as ae  # noqa: E402  (module level is stdlib-only; the harness is imported lazily)
import capture_proxy as cp  # noqa: E402
import mock_llm  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "run_command", "parameters": {"type": "object"}}}]
AGENT_BODY = {"model": "m", "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "TASK"}],
              "tools": TOOLS}


def fake_upstream(behaviour: str, seen: list, release: threading.Event | None = None):
    """A vLLM stand-in. behaviour: 'ok' honours return_token_ids, 'reject_flag' answers 400 to it,
    'overflow' answers 400 to everything (like a context-length error), 'garbage' answers 200 with
    a body that is not JSON. With release set, the FIRST POST waits for that event before answering
    (a slow generation); later ones answer at once."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            self._send(200, {"object": "list", "data": [{"id": "m"}]})

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            if release is not None and len(seen) == 1:
                release.wait(10)
            if behaviour == "overflow" or (behaviour == "reject_flag" and "return_token_ids" in body):
                return self._send(400, {"error": {"message": "bad request: maximum context length"}})
            if behaviour == "garbage":
                self.send_response(200)
                self.send_header("Content-Length", "8")
                self.end_headers()
                return self.wfile.write(b"not json")
            resp = {"id": "x", "object": "chat.completion", "model": body["model"],
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                                 "finish_reason": "stop"}]}
            if body.get("return_token_ids"):
                resp["prompt_token_ids"] = [2, 105, 9]
                resp["choices"][0]["token_ids"] = [7, 8, 106]
            self._send(200, resp)

    return Handler


@pytest.fixture
def run_proxy(tmp_path):
    """Start upstream + proxy; yields a function (handler) -> (proxy base URL, out_dir)."""
    servers = []
    cp.TAG.update(variant="v1", instance_id="task_1", seq=0)

    def start(upstream_handler, upstream_port: int | None = None):
        if upstream_port is None:
            up = ThreadingHTTPServer(("127.0.0.1", 0), upstream_handler)
            threading.Thread(target=up.serve_forever, daemon=True).start()
            servers.append(up)
            upstream_port = up.server_address[1]
        proxy = cp.start_proxy(0, "127.0.0.1", upstream_port, tmp_path / "captures")
        servers.append(proxy)
        return f"http://127.0.0.1:{proxy.server_address[1]}/v1", tmp_path / "captures"

    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def post(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def records(out_dir: Path, variant: str = "v1", iid: str = "task_1") -> list[dict]:
    with gzip.open(out_dir / variant / f"{iid}.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_record_written_and_ids_hidden_from_agent(run_proxy):
    seen = []
    url, out = run_proxy(fake_upstream("ok", seen))
    status, resp = post(url, AGENT_BODY)
    assert status == 200
    assert seen[0]["return_token_ids"] is True                          # the proxy asked vLLM for ids
    assert "prompt_token_ids" not in resp and "token_ids" not in resp["choices"][0]  # agent sees plain reply
    (rec,) = records(out)
    assert rec["request"] == AGENT_BODY                                   # what the agent sent, without the flag
    assert rec["response"]["choices"][0]["message"]["content"] == "hi"
    assert "prompt_token_ids" not in rec["response"] and "token_ids" not in rec["response"]["choices"][0]
    assert rec["prompt_token_ids"] == [2, 105, 9]
    assert rec["completion_token_ids"] == [[7, 8, 106]]
    assert (rec["kind"], rec["http_status"], rec["error"], rec["retried_without_token_ids"]) == \
        ("agent", 200, None, False)
    assert (rec["variant"], rec["instance_id"], rec["seq"]) == ("v1", "task_1", 1)


def test_tag_routes_records_to_per_task_files(run_proxy):
    url, out = run_proxy(fake_upstream("ok", []))
    post(url, AGENT_BODY)
    cp.TAG.update(variant="v2", instance_id="task_2")
    post(url, AGENT_BODY)
    assert len(records(out, "v1", "task_1")) == 1
    assert records(out, "v2", "task_2")[0]["seq"] == 2


def test_late_answer_stays_with_the_task_that_sent_it(run_proxy):
    """A task times out mid-call (the harness cancels the agent's request), the next task starts,
    and only then does vLLM answer. The record must land in the first task's file, not the next."""
    release = threading.Event()
    url, out = run_proxy(fake_upstream("ok", [], release))
    port = int(url.split(":")[2].split("/")[0])
    data = json.dumps(AGENT_BODY).encode()
    with socket.create_connection(("127.0.0.1", port)) as s:  # the agent's call for task_1 ...
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                  + f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
        time.sleep(0.3)                                       # ... reaches the proxy, then is cancelled
    cp.TAG.update(instance_id="task_2")                       # the harness moves on to the next task
    status, _ = post(url, dict(AGENT_BODY, messages=[{"role": "user", "content": "TASK 2"}]))
    assert status == 200                                      # task_2's first call, answered at once
    release.set()                                             # vLLM finally answers task_1's call
    path = out / "v1" / "task_1.jsonl.gz"
    for _ in range(100):
        if path.exists():
            break
        time.sleep(0.05)
    (late,) = records(out, "v1", "task_1")
    assert late["request"] == AGENT_BODY and late["seq"] == 1
    (task2,) = records(out, "v1", "task_2")
    assert task2["request"]["messages"][0]["content"] == "TASK 2" and task2["seq"] == 2


def test_rejected_flag_is_retried_without_it_and_then_dropped(run_proxy):
    seen = []
    url, out = run_proxy(fake_upstream("reject_flag", seen))
    status, resp = post(url, AGENT_BODY)
    assert status == 200 and resp["choices"][0]["message"]["content"] == "hi"
    assert ["return_token_ids" in b for b in seen] == [True, False]
    rec = records(out)[0]
    assert rec["retried_without_token_ids"] is True
    assert rec["prompt_token_ids"] is None and rec["completion_token_ids"] is None
    # Once vLLM has shown it does not take the flag, later calls go without it (no double requests).
    post(url, AGENT_BODY)
    assert ["return_token_ids" in b for b in seen] == [True, False, False]


def test_other_400s_reach_the_agent_and_are_logged(run_proxy):
    seen = []
    url, out = run_proxy(fake_upstream("overflow", seen))
    status, resp = post(url, AGENT_BODY)
    assert status == 400 and "maximum context length" in resp["error"]["message"]
    rec = records(out)[0]
    assert rec["http_status"] == 400 and "maximum context length" in rec["error"]
    assert rec["response"] is None and rec["prompt_token_ids"] is None
    post(url, AGENT_BODY)  # a context overflow must not switch the flag off for good
    assert "return_token_ids" in seen[2]


def test_fail_open_when_recording_breaks(run_proxy):
    url, out = run_proxy(fake_upstream("ok", []))
    (out / "v1").write_text("a file where the variant folder should be")  # makes every record write fail
    status, resp = post(url, AGENT_BODY)
    assert status == 200 and resp["choices"][0]["message"]["content"] == "hi"


def test_fail_open_when_response_is_not_json(run_proxy):
    url, out = run_proxy(fake_upstream("garbage", []))
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(AGENT_BODY).encode())
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert (resp.status, resp.read()) == (200, b"not json")  # the agent gets the upstream's bytes
    rec = records(out)[0]
    assert rec["http_status"] == 200 and rec["response"] is None and rec["prompt_token_ids"] is None


def test_fail_open_when_the_proxy_itself_breaks(run_proxy, monkeypatch):
    """A bug before the call reaches the model: the agent's request is forwarded untouched, once."""

    class BrokenClock:
        @staticmethod
        def time():
            raise RuntimeError("proxy bug")

    seen = []
    url, out = run_proxy(fake_upstream("ok", seen))
    monkeypatch.setattr(cp, "time", BrokenClock)
    status, resp = post(url, AGENT_BODY)
    assert status == 200 and resp["choices"][0]["message"]["content"] == "hi"
    assert seen == [AGENT_BODY]                      # sent once, without the flag
    assert not (out / "v1").exists()                 # nothing recorded, but the run goes on


def test_agent_that_asks_for_ids_gets_them(run_proxy):
    seen = []
    url, out = run_proxy(fake_upstream("ok", seen))
    status, resp = post(url, dict(AGENT_BODY, return_token_ids=True))
    assert resp["prompt_token_ids"] == [2, 105, 9] and resp["choices"][0]["token_ids"] == [7, 8, 106]
    assert records(out)[0]["completion_token_ids"] == [[7, 8, 106]]


def test_split_token_ids():
    data = json.dumps({"id": "x", "prompt_token_ids": [1, 2],
                       "choices": [{"index": 0, "token_ids": [5, 50]}, {"index": 1, "token_ids": [6, 106]}]})
    response, prompt_ids, completion_ids = cp.split_token_ids(data.encode())
    assert response == {"id": "x", "choices": [{"index": 0}, {"index": 1}]}
    assert (prompt_ids, completion_ids) == ([1, 2], [[5, 50], [6, 106]])
    assert cp.split_token_ids(b'{"choices": [{"index": 0}]}')[1:] == (None, None)  # flag ignored


def test_upstream_down_gives_502_and_a_record(run_proxy):
    import socket

    with socket.socket() as s:  # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]
    url, out = run_proxy(None, upstream_port=dead_port)
    status, resp = post(url, AGENT_BODY)
    assert status == 502 and "upstream error" in resp["error"]["message"]
    assert records(out)[0]["http_status"] == 502


def test_non_chat_requests_pass_through(run_proxy):
    url, out = run_proxy(fake_upstream("ok", []))
    with urllib.request.urlopen(url + "/models", timeout=10) as resp:
        assert json.loads(resp.read())["data"][0]["id"] == "m"
    assert not out.joinpath("v1").exists()


def test_summarizer_calls_are_labelled(run_proxy):
    url, out = run_proxy(fake_upstream("ok", []))
    text = "The following is a conversation history between a user and an AI agent. ..."
    post(url, {"model": "m", "messages": [{"role": "user", "content": text}]})
    assert records(out)[0]["kind"] == "summarizer"


def test_mock_llm_token_ids_end_in_gemma_stop_tokens(run_proxy):
    """scripts/mock_llm.py through the proxy: tool calls end in 50, the final text answer in 106."""
    url, out = run_proxy(mock_llm.make_handler(None))
    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "TASK"}]
    for _ in range(len(mock_llm.SCRIPT) + 1):
        status, resp = post(url, {"model": "m", "messages": messages, "tools": TOOLS})
        assert status == 200 and "prompt_token_ids" not in resp
        message = resp["choices"][0]["message"]
        messages = messages + [message] + [{"role": "tool", "tool_call_id": c["id"], "content": "ok"}
                                           for c in message.get("tool_calls") or []]
    ends = [r["completion_token_ids"][0][-1] for r in records(out)]
    assert ends == [50] * len(mock_llm.SCRIPT) + [106]
    assert all(r["prompt_token_ids"] for r in records(out))


# ---- agent_eval.py helpers (no harness needed) ----

def test_capture_view_points_registry_at_the_proxy(tmp_path, monkeypatch):
    """The copy's base_url is the proxy's; the real server keeps its own config."""

    class Config:
        host, port = "127.0.0.1", 0

    class FakeServer:  # same base_url rule as adk_submission's BaseInferenceServer (server.py)
        def __init__(self, config):
            self.config = config

        @property
        def base_url(self):
            return f"http://{self.config.host}:{self.config.port}/v1"

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), fake_upstream("ok", []))
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    config = Config()
    config.port = upstream.server_address[1]
    server = FakeServer(config)
    monkeypatch.setattr(ae, "OUT_DIR", tmp_path)
    cp.TAG.update(variant="v1", instance_id="task_1", seq=0)

    view = ae.capture_view(server)
    assert view.base_url != server.base_url and server.config.port == upstream.server_address[1]
    status, _ = post(view.base_url, AGENT_BODY)
    assert status == 200 and records(tmp_path / "captures")[0]["prompt_token_ids"] == [2, 105, 9]
    upstream.shutdown()


def test_tag_captures_with_task_keeps_signature_and_sets_instance_id():
    """swegemma's Evaluator inspects _run_agent_sandbox's signature to decide which kwargs to pass."""

    class Task:
        instance_id = "rich_3718"

    class Evaluator:
        async def _run_agent_sandbox(self, task, snapshot_path, task_index, total_tasks, slot_id=None,
                                     base_snapshot_path=None, patch_path=None):
            return cp.TAG["instance_id"], base_snapshot_path

    before = inspect.signature(Evaluator._run_agent_sandbox)
    ae.tag_captures_with_task(Evaluator)
    ae.tag_captures_with_task(Evaluator)  # idempotent: wrapping twice would still work, but is skipped
    assert inspect.signature(Evaluator._run_agent_sandbox) == before
    assert Evaluator._run_agent_sandbox.__wrapped__.__name__ == "_run_agent_sandbox"
    assert not hasattr(Evaluator._run_agent_sandbox.__wrapped__, "__wrapped__")
    import asyncio

    got = asyncio.run(Evaluator()._run_agent_sandbox(Task(), "snap", 0, 1, base_snapshot_path="b"))
    assert got == ("rich_3718", "b")


def test_find_adapters_layouts_and_errors(tmp_path):
    for root in (tmp_path / "flat", tmp_path / "nested" / "adapters"):
        (root / "ft0").mkdir(parents=True)
        for f in ae.ADAPTER_FILES:
            (root / "ft0" / f).write_text("x")
    assert [a.name for a in ae.find_adapters(tmp_path / "flat")] == ["ft0"]
    assert [a.name for a in ae.find_adapters(tmp_path / "nested")] == ["ft0"]
    (tmp_path / "flat" / "ft0" / "adapter_model.safetensors").unlink()
    with pytest.raises(SystemExit, match="missing"):
        ae.find_adapters(tmp_path / "flat")
    with pytest.raises(SystemExit, match="no <name>/adapter_config.json"):
        ae.find_adapters(tmp_path / "empty")
