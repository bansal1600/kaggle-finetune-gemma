"""FT-0 / FT-0R: does a LoRA adapter start, apply and fit on the scorer's vLLM (4x L4, patched vLLM 0.19.1)?

Each server is started the way the scorer starts it: discover_adapters() on a submission dir, then
adk_submission's VllmServer with the getting-started notebook's VllmConfig. adk then emits the LoRA
flags itself (--enable-lora --max-loras <#adapters> --max-lora-rank <rank bucket> --lora-modules).
One server serves both names: the base model (its path) and the adapter, so "base vs adapter" is
compared on identical memory settings.

MODE=dummy (plan Stage 1, FT-0): a random r16 adapter on all 7 projections with the k_eq_v
duplicate (adapter_tools.make_dummy; both A and B nonzero so its effect is visible). Servers:
    S2_r16_080   gmu 0.80, the HARNESS_README value. PASS-critical.
    S1_r16_090   gmu 0.90, the adk default.
    S2p_r16_080  only if test e failed on S2: gmu 0.80 + --max-num-batched-tokens 512, test e only.
    S2b_r8_080   only if S2 did not start for a reason other than the sampler-warmup OOM.
MODE=real (plan Stage 6, FT-0R + E0 parity): the trained adapter from ADAPTER_SRC as main_lora at
GMU (default 0.80), then the parity measurement on PARITY_DATA (sft_val.jsonl or its 4k slices).

Tests, riskiest last so an engine crash loses nothing (letters as in the plan):
    d  KV cache: /metrics cache_config_info + the 'Available KV cache memory', 'GPU KV cache size'
       and 'Maximum concurrency for 32,768 tokens' log lines
    c  adapter applied: temperature-0 top-5 logprobs, base name twice and adapter name twice
    a  a 28k-token prompt + 256 output tokens on both names (an old LoRA bug hung such prompts)
    b  streaming decode tok/s at a ~10k prompt, 512 tokens, 3 reps alternating names
    e  prompt_logprobs on a 16,384- then a 4,096-token prompt, both names, sampling GPU memory:
       full-vocabulary prompt logits are memory the vLLM profiler never reserved
    parity  (real mode) mean NLL over labelled tokens, base / adapter / base again

Built and pushed with:
    python scripts/kaggle_build.py ft0_check --env MODE=dummy --push
    python scripts/kaggle_build.py ft0_check --env MODE=real --env GMU=0.80 \
        --env ADAPTER_SRC=/kaggle/input/datasets/<owner>/gemma-ft-adapter \
        --env PARITY_DATA=/kaggle/input/datasets/<owner>/gemma-ft-sft/sft_val.jsonl \
        --dataset <owner>/gemma-ft-adapter --dataset <owner>/gemma-ft-sft --push

Outputs in OUT_DIR (/kaggle/working), rewritten after every test:
    results.json      per server: command, startup time, KV lines, each test, plus the verdict
    logs/<server>.log full vLLM log (adk deletes its temp log on stop, so it is copied first)
    env.json          provenance: wheel sizes, pip freeze, GPUs, adapter validation
    parity.json       real mode: per-sequence NLLs and the summary

Environment variables (defaults match Kaggle):
    MODE              dummy | real
    GMU               real mode gpu_memory_utilization (dummy mode uses 0.80 and 0.90)
    ADAPTER_SRC       real mode: dir holding adapter_config.json + adapter_model.safetensors
    PARITY_DATA       real mode: jsonl of {input_ids, labels} (labels -100 = not trained on)
    VLLM_EXTRA_ARGS   JSON list of extra vLLM flags for every server, e.g. ["--max-num-batched-tokens","512"]
    TESTS             comma list overriding the tests to run (dummy: d,c,a,b,e; real: d,c,a,parity)
    HF_NLL_DROP       real mode: training's val NLL drop (step 0 minus final, nats/token) to compare against
    PARITY_MAX        cap on the number of parity sequences (default all)
    WHEELHOUSE, MODEL_PATH, OUT_DIR, SUB_ROOT, PORT   for local dry runs
"""

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import requests

# @embed env

MODE = os.environ.get("MODE", "dummy")
GMU = float(os.environ.get("GMU", "0.80"))
WHEELHOUSE = Path(os.environ.get("WHEELHOUSE", "/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
ADAPTER_SRC = os.environ.get("ADAPTER_SRC", "")
PARITY_DATA = os.environ.get("PARITY_DATA", "")
VLLM_EXTRA_ARGS: list[str] = json.loads(os.environ.get("VLLM_EXTRA_ARGS", "[]"))
TESTS = [t for t in os.environ.get("TESTS", "d,c,a,b,e" if MODE == "dummy" else "d,c,a,parity").split(",") if t]
HF_NLL_DROP = float(os.environ["HF_NLL_DROP"]) if os.environ.get("HF_NLL_DROP") else None
PARITY_MAX = int(os.environ.get("PARITY_MAX", "0"))
SUB_ROOT = Path(os.environ.get("SUB_ROOT", "/tmp/ft0_sub"))  # outside OUT_DIR: not downloaded with the output
PORT = int(os.environ.get("PORT", "8000"))
BASE_URL = f"http://127.0.0.1:{PORT}"
WARMUP_OOM = "CUDA out of memory occurred when warming up sampler"  # vllm gpu_model_runner.py; forum 746503

from kaggle_common import install_harness  # inlined by scripts/kaggle_build.py
from adapter_tools import make_dummy, validate  # inlined by scripts/kaggle_build.py

RESULTS: dict = {"mode": MODE, "servers": {}}


def save() -> None:
    (OUT_DIR / "results.json").write_text(json.dumps(RESULTS, indent=1))


# ---- adapters and servers ----

def dummy_submission(name: str, rank: int) -> Path:
    """A submission dir holding only adapters/<name>/, sized from the served model's config.json."""
    sub = SUB_ROOT / name
    config = MODEL_PATH / "config.json"
    make_dummy(sub / "adapters" / name, rank, dup_k_as_v=True, config=config if config.is_file() else None)
    RESULTS.setdefault("adapters", {})[name] = validate(sub / "adapters" / name, real=True)
    return sub


def find_file(hint: str, name: str) -> Path:
    """Kaggle's dataset mount path has moved before: fall back to searching /kaggle/input."""
    p = Path(hint)
    for cand in (p / name, p / "main_lora" / name):
        if cand.is_file():
            return cand
    if p.is_file():
        return p
    for root in (p, Path("/kaggle/input")):
        hits = sorted(root.rglob(name)) if root.is_dir() else []
        if hits:
            print(f"{name} not at {hint}; using {hits[0]}", flush=True)
            return hits[0]
    raise SystemExit(f"{name} not found at {hint!r} or under /kaggle/input")


def parity_file() -> Path:
    return find_file(PARITY_DATA, Path(PARITY_DATA).name or "sft_val.jsonl")


def real_submission() -> Path:
    """Copy exactly the two adapter files into a fresh submission dir, as the shipped zip has them."""
    src = find_file(ADAPTER_SRC, "adapter_config.json").parent
    dest = SUB_ROOT / "real" / "adapters" / "main_lora"
    dest.mkdir(parents=True, exist_ok=True)
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(src / f, dest / f)
    report = validate(dest, real=True, strict=False)
    RESULTS.setdefault("adapters", {})["main_lora"] = {**report, "source": str(src)}
    save()
    if not report["ok"]:
        raise SystemExit("adapter failed validation (see results.json); fix it locally before spending GPU time")
    return dest.parents[1]


def log_facts(log: str) -> dict:
    """The memory numbers vLLM prints at startup, and any OOM or error lines."""
    def ints(pattern):
        return [int(m.replace(",", "")) for m in re.findall(pattern, log)]

    lines = log.splitlines()
    return {
        "available_kv_gib": [float(x) for x in re.findall(r"Available KV cache memory: ([\d.]+) GiB", log)],
        "kv_tokens": ints(r"GPU KV cache size: ([\d,]+) tokens"),
        "max_concurrency_32k": [float(x) for x in re.findall(r"Maximum concurrency for 32,768 tokens per request: ([\d.]+)x", log)],
        "sampler_warmup_oom": WARMUP_OOM in log,
        "oom_lines": [l[-300:] for l in lines if "out of memory" in l.lower()][:10],
        "lora_lines": [l[-300:] for l in lines if "lora" in l.lower()][:30],
        "error_lines": [l[-300:] for l in lines if "Traceback" in l or "Error" in l][-30:],
    }


def run_server(name: str, sub: Path, gmu: float, tests: list[str], extra_args: list[str] | None = None) -> dict:
    """Start one server exactly as the scorer would, run `tests`, save its log, stop it."""
    import torch
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

    row: dict = {"gmu": gmu, "started": False, "tests": {}}
    RESULTS["servers"][name] = row
    adapters = discover_adapters(str(sub), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    adapter = next(iter(adapters.adapters))
    row["adapters"] = {n: a.rank for n, a in adapters.adapters.items()}
    cfg = VllmConfig(  # agent_eval.py / the organizers' getting-started notebook
        model=str(MODEL_PATH), port=PORT, host="127.0.0.1", tool_call_parser="gemma4", reasoning_parser="gemma4",
        max_model_len=32768, dtype="bfloat16" if torch.cuda.is_bf16_supported() else "auto",
        gpu_memory_utilization=gmu, enable_auto_tool_choice=True, enable_lora=True, max_loras=8,
        max_lora_rank=128, tensor_parallel_size=4, startup_timeout=60 * 20,
        extra_args=[*VLLM_EXTRA_ARGS, *(extra_args or [])])
    server = VllmServer(cfg, adapter_manifest=adapters)
    row["cmd"] = " ".join(server.build_cmd()[3:])
    row["vllm_config"] = asdict(cfg)
    print(f"\n=== {name}: {row['cmd']}", flush=True)
    save()
    log = ""
    start = time.time()
    try:
        server.start()
        row.update(started=True, startup_s=round(time.time() - start))
        base = str(MODEL_PATH)
        row["served_models"] = [m["id"] for m in requests.get(f"{BASE_URL}/v1/models", timeout=30).json()["data"]]
        for test in tests:
            t0 = time.time()
            try:
                row["tests"][test] = TEST_FUNCS[test](server, base, adapter)
            except Exception as e:  # a dead engine shows up here as a connection error
                row["tests"][test] = {"ok": False, "error": repr(e)[:1000]}
            row["tests"][test]["seconds"] = round(time.time() - t0)
            print(f"  {test}: {json.dumps(row['tests'][test])[:400]}", flush=True)
            save()
            if not alive(server):
                row["engine_died_during"] = test
                break
    except Exception as e:  # start-up failure: adk's ServerStartupError carries the log
        row["error"] = repr(e)[-3000:]
        log = getattr(e, "output", None) or ""
    finally:
        log = server._read_captured_output() or log  # read before stop(): stop() deletes the file
        (OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "logs" / f"{name}.log").write_text(log)
        row.update(log_facts(log))
        server.stop()
        save()
        time.sleep(20)  # let the GPUs free before the next start
    return row


def alive(server) -> bool:
    return server.process is not None and server.process.poll() is None and server.is_healthy()


# ---- requests ----

def complete(model: str, prompt, max_tokens: int, timeout: float = 600, **kw) -> dict:
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, **kw}
    r = requests.post(f"{BASE_URL}/v1/completions", json=body, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


def tokenize(model: str, text: str, special: bool) -> list[int]:
    body = {"model": model, "prompt": text, "add_special_tokens": special}
    return requests.post(f"{BASE_URL}/tokenize", json=body, timeout=120).json()["tokens"]


_CORPUS: list[int] = []


def corpus(model: str, n: int) -> list[int]:
    """n token ids of real Python source (the stdlib), starting with BOS. Token-id prompts have an
    exact length, which a text prompt cut by characters does not. Files are tokenized one at a time
    (at most 50k characters each) so no request exceeds max_model_len."""
    if len(_CORPUS) < n:
        _CORPUS[:] = tokenize(model, "", special=True)  # just BOS
        for f in sorted(Path(os.path.dirname(os.__file__)).glob("*.py")):
            _CORPUS.extend(tokenize(model, f.read_text(errors="ignore")[:50_000], special=False))
            if len(_CORPUS) >= 40_000:
                break
    assert len(_CORPUS) >= n, (len(_CORPUS), n)
    return _CORPUS[:n]


def nonce_prompt(model: str, ids: list[int]) -> list[int]:
    """Fresh tokens right after BOS, so prefix caching cannot skip the prefill."""
    return ids[:1] + tokenize(model, f"# run {time.time_ns()}\n", special=False) + ids[1:]


class GpuMemory:
    """nvidia-smi memory.used per GPU (MiB): one sample before the request, then the peak of samples
    taken every 0.5 s in a background thread while it runs."""

    @staticmethod
    def sample() -> list[int]:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
            return [int(x) for x in out.split()]
        except Exception:  # no GPU (local dry run) or a slow driver: record nothing
            return []

    def __enter__(self):
        self.start = self.peak = self.sample()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(0.5):
            used = self.sample()
            if used:
                self.peak = [max(a, b) for a, b in zip(self.peak, used)] if self.peak else used

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


# ---- tests ----

def test_kv(server, base, adapter) -> dict:
    metrics = requests.get(f"{BASE_URL}/metrics", timeout=30).text
    facts = log_facts(server._read_captured_output())
    conc = facts["max_concurrency_32k"]
    return {"ok": bool(conc) and min(conc) >= 1.0,
            "cache_config_info": [l for l in metrics.splitlines() if l.startswith("vllm:cache_config_info")][:3],
            **{k: facts[k] for k in ("available_kv_gib", "kv_tokens", "max_concurrency_32k")}}


def test_applied(server, base, adapter) -> dict:
    """Same prompt at temperature 0: base twice and adapter twice must each agree (< 1e-3), and base
    vs adapter must differ (text, or > 0.05 in a first-token logprob)."""
    prompt = 'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n'
    runs = {k: complete(m, prompt, 32, timeout=120, logprobs=5)["choices"][0]
            for k, m in (("base1", base), ("base2", base), ("ad1", adapter), ("ad2", adapter))}
    first = {k: c["logprobs"]["token_logprobs"][0] for k, c in runs.items()}
    top = {k: c["logprobs"]["top_logprobs"][0] for k, c in runs.items()}
    common = set(top["base1"]) & set(top["ad1"])
    res = {
        "text": {k: c["text"] for k, c in runs.items()},
        "diff_base_base": abs(first["base1"] - first["base2"]),
        "diff_ad_ad": abs(first["ad1"] - first["ad2"]),
        "diff_base_ad": abs(first["base1"] - first["ad1"]),
        "top5_max_diff_base_ad": max((abs(top["base1"][t] - top["ad1"][t]) for t in common), default=None),
        "same_text_base_ad": runs["base1"]["text"] == runs["ad1"]["text"],
    }
    res["deterministic"] = res["diff_base_base"] < 1e-3 and res["diff_ad_ad"] < 1e-3
    res["applied"] = (not res["same_text_base_ad"]) or res["diff_base_ad"] > 0.05 \
        or (res["top5_max_diff_base_ad"] or 0) > 0.05
    res["ok"] = res["deterministic"] and res["applied"]
    return res


def test_long(server, base, adapter) -> dict:
    ids = corpus(base, 28_000)
    res = {}
    for label, model in (("base", base), ("adapter", adapter)):
        t0 = time.time()
        try:
            out = complete(model, ids, 256, timeout=600, ignore_eos=True)
            res[label] = {"ok": out["usage"]["completion_tokens"] == 256, "usage": out["usage"],
                          "s": round(time.time() - t0)}
        except Exception as e:  # a timeout means the long-prompt hang came back
            res[label] = {"ok": False, "error": repr(e)[:500], "s": round(time.time() - t0)}
    res["ok"] = res["base"]["ok"] and res["adapter"]["ok"]
    return res


def stream_speed(model: str, ids: list[int]) -> dict:
    body = {"model": model, "prompt": ids, "max_tokens": 512, "temperature": 0.0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    t0, times, usage = time.time(), [], None
    with requests.post(f"{BASE_URL}/v1/completions", json=body, stream=True, timeout=600) as r:
        for line in r.iter_lines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            if chunk.get("choices") and chunk["choices"][0].get("text"):
                times.append(time.time() - t0)
    n = (usage or {}).get("completion_tokens") or len(times)
    return {"ttft_s": round(times[0], 2) if times else None, "tokens": n,
            "decode_tok_s": round((n - 1) / (times[-1] - times[0]), 2) if len(times) > 1 else None}


def test_speed(server, base, adapter) -> dict:
    ids = corpus(base, 10_000)
    res = {"base": [], "adapter": []}
    for rep in range(3):
        order = (("base", base), ("adapter", adapter)) if rep % 2 == 0 else (("adapter", adapter), ("base", base))
        for label, model in order:
            res[label].append(stream_speed(model, nonce_prompt(base, ids)))
    med = {k: statistics.median(r["decode_tok_s"] for r in res[k]) for k in ("base", "adapter")}
    res["median_tok_s"] = med
    res["ratio_adapter_base"] = round(med["adapter"] / med["base"], 3)
    res["ok"] = res["ratio_adapter_base"] >= 0.85
    return res


def prompt_nll(model: str, ids: list[int], labels: list[int] | None = None, timeout: float = 900) -> dict:
    """Mean NLL (nats/token) of the prompt's own tokens, via prompt_logprobs=0 (the logprob of each
    actual prompt token given everything before it). With labels, only positions whose label is
    not -100 count; labels are aligned with input_ids (labels[i] is input_ids[i] or -100)."""
    out = complete(model, ids, 1, timeout=timeout, prompt_logprobs=0)
    plp = out["choices"][0]["prompt_logprobs"]
    if len(plp) != len(ids):
        raise RuntimeError(f"{len(plp)} prompt_logprobs for {len(ids)} tokens")
    positions = [i for i in range(1, len(ids)) if labels is None or labels[i] != -100]
    nll = [-plp[i][str(ids[i])]["logprob"] for i in positions]
    if not nll:  # a row with no labelled token carries no signal; the caller skips it
        return {"nll": None, "tokens": 0}
    return {"nll": sum(nll) / len(nll), "tokens": len(nll)}


def test_prompt_logprobs(server, base, adapter) -> dict:
    """Survives a 16k (then 4k) prompt_logprobs request? E0 parity needs it; it is not a PASS condition."""
    res = {}
    for n in (16_384, 4_096):
        ids = corpus(base, n)
        for label, model in (("base", base), ("adapter", adapter)):
            key = f"{n}_{label}"
            t0 = time.time()
            with GpuMemory() as mem:
                try:
                    res[key] = {"ok": True, **prompt_nll(model, ids)}
                except Exception as e:
                    res[key] = {"ok": False, "error": repr(e)[:500]}
            res[key].update(s=round(time.time() - t0), start_mib=mem.start, peak_mib=mem.peak)
            print(f"    {key}: {res[key]}", flush=True)
            if not alive(server):
                res[key]["engine_died"] = True
                res["ok"] = False
                return res
    res["ok_16k"] = res["16384_base"]["ok"] and res["16384_adapter"]["ok"]
    res["ok"] = all(v["ok"] for k, v in res.items() if isinstance(v, dict))
    return res


def test_parity(server, base, adapter) -> dict:
    """E0: does training's val NLL drop survive serving (NF4 training vs W4A16 serving)?"""
    path = parity_file()
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = rows[:PARITY_MAX] if PARITY_MAX else rows
    seqs = []
    for i, row in enumerate(rows):
        ids, labels = row["input_ids"], row["labels"]
        if not any(l != -100 for l in labels[1:]):  # no labelled token: nothing to measure on this row
            print(f"    parity {i}: skipped (no labelled tokens)", flush=True)
            continue
        seq = {"i": i, "meta": row.get("meta"), "len": len(ids),
               "label_mismatches": sum(1 for t, l in zip(ids, labels) if l not in (-100, t))}
        for label, model in (("base1", base), ("lora", adapter), ("base2", base)):
            seq[label] = prompt_nll(model, ids, labels)["nll"]
        seq["tokens"] = sum(1 for l in labels[1:] if l != -100)
        seqs.append(seq)
        print(f"    parity {i}: {seq}", flush=True)
        # Written after every sequence: if a long one OOMs the engine, the shorter results survive.
        (OUT_DIR / "parity.json").write_text(json.dumps({"summary": None, "sequences": seqs}, indent=1))
    # Token-weighted means, like HF's loss over all labelled tokens.
    total = sum(s["tokens"] for s in seqs)
    if not seqs or not total:
        raise RuntimeError(f"{path}: no sequence with labelled tokens to compare")
    mean = {k: sum(s[k] * s["tokens"] for s in seqs) / total for k in ("base1", "lora", "base2")}
    summary = {"file": str(path), "sequences": len(seqs), "tokens": total, "mean_nll": mean,
               "base_instability": max(abs(s["base1"] - s["base2"]) for s in seqs),
               "vllm_drop": mean["base1"] - mean["lora"], "hf_drop": HF_NLL_DROP}
    summary["base_stable"] = summary["base_instability"] < 1e-3
    summary["drop_ratio"] = summary["vllm_drop"] / HF_NLL_DROP if HF_NLL_DROP else None
    # Without HF_NLL_DROP the 50% rule cannot be checked; at least the adapter must lower the NLL.
    summary["ok"] = summary["base_stable"] and summary["vllm_drop"] > 0 and (
        summary["drop_ratio"] is None or summary["drop_ratio"] >= 0.5)
    (OUT_DIR / "parity.json").write_text(json.dumps({"summary": summary, "sequences": seqs}, indent=1))
    return summary


TEST_FUNCS = {"d": test_kv, "c": test_applied, "a": test_long, "b": test_speed, "e": test_prompt_logprobs,
              "parity": test_parity}


# ---- verdict and provenance ----

def verdict(row: dict) -> dict:
    """The plan's PASS criteria for one server (Stage 1 for dummy, Stage 6 for real)."""
    t = row.get("tests", {})
    rank = max(row.get("adapters", {}).values(), default=None)
    checks = {
        "started": row.get("started", False),
        "lora_flags": f"--enable-lora --max-loras 1 --max-lora-rank {rank} " in row.get("cmd", ""),
        "no_sampler_warmup_oom": not row.get("sampler_warmup_oom", False),
        "concurrency_32k_ge_1": t.get("d", {}).get("ok", False),
        "long_28k_both_names": t.get("a", {}).get("ok", False),
        "adapter_applied_and_deterministic": t.get("c", {}).get("ok", False),
    }
    if "b" in t:
        checks["adapter_decode_ge_85pct"] = t["b"].get("ok", False)
    if "parity" in t:
        checks["parity"] = t["parity"].get("ok", False)
    return {"pass": all(checks.values()), **checks}


def write_env() -> None:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
        except Exception as e:
            return repr(e)

    env = {
        "python": sys.version, "mode": MODE, "gmu": GMU, "vllm_extra_args": VLLM_EXTRA_ARGS, "tests": TESTS,
        "adapter_src": ADAPTER_SRC, "parity_data": PARITY_DATA, "model_path": str(MODEL_PATH),
        # Later stages compare these sizes (vllm 433,132,704 B) to detect harness drift.
        "wheelhouse": {w.name: w.stat().st_size for w in sorted(WHEELHOUSE.glob("*.whl"))},
        "nvidia_smi": run(["nvidia-smi", "-L"]).splitlines(),
        "pip_freeze": run([sys.executable, "-m", "pip", "freeze"]).splitlines(),
    }
    (OUT_DIR / "env.json").write_text(json.dumps(env, indent=1))


def main() -> None:
    # Same environment as the organizers' notebook (offline vLLM).
    for key, value in {
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "TRANSFORMERS_NO_TF": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "1200",
        "VLLM_NO_USAGE_STATS": "1",
        "OTEL_SDK_DISABLED": "true",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }.items():
        os.environ.setdefault(key, value)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    install_harness(WHEELHOUSE, gpu=True)
    write_env()
    servers = RESULTS["servers"]

    if MODE == "dummy":
        sub16 = dummy_submission("ft0_r16", 16)
        s2 = run_server("S2_r16_080", sub16, 0.80, TESTS)
        run_server("S1_r16_090", sub16, 0.90, TESTS)
        e = s2["tests"].get("e")
        if s2["started"] and e is not None and not e.get("ok_16k", False):
            run_server("S2p_r16_080", sub16, 0.80, ["e"], ["--max-num-batched-tokens", "512"])
        if not s2["started"] and not s2.get("sampler_warmup_oom"):
            run_server("S2b_r8_080", dummy_submission("ft0_r8", 8), 0.80, TESTS)
        RESULTS["verdict"] = {name: verdict(row) for name, row in servers.items() if name != "S2p_r16_080"}
        RESULTS["verdict"]["FT0_PASS"] = RESULTS["verdict"]["S2_r16_080"]["pass"]
        if "S2b_r8_080" in servers:  # S2 failed for a non-warmup reason: r8 is the fallback rank
            RESULTS["verdict"]["train_r8_instead"] = RESULTS["verdict"]["S2b_r8_080"]["pass"]
        # Which way Stage 6 runs parity: full sequences, the 512-token prefill chunk flag, or 4k slices.
        # Judged at gmu 0.80 (Stage 6's setting): S2, or the r8 fallback server if S2 did not start.
        if e is None and "S2b_r8_080" in servers:
            e = servers["S2b_r8_080"]["tests"].get("e")
        e_s2p = servers.get("S2p_r16_080", {}).get("tests", {}).get("e", {})
        RESULTS["verdict"]["parity_plan"] = (None if e is None else "full" if e.get("ok_16k") else
                                             "max_num_batched_tokens_512" if e_s2p.get("ok_16k") else "slices_4k")
    elif MODE == "real":
        name = f"R_main_lora_{round(GMU * 100):03d}"
        if "parity" in TESTS:
            parity_file()  # a wrong PARITY_DATA path should fail before the 7-minute server start
        run_server(name, real_submission(), GMU, TESTS)
        RESULTS["verdict"] = {name: verdict(servers[name])}
    else:
        raise SystemExit(f"MODE must be dummy or real, not {MODE!r}")
    save()
    print("\n" + json.dumps(RESULTS["verdict"], indent=1), flush=True)


if __name__ == "__main__":
    main()
