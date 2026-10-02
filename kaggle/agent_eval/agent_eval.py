"""Step 2c: run our agent configs on the dev tasks, the way the real scorer would.

Same model (gemma-4-31b-it-qat-w4a16-ct served by vLLM on 4x L4), same harness (swegemma), same
per-task budgets (the config's eval_config.yaml), tasks run one at a time like the scorer, and each
task sandbox gets the scorer's package set (kaggle_common.use_scorer_environment).

Several configs (variants) can run in one notebook, one after another on the same model server,
so the ~6-minute vLLM start-up is paid once. They run in the order given; each is saved as soon
as it finishes, so a notebook that runs out of time still keeps the finished ones.

Built and pushed with:
    python scripts/kaggle_build.py agent_eval --submission submission --split dev --push
    python scripts/kaggle_build.py agent_eval --split dev --push \
        --submission v2c=experiments/v2c_both --submission v2a=experiments/v2a_sampling

Outputs in /kaggle/working (or OUT_DIR), per variant under runs/<variant>/:
    agent/              the config that ran
    results/            the harness's own output: summary.json, task_results.jsonl, patches/,
                        test_outputs/, traces/ (full agent transcripts), logs/
    run_summary.json    one line per task: resolved, error, tool calls, time, patch size
plus runs/overview.json with one line per variant.

Environment variables for local testing (defaults match Kaggle):
    DATA_DIR, WHEELHOUSE, OUT_DIR   as in the grader check
    MODEL_URL       use an already-running OpenAI-compatible server (e.g. scripts/mock_llm.py)
                    instead of starting vLLM; skips the GPU packages
    SUBMISSION_DIR  agent config folder to use when nothing is embedded (one variant, "local")
    TASK_IDS        comma-separated override of the embedded task list
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/kaggle/input/competitions/gemma-4-developer-agent"))
WHEELHOUSE = Path(os.environ.get("WHEELHOUSE", "/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
EXTRA_WHEELS = Path(os.environ.get("EXTRA_WHEELS", "/kaggle/input/datasets/guaravbansal/gemma-agent-extra-wheels"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2"))
MODEL_URL = os.environ.get("MODEL_URL", "")
TARGET_MODEL = "gemma-4-31b-it-qat-w4a16-ct"

VARIANTS: dict[str, dict[str, str]] = {}  # @embed variants
EMBEDDED_TASK_IDS: list[str] = []  # @embed task_ids

from kaggle_common import install_harness, use_scorer_environment  # inlined by scripts/kaggle_build.py


def write_agent_configs(runs_dir: Path) -> dict[str, Path]:
    """Materialize each variant's config: embedded files on Kaggle, SUBMISSION_DIR locally."""
    dirs = {}
    if VARIANTS:
        for name, files in VARIANTS.items():
            dest = runs_dir / name / "agent"
            for rel, text in files.items():
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                (dest / rel).write_text(text, encoding="utf-8")
            dirs[name] = dest
    else:
        import shutil

        dest = runs_dir / "local" / "agent"
        shutil.copytree(Path(os.environ["SUBMISSION_DIR"]), dest, dirs_exist_ok=True)
        dirs["local"] = dest
    return dirs


def start_model(agent_dir: Path):
    """Return (model registry, adapter manifest) for the agent, starting vLLM unless MODEL_URL is set."""
    from adk_submission import ModelRegistry, discover_adapters
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

    adapters = discover_adapters(str(agent_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    if MODEL_URL:
        from google.adk.models.lite_llm import LiteLlm

        models = ModelRegistry()
        models.register(TARGET_MODEL, LiteLlm(model=f"openai/{TARGET_MODEL}", api_base=MODEL_URL, api_key="EMPTY"))
        return models, adapters

    # Same server settings as the organizers' getting-started notebook.
    import torch
    from adk_submission import VllmConfig, VllmServer
    from swegemma.models.discovery import validate_single_declared_model

    declared_model = validate_single_declared_model(agent_dir)
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    vllm_cfg = VllmConfig(
        model=str(MODEL_PATH),
        port=8000,
        host="127.0.0.1",
        tool_call_parser="gemma4",
        reasoning_parser="gemma4",
        max_model_len=32768,
        dtype="bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "auto",
        gpu_memory_utilization=0.90,
        enable_auto_tool_choice=True,
        enable_lora=True,
        max_loras=8,
        max_lora_rank=128,
        tensor_parallel_size=4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1),
        startup_timeout=60 * 20,
    )
    start = time.time()
    server = VllmServer(vllm_cfg, adapter_manifest=adapters)
    server.start()
    print(f"vLLM ready on {server.base_url} after {(time.time() - start) / 60:.1f} min ({gpu_count} GPUs)", flush=True)
    models = server.create_model_registry(aliases=[declared_model, TARGET_MODEL], model_prefix="openai/", api_key="EMPTY")
    return models, adapters


def main() -> None:
    # Same environment as the organizers' notebook (offline vLLM + LiteLLM routing).
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

    install_harness(WHEELHOUSE, gpu=not MODEL_URL)
    runs_dir = OUT_DIR / "runs"
    agent_dirs = write_agent_configs(runs_dir)
    use_scorer_environment(DATA_DIR / "wheels", Path(os.environ.get("SCORER_ENV", "/tmp/scorer_env")), EXTRA_WHEELS)

    import litellm

    litellm.drop_params = True
    # One server for all variants. Adapters (and the served model) come from the first variant;
    # variants with different adapters need separate notebooks.
    first = next(iter(agent_dirs.values()))
    models, adapters = start_model(first)
    task_ids = [t for t in os.environ.get("TASK_IDS", "").split(",") if t] or EMBEDDED_TASK_IDS
    overview = asyncio.run(run_variants(agent_dirs, models, adapters, task_ids, runs_dir))
    print("\n" + "\n".join(f"{o['variant']:20} {o['resolved']}/{o['total']} ({o['resolution_rate']:.1%}) "
                           f"in {o['hours']:.2f} h" for o in overview))


async def run_variants(agent_dirs: dict[str, Path], models, adapters, task_ids: list[str], runs_dir: Path) -> list:
    """Run each variant on the same tasks, in order, inside one event loop (one model client)."""
    from adk_submission import discover_adapters
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

    overview = []
    for name, agent_dir in agent_dirs.items():
        if discover_adapters(str(agent_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS) != adapters:
            raise SystemExit(f"variant {name} declares different adapters from the first variant")
        summary = await run_one(name, agent_dir, models, adapters, task_ids, runs_dir / name)
        overview.append({"variant": name, **{k: summary[k] for k in ("resolved", "total", "resolution_rate", "hours")}})
        (runs_dir / "overview.json").write_text(json.dumps(overview, indent=2))
    return overview


async def run_one(name: str, agent_dir: Path, models, adapters, task_ids: list[str], out: Path) -> dict:
    import yaml
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
    from swegemma.config import EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator

    # Per-task budgets exactly as the scorer reads them from the config's eval_config.yaml.
    eval_cfg_path = agent_dir / "eval_config.yaml"
    raw = yaml.safe_load(eval_cfg_path.read_text()) if eval_cfg_path.exists() else {}
    section = (raw or {}).get("evaluation", raw or {})
    limits, gen_constraints = build_submission_limits()
    config = EvalConfig(
        tasks_path=DATA_DIR / "tasks.jsonl",
        snapshots_dir=DATA_DIR / "snapshots",
        results_dir=out / "results",
        submission_dir=agent_dir,
        models=models,
        sandbox="subprocess",
        timeout_seconds=int(section.get("timeout_seconds", 300)),
        max_time_minutes=float(section.get("max_time_minutes", 60.0)),
        max_tool_calls=int(section.get("max_tool_calls", 100)),
        max_turns=int(section["max_turns"]) if section.get("max_turns") is not None else None,
        limits=limits,
        generation_constraints=gen_constraints,
        adapter_manifest=adapters,
        # Organizers' getting-started notebook values; the scorer's exact values are unconfirmed.
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
        graph_dir=str(DATA_DIR / "graphs"),
        embeddings_dir=str(DATA_DIR / "embeddings"),
        wheels_dir=DATA_DIR / "wheels",
        task_ids=task_ids or None,
        concurrency=1,  # the scorer runs tasks one after another on one model server
        display_mode="quiet",
    )
    print(f"\n=== {name}: {len(task_ids) or 'all'} tasks, budgets {dict(section)}", flush=True)
    start = time.time()
    result = await Evaluator(config).run()
    hours = (time.time() - start) / 3600

    rows = []
    for r in sorted(result.task_results, key=lambda r: r.instance_id):
        rows.append({
            "instance_id": r.instance_id,
            "resolved": bool(r.resolved),
            "error": (r.error or "")[:300],
            "tool_calls": r.tool_calls,
            "llm_calls": r.total_llm_calls,
            "seconds": round(r.duration_seconds or 0, 1),
            "patch_chars": len(r.agent_patch or ""),
        })
        print(f"{r.instance_id:18} {'PASS' if r.resolved else 'fail'}  calls={r.tool_calls}  "
              f"{(r.duration_seconds or 0) / 60:.1f} min  {(r.error or '')[:80]}", flush=True)
    summary = {"variant": name, "resolved": result.resolved, "total": result.total,
               "resolution_rate": result.resolution_rate, "hours": round(hours, 2),
               "python": sys.version.split()[0], "tasks": rows}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{name}: resolved {result.resolved}/{result.total} ({result.resolution_rate:.1%}) in {hours:.2f} h", flush=True)
    return summary


if __name__ == "__main__":
    main()
