"""Step 2b: can the official grader score each public task in our Kaggle environment?

For every task in tasks.jsonl, run the organizers' own verifier (swegemma's Evaluator,
phase 2 unchanged) twice:

    gold  -> apply the task's reference fix   (expected: resolved)
    none  -> apply no change at all           (expected: NOT resolved)

A task is "healthy" when gold passes and none fails. Unhealthy tasks would silently count
against any agent in local evaluation, so we leave them out of our dev/train splits.

No model is loaded, so this runs on a free CPU notebook. Outputs (in /kaggle/working or OUT_DIR):
    grader_check.csv        one row per task: gold/none result, healthy flag, errors
    results_gold/, results_none/   the harness's own summary.json, task_results.jsonl, test logs

Environment variables (for local testing; the defaults match the Kaggle mounts):
    DATA_DIR      competition data root      (/kaggle/input/competitions/gemma-4-developer-agent)
    WHEELHOUSE    organizers' wheel dataset  (/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse)
    OUT_DIR       where to write results     (/kaggle/working)
    TASK_IDS      comma-separated subset of instance_ids (default: all)
    CONCURRENCY   tasks verified in parallel (default: 4, one per CPU on a Kaggle CPU notebook)
"""

import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/kaggle/input/competitions/gemma-4-developer-agent"))
WHEELHOUSE = Path(os.environ.get("WHEELHOUSE", "/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
EXTRA_WHEELS = Path(os.environ.get("EXTRA_WHEELS", "/kaggle/input/datasets/guaravbansal/gemma-agent-extra-wheels"))
TASK_IDS = [t for t in os.environ.get("TASK_IDS", "").split(",") if t]
CONCURRENCY = int(os.environ.get("CONCURRENCY", "4"))

from kaggle_common import install_harness, use_scorer_environment  # inlined by scripts/kaggle_build.py


def main() -> None:
    install_harness(WHEELHOUSE, gpu=False)
    use_scorer_environment(DATA_DIR / "wheels", Path(os.environ.get("SCORER_ENV", "/tmp/scorer_env")), EXTRA_WHEELS)

    from adk_submission import ModelRegistry
    from adk_eval_core.tracing import SessionTrace
    from swegemma.config import EvalConfig
    from swegemma.evaluate import Evaluator
    from swegemma.models import load_tasks

    class GoldEvaluator(Evaluator):
        """Phase 1 replaced by 'the agent produced exactly the reference fix'. Phase 2 untouched."""

        async def _run_agent_sandbox(self, task, snapshot_path, task_index, total_tasks, **kwargs):
            return task.patch, None, SessionTrace()

    tasks_path = DATA_DIR / "tasks.jsonl"
    all_ids = [t.instance_id for t in load_tasks(tasks_path)]
    ids = TASK_IDS or all_ids
    dummy_submission = OUT_DIR / "no_agent"  # required by EvalConfig, never used here
    dummy_submission.mkdir(parents=True, exist_ok=True)

    def make_config(mode: str) -> EvalConfig:
        return EvalConfig(
            tasks_path=tasks_path,
            snapshots_dir=DATA_DIR / "snapshots",
            results_dir=OUT_DIR / f"results_{mode}",
            submission_dir=dummy_submission,
            models=ModelRegistry(),
            sandbox="subprocess",
            wheels_dir=DATA_DIR / "wheels",
            graph_dir=str(DATA_DIR / "graphs"),
            embeddings_dir=str(DATA_DIR / "embeddings"),
            timeout_seconds=300,
            task_ids=ids,
            skip_agent_patch=(mode == "none"),
            concurrency=CONCURRENCY,
            display_mode="quiet",
        )

    outcomes: dict[str, dict] = {i: {"instance_id": i} for i in ids}
    for mode, cls in (("gold", GoldEvaluator), ("none", Evaluator)):
        start = time.time()
        print(f"\n=== {mode}: verifying {len(ids)} tasks (concurrency {CONCURRENCY}) ===", flush=True)
        result = asyncio.run(cls(make_config(mode)).run())
        print(f"{mode}: {result.resolved}/{result.total} resolved in {(time.time() - start) / 60:.1f} min", flush=True)
        for r in result.task_results:
            row = outcomes.setdefault(r.instance_id, {"instance_id": r.instance_id})
            row[f"{mode}_resolved"] = bool(r.resolved)
            row[f"{mode}_exit_code"] = r.test_exit_code
            row[f"{mode}_seconds"] = round(r.duration_seconds or 0, 1)
            row[f"{mode}_error"] = (r.error or "")[:300]

    repo_of = {t.instance_id: t.repo for t in load_tasks(tasks_path)}
    rows = []
    for i in ids:
        row = outcomes[i]
        row["repo"] = repo_of.get(i, "")
        row["healthy"] = bool(row.get("gold_resolved")) and not row.get("none_resolved", True)
        rows.append(row)

    fields = ["instance_id", "repo", "healthy", "gold_resolved", "none_resolved", "gold_exit_code",
              "none_exit_code", "gold_seconds", "none_seconds", "gold_error", "none_error"]
    with open(OUT_DIR / "grader_check.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    healthy = sum(r["healthy"] for r in rows)
    print(f"\nHealthy tasks: {healthy}/{len(rows)}")
    for r in rows:
        if not r["healthy"]:
            why = "gold fails" if not r.get("gold_resolved") else "passes without any fix"
            print(f"  unhealthy: {r['instance_id']:18} {why}  {r.get('gold_error') or ''}")
    (OUT_DIR / "grader_check_summary.json").write_text(json.dumps(
        {"tasks": len(rows), "healthy": healthy, "python": sys.version.split()[0]}, indent=2))


if __name__ == "__main__":
    main()
