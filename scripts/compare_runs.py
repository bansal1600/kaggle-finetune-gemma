#!/usr/bin/env python3
"""Compare agent_eval runs task by task, with the mechanism metrics behind the score.

A dev score of 33 tasks moves by +-2 from luck alone, so a config is judged on more than its score:
which tasks it solves (pass matrix), whether it keeps the always-solved "core" tasks, how many of the
"fixable" tasks it gets (eval/fairness.json), and whether the mechanism it targets actually changed
(garbled tool calls, repeated calls, script edits, ...). It also projects a 120-task scorer run
against the 12-hour limit.

Usage:
    python scripts/compare_runs.py v1=eval/runs/v1_dev v2a=eval/runs/v2_batch/runs/v2a_sampling ...
    python scripts/compare_runs.py ... --json eval/runs/compare.json

Each run directory holds run_summary.json and results/ (task_results.jsonl, traces/, patches/).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MALFORMED = "mandatory input parameters"
SCRATCH = re.compile(r"^diff --git a/(\S+)", re.M)
SETUP_SECONDS = 45      # measured per-task sandbox setup/teardown on Kaggle (4-60 s); conservative
STARTUP_MINUTES = 10    # vLLM start-up plus harness install
HIDDEN_TASKS = 120


def calls_of(trace: dict):
    """Yield (tool name, arguments json, observation text, step) for every tool call."""
    for step in trace.get("steps", []):
        if step.get("source") != "agent":
            continue
        obs = json.dumps(step.get("observation") or "")
        for call in step.get("tool_calls") or []:
            yield call.get("function_name", ""), json.dumps(call.get("arguments"), sort_keys=True, default=str), obs, step


def is_script_edit(name: str, args: str) -> bool:
    return name == "run_command" and "python" in args and "write_text" in args


def task_metrics(run: Path, tid: str, result: dict) -> dict:
    trace_path = run / "results" / "traces" / f"trace_{tid}.json"
    trace = json.loads(trace_path.read_text()) if trace_path.exists() else {}
    prev, n, malformed, repeats, edits, script_edits, assert0, undeclared = None, 0, 0, 0, 0, 0, 0, 0
    prompt_tokens, compactions = [], 0
    for step in trace.get("steps", []):
        metrics = step.get("metrics") or {}
        tokens = metrics.get("prompt_tokens") if isinstance(metrics, dict) else None
        if tokens:
            if prompt_tokens and prompt_tokens[-1] > 10000 and tokens < 0.5 * prompt_tokens[-1]:
                compactions += 1
            prompt_tokens.append(tokens)
    for name, args, obs, _ in calls_of(trace):
        n += 1
        repeats += (name, args) == prev
        prev = (name, args)
        malformed += MALFORMED in obs
        undeclared += "not found" in obs and "Tool" in obs
        if name == "edit_file":
            edits += 1
        if is_script_edit(name, args):
            script_edits += 1
            assert0 += "AssertionError: 0" in obs
    patch_path = run / "results" / "patches" / f"{tid}.patch"
    files = SCRATCH.findall(patch_path.read_text()) if patch_path.exists() else []
    error = result.get("error") or ""
    return {
        "resolved": bool(result.get("resolved")),
        "calls": n, "malformed": malformed, "repeats": repeats, "edit_file": edits,
        "script_edits": script_edits, "script_assert0": assert0, "undeclared_tool": undeclared,
        "compacted": compactions > 0, "max_prompt_tokens": max(prompt_tokens, default=0),
        "stray_files": sum(1 for f in files if Path(f).name.startswith(("repro", "test_repro", "debug", "tmp"))),
        "empty_patch": not result.get("agent_patch_size") and not files,
        "timeout": "session timeout" in error, "turn_limit": "turns budget" in error,
        "overflow": "context" in error.lower() and ("length" in error.lower() or "window" in error.lower()),
        "seconds": float(result.get("duration_seconds") or 0),
    }


def load_run(run: Path) -> dict[str, dict]:
    results = {}
    for line in (run / "results" / "task_results.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            results[r["instance_id"]] = r
    return {tid: task_metrics(run, tid, r) for tid, r in sorted(results.items())}


def summarize(tasks: dict[str, dict], fairness: dict, max_minutes: float) -> dict:
    n = len(tasks)
    calls = sum(t["calls"] for t in tasks.values()) or 1
    secs = sorted(t["seconds"] for t in tasks.values())
    mean = statistics.mean(secs)
    p90 = secs[min(n - 1, int(0.9 * n))]
    summary = {
        "resolved": sum(t["resolved"] for t in tasks.values()), "tasks": n,
        "core_kept": sum(tasks[t]["resolved"] for t in fairness.get("core", []) if t in tasks),
        "fixable_solved": sum(tasks[t]["resolved"] for t in fairness.get("fixable", []) if t in tasks),
        "unfair_solved": sum(tasks[t]["resolved"] for t in fairness.get("unfair", []) if t in tasks),
        "malformed": sum(t["malformed"] for t in tasks.values()),
        "tasks_10plus_malformed": sum(t["malformed"] >= 10 for t in tasks.values()),
        "repeat_pct": round(100 * sum(t["repeats"] for t in tasks.values()) / calls, 1),
        "edit_file_calls": sum(t["edit_file"] for t in tasks.values()),
        "script_edits": sum(t["script_edits"] for t in tasks.values()),
        "script_assert0": sum(t["script_assert0"] for t in tasks.values()),
        "undeclared_tool": sum(t["undeclared_tool"] for t in tasks.values()),
        "overflow": sum(t["overflow"] for t in tasks.values()),
        "compacted_tasks": sum(t["compacted"] for t in tasks.values()),
        "stray_files": sum(t["stray_files"] for t in tasks.values()),
        "empty_patches": sum(t["empty_patch"] for t in tasks.values()),
        "timeouts": sum(t["timeout"] for t in tasks.values()),
        "turn_limits": sum(t["turn_limit"] for t in tasks.values()),
        "mean_s": round(mean, 1), "p90_s": round(p90, 1), "max_s": round(secs[-1], 1),
        "proj_120_expected_h": round((HIDDEN_TASKS * (mean + SETUP_SECONDS) / 60 + STARTUP_MINUTES) / 60, 1),
        "proj_120_worst_h": round((HIDDEN_TASKS * (max_minutes + SETUP_SECONDS / 60) + STARTUP_MINUTES) / 60, 1),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", metavar="NAME=DIR")
    parser.add_argument("--fairness", type=Path, default=ROOT / "eval" / "fairness.json")
    parser.add_argument("--max-minutes", type=float, default=4.5, help="max_time_minutes, for the worst-case projection")
    parser.add_argument("--json", type=Path, help="write per-task metrics and summaries here")
    args = parser.parse_args()

    fairness = json.loads(args.fairness.read_text()) if args.fairness.exists() else {}
    runs = {}
    for spec in args.runs:
        name, _, path = spec.partition("=")
        runs[name] = load_run(Path(path))
    names = list(runs)
    label = {t: k for k in ("core", "fixable", "unfair") for t in fairness.get(k, [])}

    all_ids = sorted({t for r in runs.values() for t in r})
    print(f"{'task':16} {'label':8} " + " ".join(f"{n[:9]:>9}" for n in names))
    for tid in all_ids:
        marks = [("PASS" if runs[n][tid]["resolved"] else ".") if tid in runs[n] else "-" for n in names]
        if any(m == "PASS" for m in marks) or label.get(tid) != "unfair":
            print(f"{tid:16} {label.get(tid, ''):8} " + " ".join(f"{m:>9}" for m in marks))

    summaries = {n: summarize(r, fairness, args.max_minutes) for n, r in runs.items()}
    print()
    keys = list(next(iter(summaries.values())))
    print(f"{'metric':22} " + " ".join(f"{n[:9]:>9}" for n in names))
    for k in keys:
        print(f"{k:22} " + " ".join(f"{summaries[n][k]!s:>9}" for n in names))
    union = sum(any(runs[n].get(t, {}).get("resolved") for n in names) for t in all_ids)
    print(f"\nunion of passes: {union}/{len(all_ids)}")
    if args.json:
        args.json.write_text(json.dumps({"summaries": summaries, "tasks": runs}, indent=1))


if __name__ == "__main__":
    main()
