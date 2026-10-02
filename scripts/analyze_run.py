#!/usr/bin/env python3
"""Step 2d: sort every task of an agent_eval run into one failure category.

Reads the harness output of a run (run_summary.json, results/traces/, results/patches/,
results/test_outputs/) and prints one line per task plus a count per category. Categories are
checked in this order; the first that matches wins:

    pass            the hidden tests passed
    loop            the same tool call (name + arguments) repeated 5+ times in a row
    timeout         hit the 4.5-minute session limit
    turn_limit      hit the max_turns limit
    touched_tests   the patch edits test files, so the hidden tests could not be applied
    no_patch        submitted nothing (or never called submit_patch)
    budget_out      used all tool calls; the patch it had was wrong
    wrong_fix       finished normally with a patch that fails the hidden tests

Usage:
    python scripts/analyze_run.py eval/runs/v1_dev [--json eval/runs/v1_dev/analysis.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

LOOP_MIN = 5
ORDER = ["pass", "loop", "timeout", "turn_limit", "touched_tests", "no_patch", "budget_out", "wrong_fix"]


def tool_calls(trace: dict) -> list[tuple[str, str]]:
    calls = []
    for step in trace.get("steps", []):
        for tc in step.get("tool_calls") or []:
            calls.append((tc.get("function_name", ""), json.dumps(tc.get("arguments"), sort_keys=True)))
    return calls


def longest_repeat(calls: list[tuple[str, str]]) -> tuple[int, str]:
    best, run, best_call = 0, 0, ""
    for i, call in enumerate(calls):
        run = run + 1 if i and call == calls[i - 1] else 1
        if run > best:
            best, best_call = run, f"{call[0]} {call[1][:80]}"
    return best, best_call


def patch_files(patch: str) -> list[str]:
    return re.findall(r"^diff --git a/(\S+)", patch, flags=re.M)


def classify(task: dict, calls: list, repeat: int, files: list[str], budget: int) -> str:
    error = task["error"]
    if task["resolved"]:
        return "pass"
    if repeat >= LOOP_MIN:
        return "loop"
    if "session timeout" in error:
        return "timeout"
    if "turns budget" in error:
        return "turn_limit"
    if "test_patch" in error or any(re.search(r"(^|/)tests?/|test_[^/]*\.py$", f) for f in files):
        return "touched_tests"
    if not task["patch_chars"]:
        return "no_patch"
    if len(calls) >= budget:
        return "budget_out"
    return "wrong_fix"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--budget", type=int, default=40, help="max_tool_calls of the run")
    parser.add_argument("--json", type=Path, help="also write the per-task rows here")
    args = parser.parse_args()

    summary = json.loads((args.run_dir / "run_summary.json").read_text())
    results = args.run_dir / "results"
    rows = []
    for task in summary["tasks"]:
        tid = task["instance_id"]
        trace_path = results / "traces" / f"trace_{tid}.json"
        trace = json.loads(trace_path.read_text()) if trace_path.exists() else {}
        calls = tool_calls(trace)
        repeat, repeated = longest_repeat(calls)
        patch_path = results / "patches" / f"{tid}.patch"
        files = patch_files(patch_path.read_text()) if patch_path.exists() else []
        category = classify(task, calls, repeat, files, args.budget)
        tools = collections.Counter(name for name, _ in calls)
        rows.append({"instance_id": tid, "category": category, "tool_calls": len(calls),
                     "longest_repeat": repeat, "repeated_call": repeated if repeat >= LOOP_MIN else "",
                     "submits": tools["submit_patch"], "edits": tools["edit_file"] + tools["write_file"],
                     "files": files, "seconds": task["seconds"], "error": task["error"][:120]})

    for r in rows:
        print(f"{r['instance_id']:16} {r['category']:13} calls={r['tool_calls']:>2} repeat={r['longest_repeat']:>2} "
              f"edits={r['edits']:>2} submits={r['submits']} files={','.join(r['files'])[:60]}")
        if r["repeated_call"]:
            print(f"{'':16}   repeated: {r['repeated_call']}")
    counts = collections.Counter(r["category"] for r in rows)
    print(f"\n{summary['resolved']}/{summary['total']} resolved ({summary['resolution_rate']:.1%})")
    for cat in ORDER:
        if counts[cat]:
            print(f"  {cat:13} {counts[cat]:>2}")
    looping = sum(r["longest_repeat"] >= LOOP_MIN for r in rows)
    print(f"tasks with a {LOOP_MIN}+ repeat anywhere: {looping}/{len(rows)}")
    if args.json:
        args.json.write_text(json.dumps({"counts": dict(counts), "tasks": rows}, indent=2))


if __name__ == "__main__":
    main()
