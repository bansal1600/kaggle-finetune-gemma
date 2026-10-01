#!/usr/bin/env python3
"""Step 2a: split the public tasks into a dev set (for testing) and a train set (for fine-tuning data).

Rules (see concepts.md -> "Train / validation split", "Data contamination"):
- Only tasks the grader can actually score ("healthy" in the grader check) are used.
- Per repo, the NEWEST tasks go to dev, in proportion to each repo's share of healthy tasks.
  Newer issues are the least likely to be in Gemma's training data, so dev is closest to the
  unseen hidden tasks; and within each repo, every train task is older than every dev task.
- Tasks that share a repo snapshot stay on the same side, so dev never leaks into train.

Usage:
    python scripts/make_split.py --grader-check eval/grader_check.csv            # writes eval/splits.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_tasks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fix_size(task: dict) -> str:
    lines = sum(1 for l in task["patch"].splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---")))
    return "small" if lines <= 10 else "medium" if lines <= 40 else "large"


def allocate(target: int, counts: dict[str, int]) -> dict[str, int]:
    """Split `target` across repos in proportion to `counts` (largest-remainder rounding)."""
    total = sum(counts.values())
    exact = {repo: target * n / total for repo, n in counts.items()}
    alloc = {repo: int(x) for repo, x in exact.items()}
    for repo in sorted(exact, key=lambda r: exact[r] - alloc[r], reverse=True)[: target - sum(alloc.values())]:
        alloc[repo] += 1
    return alloc


def make_split(tasks: list[dict], healthy: set[str], dev_size: int) -> dict:
    usable = [t for t in tasks if t["instance_id"] in healthy]
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for t in usable:
        by_repo[t["repo"]].append(t)
    quota = allocate(dev_size, {repo: len(ts) for repo, ts in by_repo.items()})

    dev: set[str] = set()
    for repo, ts in by_repo.items():
        newest_first = sorted(ts, key=lambda t: t["created_at"], reverse=True)
        dev.update(t["instance_id"] for t in newest_first[: quota[repo]])

    # Keep tasks that share a snapshot (same base_commit) on the same side.
    by_commit: dict[str, list[str]] = defaultdict(list)
    for t in usable:
        by_commit[t["base_commit"]].append(t["instance_id"])
    for ids in by_commit.values():
        if len(ids) > 1 and any(i in dev for i in ids):
            dev.update(ids)

    train = sorted(t["instance_id"] for t in usable if t["instance_id"] not in dev)
    return {
        "method": "per-repo newest healthy tasks -> dev (proportional), rest -> train; snapshot-sharing tasks kept together",
        "dev": sorted(dev),
        "train": train,
        "excluded_unhealthy": sorted(t["instance_id"] for t in tasks if t["instance_id"] not in healthy),
    }


def describe(name: str, ids: list[str], tasks_by_id: dict[str, dict]) -> str:
    ts = [tasks_by_id[i] for i in ids]
    if not ts:
        return f"{name:9} 0 tasks"
    repos = Counter(t["repo"].split("/")[1] for t in ts)
    sizes = Counter(fix_size(t) for t in ts)
    dates = sorted(t["created_at"][:10] for t in ts)
    return (f"{name:9} {len(ts):3} tasks | " + ", ".join(f"{r} {n}" for r, n in repos.most_common())
            + f" | fixes: " + ", ".join(f"{s} {sizes[s]}" for s in ("small", "medium", "large"))
            + f" | created {dates[0]} .. {dates[-1]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=Path, default=Path("data/raw/tasks.jsonl"))
    parser.add_argument("--grader-check", type=Path, required=True, help="grader_check.csv from the Kaggle run")
    parser.add_argument("--dev-size", type=int, default=33)
    parser.add_argument("--out", type=Path, default=Path("eval/splits.json"))
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    with args.grader_check.open() as fh:
        healthy = {row["instance_id"] for row in csv.DictReader(fh) if row["healthy"] == "True"}
    split = make_split(tasks, healthy, args.dev_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(split, indent=2) + "\n")
    by_id = {t["instance_id"]: t for t in tasks}
    print(describe("dev", split["dev"], by_id))
    print(describe("train", split["train"], by_id))
    print(describe("excluded", split["excluded_unhealthy"], by_id))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
