"""Tests for scripts/make_split.py. Run with: python -m pytest -q"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_split as ms  # noqa: E402


def task(iid, repo, created, commit=None):
    return {"instance_id": iid, "repo": repo, "created_at": created, "base_commit": commit or iid, "patch": "+x\n"}


TASKS = [
    task("a1", "o/a", "2024-01-01"), task("a2", "o/a", "2025-01-01"), task("a3", "o/a", "2026-01-01"),
    task("a4", "o/a", "2026-02-01"),
    task("b1", "o/b", "2024-01-01", commit="shared"), task("b2", "o/b", "2026-03-01", commit="shared"),
    task("bad", "o/b", "2026-05-01"),
]
HEALTHY = {t["instance_id"] for t in TASKS} - {"bad"}


def test_unhealthy_tasks_are_excluded_everywhere():
    split = ms.make_split(TASKS, HEALTHY, dev_size=3)
    assert split["excluded_unhealthy"] == ["bad"]
    assert "bad" not in split["dev"] + split["train"]


def test_newest_tasks_per_repo_go_to_dev():
    split = ms.make_split(TASKS, HEALTHY, dev_size=3)
    assert {"a3", "a4"} <= set(split["dev"])
    assert {"a1", "a2"} <= set(split["train"])


def test_snapshot_sharing_tasks_stay_together():
    split = ms.make_split(TASKS, HEALTHY, dev_size=3)
    assert ("b1" in split["dev"]) == ("b2" in split["dev"])


def test_dev_and_train_do_not_overlap_and_cover_all_healthy():
    split = ms.make_split(TASKS, HEALTHY, dev_size=3)
    assert not set(split["dev"]) & set(split["train"])
    assert set(split["dev"]) | set(split["train"]) == HEALTHY


def test_allocate_is_proportional_and_exact():
    alloc = ms.allocate(33, {"fastapi": 60, "rich": 40, "requests": 10, "httpx": 1})
    assert sum(alloc.values()) == 33
    assert alloc["fastapi"] > alloc["rich"] > alloc["requests"]
