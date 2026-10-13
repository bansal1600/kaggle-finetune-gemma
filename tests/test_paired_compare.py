"""Tests for scripts/paired_compare.py on synthetic run directories. Run with: python -m pytest -q tests/test_paired_compare.py"""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import paired_compare as pc  # noqa: E402

CORE = [f"core{i}" for i in range(6)]
FIXABLE = [f"fix{i}" for i in range(8)]
TASKS = CORE + FIXABLE  # 14 tasks, the size of dev_dec
FAIRNESS = {"core": CORE, "fixable": FIXABLE, "unfair": []}


def make_run(root: Path, name: str, passed: set[str], seconds: float = 100.0, errors: dict | None = None,
             traces: dict | None = None, skip: set[str] = frozenset()) -> Path:
    """A minimal agent_eval run dir: run_summary.json, results/task_results.jsonl and optional traces."""
    run = root / name
    (run / "results" / "traces").mkdir(parents=True)
    rows = [{"instance_id": t, "resolved": t in passed, "duration_seconds": seconds, "agent_patch_size": 100,
             "error": (errors or {}).get(t)} for t in TASKS if t not in skip]
    (run / "results" / "task_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for t, trace in (traces or {}).items():
        (run / "results" / "traces" / f"trace_{t}.json").write_text(json.dumps(trace))
    (run / "run_summary.json").write_text(json.dumps({
        "variant": name, "resolved": len(passed), "total": len(rows), "hours": len(rows) * seconds / 3600}))
    return run


def base_run(root: Path, hours: float = 2.5) -> Path:
    """Stage 2 full-dev run: 120 x (2.5 / 33) + 0.2 = 9.3 h projected at s/task ratio 1."""
    run = root / "t0_think"
    run.mkdir()
    (run / "run_summary.json").write_text(json.dumps({"variant": "think", "resolved": 9, "total": 33, "hours": hours}))
    return run


def malformed_trace(n: int) -> dict:
    """n distinct tool calls, each answered with ADK's missing-parameter error."""
    return {"steps": [{"source": "agent", "tool_calls": [{"function_name": "edit_file", "arguments": {"i": i}}],
                       "observation": "Error: missing mandatory input parameters: path"} for i in range(n)]}


CTL_PASS = set(CORE) | {"fix0", "fix1"}          # 8 of 14, about the historical mean
WIN_PASS = CTL_PASS | {"fix2", "fix3"}           # LoRA solves 2 more per run


def pairs_of(root: Path, lora_kwargs: list[dict], ctl_kwargs: list[dict] | None = None) -> list[tuple[Path, Path]]:
    ctl_kwargs = ctl_kwargs or [{} for _ in lora_kwargs]
    return [(make_run(root, f"lora{k}", **{"passed": WIN_PASS, **lk}), make_run(root, f"ctl{k}", **{"passed": CTL_PASS, **ck}))
            for k, (lk, ck) in enumerate(zip(lora_kwargs, ctl_kwargs), 1)]


def failed(rep: dict) -> list[str]:
    return [x["check"] for x in rep["checks"] if x["ok"] is False]


def test_mcnemar_exact_values():
    assert pc.mcnemar_p(0, 0) == 1.0           # no discordant cells: no evidence at all
    assert pc.mcnemar_p(6, 0) == 1 / 64        # six coin flips all landing "LoRA"
    assert pc.mcnemar_p(7, 1) == 9 / 256
    assert pc.mcnemar_p(8, 2) == 56 / 1024     # just above 0.05
    assert pc.mcnemar_p(0, 5) == 1.0           # one-sided: losing is never significant
    # matches a direct binomial tail sum for every small (b, c)
    for n in range(12):
        for b in range(n + 1):
            tail = sum(math.comb(n, k) * 0.5 ** n for k in range(b, n + 1))
            assert pc.mcnemar_p(b, n - b) == pytest.approx(tail)


def test_identical_runs_give_p1_and_nogo(tmp_path):
    pairs = pairs_of(tmp_path, [{"passed": CTL_PASS}] * 4)
    rep = pc.compare(pairs, TASKS, FAIRNESS, base_run=base_run(tmp_path))
    assert rep["stage"] == "e2"
    assert (rep["b"], rep["c"], rep["p"], rep["mean_delta"]) == (0, 0, 1.0, 0)
    assert [r["delta"] for r in rep["pairs"]] == [0, 0, 0, 0]
    assert failed(rep) == ["mean delta", "McNemar p (one-sided)"]
    assert rep["verdict"] == "NO-GO"


def test_clear_win_gives_small_p_and_go(tmp_path):
    pairs = pairs_of(tmp_path, [{}] * 4)
    rep = pc.compare(pairs, TASKS, FAIRNESS, base_run=base_run(tmp_path))
    assert (rep["b"], rep["c"]) == (8, 0)
    assert rep["p"] == 1 / 256
    assert rep["mean_delta"] == 2.0
    assert [(r["lora"]["resolved"], r["ctl"]["resolved"]) for r in rep["pairs"]] == [(10, 8)] * 4
    assert failed(rep) == []
    assert rep["verdict"] == "GO"


def test_go_needs_runtime_projection(tmp_path):
    rep = pc.compare(pairs_of(tmp_path, [{}] * 4), TASKS, FAIRNESS)
    assert rep["verdict"].startswith("UNDECIDED")


def test_win_with_mixed_discordant_cells(tmp_path):
    """b and c are counted per task-run cell, so a LoRA that trades tasks gets c > 0."""
    trade = WIN_PASS - {"fix0"}  # +2 new, -1 old: delta +1 per run
    rep = pc.compare(pairs_of(tmp_path, [{"passed": trade}] * 4), TASKS, FAIRNESS, base_run=base_run(tmp_path))
    assert (rep["b"], rep["c"], rep["mean_delta"]) == (8, 4, 1.0)
    assert rep["p"] == pytest.approx(pc.mcnemar_p(8, 4))
    assert rep["verdict"] == "NO-GO"


@pytest.mark.parametrize("lora1, vetoed", [
    ({"traces": {"fix5": malformed_trace(3)}}, "malformed per LoRA run"),
    ({"errors": {"fix5": "litellm.BadRequestError: This model's maximum context length is 65536 tokens"}},
     "LoRA overflow"),
    ({"errors": {"fix5": "Sandbox execution error: Tool 'grep_rnI' not found.\nAvailable tools: run_command"}},
     "LoRA undeclared tool"),
    ({"errors": {"fix5": "RuntimeError: vLLM server died"}}, "LoRA crashes"),
    ({"skip": {"fix5"}}, "LoRA crashes"),                     # a task missing from the results
    ({"passed": WIN_PASS - {"core0", "core1"} | {"fix4", "fix5"}}, "LoRA core cells lost"),
    ({"seconds": 160.0}, "projected scorer h"),               # pooled ratio 1.15 -> 10.7 h
])
def test_safety_veto_overrides_a_clear_win(tmp_path, lora1, vetoed):
    pairs = pairs_of(tmp_path, [lora1, {}, {}, {}])
    rep = pc.compare(pairs, TASKS, FAIRNESS, base_run=base_run(tmp_path))
    assert rep["p"] <= 0.05 and rep["mean_delta"] >= 1.5   # the win itself still passes
    assert failed(rep) == [vetoed]
    assert rep["verdict"] == "NO-GO"


def test_benign_endings_are_not_crashes(tmp_path):
    errors = {"fix6": "Agent exceeded session timeout (5.0 min)", "fix7": "Agent exceeded turns budget (80 turns)"}
    rep = pc.compare(pairs_of(tmp_path, [{"errors": errors}] * 4), TASKS, FAIRNESS, base_run=base_run(tmp_path))
    assert rep["verdict"] == "GO"


def test_control_crash_makes_the_pair_invalid_not_go(tmp_path):
    pairs = pairs_of(tmp_path, [{}] * 4, [{"errors": {"fix7": "RuntimeError: boom"}}, {}, {}, {}])
    rep = pc.compare(pairs, TASKS, FAIRNESS, base_run=base_run(tmp_path))
    assert rep["verdict"].startswith("INVALID") and "[1]" in rep["verdict"]
    assert len(rep["warnings"]) == 1 and "pair 1 control" in rep["warnings"][0]


def test_truncated_control_cannot_fake_a_win(tmp_path):
    """The control stopped after the 6 core tasks. The adapter's passes on the missing tasks must not count."""
    skip = set(FIXABLE)
    lora = [make_run(tmp_path, f"lora{k}", WIN_PASS) for k in range(1, 5)]
    ctl = [make_run(tmp_path, f"ctl{k}", CTL_PASS - skip, skip=skip) for k in range(1, 5)]
    rep = pc.compare(list(zip(lora, ctl)), TASKS, FAIRNESS, base_run=base_run(tmp_path))
    assert rep["verdict"].startswith("INVALID")
    assert (rep["b"], rep["c"], rep["mean_delta"]) == (0, 0, 0)   # no usable cell favours the adapter
    assert rep["p"] == 1.0


def test_no_results_at_all_is_an_error(tmp_path):
    (tmp_path / "lora1").mkdir()
    (tmp_path / "ctl1").mkdir()
    with pytest.raises(ValueError):
        pc.compare([(tmp_path / "lora1", tmp_path / "ctl1")], [], FAIRNESS)
    with pytest.raises(SystemExit):   # the CLI refuses too, instead of answering "CONTINUE"
        pc.main(["--pair", str(tmp_path / "lora1"), str(tmp_path / "ctl1")])


def test_missing_fairness_file_is_an_error(tmp_path):
    pairs = pairs_of(tmp_path, [{}])
    with pytest.raises(SystemExit):
        pc.main(["--pair", str(pairs[0][0]), str(pairs[0][1]), "--fairness", str(tmp_path / "nope.json")])


def test_e1_early_stop_and_continue(tmp_path):
    lose = CTL_PASS - {"fix0"}  # delta -1 per pair, summed -2 over pairs 1-2
    rep = pc.compare(pairs_of(tmp_path / "a", [{"passed": lose}] * 2), TASKS, FAIRNESS)
    assert rep["stage"] == "e1"
    assert failed(rep) == ["summed delta pairs 1-2"]
    assert rep["verdict"] == "NO-GO (early stop)"

    rep = pc.compare(pairs_of(tmp_path / "b", [{}, {"seconds": 140.0}]), TASKS, FAIRNESS)
    assert failed(rep) == ["LoRA/control s/task"]           # pooled ratio (100 + 140) / 200 = 1.20 > 1.15
    rep = pc.compare(pairs_of(tmp_path / "c", [{}, {}]), TASKS, FAIRNESS)
    assert rep["verdict"] == "CONTINUE (run E2)"


def test_e2_needs_four_pairs(tmp_path):
    rep = pc.compare(pairs_of(tmp_path, [{}] * 2), TASKS, FAIRNESS, stage="e2", base_run=base_run(tmp_path))
    assert failed(rep) == ["pairs", "McNemar p (one-sided)"]  # b=4, c=0: p = 1/16 is not enough either
    assert rep["verdict"] == "NO-GO"


def test_cli_with_task_set_and_pair_flags(tmp_path, capsys):
    pairs = pairs_of(tmp_path / "runs", [{}] * 4)
    sets = tmp_path / "decision_sets.json"
    sets.write_text(json.dumps({"_about": "x", "dev_dec": TASKS[:-1]}))  # drop fix7: still 8 discordant cells
    fairness = tmp_path / "fairness.json"
    fairness.write_text(json.dumps(FAIRNESS))
    out = tmp_path / "rep.json"
    argv = [f"{pairs[0][0].name}:{pairs[0][1].name}", f"lora2={pairs[1][0]}:ctl2={pairs[1][1]}",
            "--pair", str(pairs[2][0]), str(pairs[2][1]), "--pair", str(pairs[3][0]), str(pairs[3][1]),
            "--runs-dir", str(tmp_path / "runs"), "--tasks", "dev_dec", "--sets-file", str(sets),
            "--fairness", str(fairness), "--base-run", str(base_run(tmp_path)), "--json", str(out)]
    rep = pc.main(argv)
    assert rep["verdict"] == "GO" and len(rep["tasks"]) == 13 and len(rep["pairs"]) == 4
    assert json.loads(out.read_text())["b"] == 8
    assert "VERDICT: GO" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        pc.main(argv[:1] + ["--runs-dir", str(tmp_path / "runs"), "--tasks", "nope", "--sets-file", str(sets)])
