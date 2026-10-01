"""Tests for scripts/build_submission.py. Run with: python -m pytest -q"""

import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_submission as bs  # noqa: E402

REPO_SUBMISSION = Path(__file__).resolve().parents[1] / "submission"


def make_submission(tmp_path: Path, agent: str, prompt: str = "Fix the issue.", eval_cfg: str | None = None) -> Path:
    sub = tmp_path / "submission"
    (sub / "prompts").mkdir(parents=True)
    (sub / "prompts" / "system.md").write_text(prompt)
    (sub / "agent.yaml").write_text(agent)
    if eval_cfg is not None:
        (sub / "eval_config.yaml").write_text(eval_cfg)
    return sub


GOOD_AGENT = """\
name: coder
model: gemma-4-31b-it-qat-w4a16-ct
instruction: !include prompts/system.md
tools: [run_command, submit_patch]
"""
GOOD_EVAL = "evaluation:\n  max_time_minutes: 4.5\n"


def errors_for(sub: Path) -> list[str]:
    report, _ = bs.check_submission(sub)
    return report.errors


def test_repo_submission_is_valid():
    report, files = bs.check_submission(REPO_SUBMISSION)
    assert report.errors == []
    assert any(f.name == "agent.yaml" for f in files)


def test_minimal_submission_passes(tmp_path):
    assert errors_for(make_submission(tmp_path, GOOD_AGENT, eval_cfg=GOOD_EVAL)) == []


def test_wrong_model_is_rejected(tmp_path):
    sub = make_submission(tmp_path, GOOD_AGENT.replace("gemma-4-31b-it-qat-w4a16-ct", "gemma-4-12b-it"), eval_cfg=GOOD_EVAL)
    assert any("model must be" in e for e in errors_for(sub))


def test_provider_prefix_on_model_is_accepted(tmp_path):
    sub = make_submission(tmp_path, GOOD_AGENT.replace("model: ", "model: openai/"), eval_cfg=GOOD_EVAL)
    assert errors_for(sub) == []


def test_unknown_tool_is_rejected(tmp_path):
    sub = make_submission(tmp_path, GOOD_AGENT.replace("submit_patch", "web_search"), eval_cfg=GOOD_EVAL)
    assert any("unknown tool 'web_search'" in e for e in errors_for(sub))


def test_disallowed_extension_is_rejected(tmp_path):
    sub = make_submission(tmp_path, GOOD_AGENT, eval_cfg=GOOD_EVAL)
    (sub / "notes.pdf").write_bytes(b"%PDF")
    assert any("notes.pdf" in e for e in errors_for(sub))


def test_two_root_configs_are_rejected(tmp_path):
    sub = make_submission(tmp_path, GOOD_AGENT, eval_cfg=GOOD_EVAL)
    (sub / "root_agent.yaml").write_text(GOOD_AGENT)
    assert any("exactly one root config" in e for e in errors_for(sub))


def test_include_cannot_escape_submission(tmp_path):
    sub = make_submission(tmp_path, GOOD_AGENT.replace("prompts/system.md", "../secret.md"), eval_cfg=GOOD_EVAL)
    (tmp_path / "secret.md").write_text("x")
    assert any("escapes the submission" in e for e in errors_for(sub))


@pytest.mark.parametrize(
    ("prompt", "ok"),
    [
        ("Issue: {problem_description}", True),
        ("Hints: {hints?}", True),
        ("Hints: {hints}", False),  # missing on tasks without hints -> KeyError at runtime
        ("Use {repo_root} here", False),  # not a session-state key
        ('JSON like {"a": 1} is left alone', True),
    ],
)
def test_prompt_placeholders(tmp_path, prompt, ok):
    sub = make_submission(tmp_path, GOOD_AGENT, prompt=prompt, eval_cfg=GOOD_EVAL)
    assert (errors_for(sub) == []) is ok


def test_budget_over_12_hours_is_rejected(tmp_path):
    sub = make_submission(tmp_path, GOOD_AGENT, eval_cfg="evaluation:\n  max_time_minutes: 10\n")
    assert any("exceeds the 12 h limit" in e for e in errors_for(sub))


def test_missing_adapter_is_rejected(tmp_path):
    sub = make_submission(tmp_path, GOOD_AGENT + "adapter: main_lora\n", eval_cfg=GOOD_EVAL)
    assert any("adapter 'main_lora'" in e for e in errors_for(sub))


def test_agent_tool_is_checked_recursively(tmp_path):
    agent = GOOD_AGENT + "  - agent_tool:\n      config_path: sub_agents/helper.yaml\n"
    agent = agent.replace("tools: [run_command, submit_patch]\n", "tools:\n  - run_command\n")
    sub = make_submission(tmp_path, agent, eval_cfg=GOOD_EVAL)
    (sub / "sub_agents").mkdir()
    (sub / "sub_agents" / "helper.yaml").write_text("name: helper\nmodel: gemma-4-12b-it\ninstruction: hi\n")
    assert any("helper" in e and "model must be" in e for e in errors_for(sub))


def test_zip_has_agent_yaml_at_root_and_skips_junk(tmp_path):
    sub = make_submission(tmp_path, GOOD_AGENT, eval_cfg=GOOD_EVAL)
    (sub / ".DS_Store").write_bytes(b"junk")
    report, files = bs.check_submission(sub)
    assert report.ok
    out = tmp_path / "submission.zip"
    bs.write_zip(sub, files, out)
    names = zipfile.ZipFile(out).namelist()
    assert "agent.yaml" in names
    assert ".DS_Store" not in names


def test_zip_is_deterministic(tmp_path):
    sub = make_submission(tmp_path, GOOD_AGENT, eval_cfg=GOOD_EVAL)
    _, files = bs.check_submission(sub)
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    bs.write_zip(sub, files, a)
    bs.write_zip(sub, files, b)
    assert a.read_bytes() == b.read_bytes()
