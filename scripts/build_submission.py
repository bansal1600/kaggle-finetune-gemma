#!/usr/bin/env python3
"""Validate a submission directory and package it as submission.zip.

Usage:
    python scripts/build_submission.py                      # submission/ -> dist/submission.zip
    python scripts/build_submission.py --check-only         # validate, do not write the zip
    python scripts/build_submission.py --submission-dir experiments/v2 --out dist/v2.zip

Two layers of checks run before anything is written:

1. Our own checks, which mirror the competition rules (HARNESS_README.md): one root
   agent.yaml, allowed file types only, no symlinks or `..` paths, under 3 GiB, the single
   allowed model everywhere, adapters present, no broken `{placeholders}` in prompts, and a
   per-task budget that fits the 12 hour limit.
2. The organizers' own compiler (`adk_submission.compile_submission`) when it is installed.
   It is not on PyPI; see README.md -> "Local validation with the official compiler".

A daily submission is precious (1 per day), so the script refuses to build when any check fails.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALLOWED_MODEL = "gemma-4-31b-it-qat-w4a16-ct"
ROOT_CONFIG_NAMES = ("agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml")
ALLOWED_EXTENSIONS = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}
MAX_TOTAL_BYTES = 3 * 1024**3
MAX_TOKENS = 32_768
BUILTIN_TOOLS = {
    "run_command",
    "submit_patch",
    "get_status",
    "read_file",
    "edit_file",
    "write_file",
    "get_code_neighbors",
    "search_similar_code",
    "get_code_subgraph",
}
# Session-state keys the harness fills before the agent starts. `hints` is only set when the
# task has hints, so a bare `{hints}` would crash on tasks without them; `{hints?}` is safe.
STATE_KEYS = {"problem_description"}
OPTIONAL_STATE_KEYS = {"hints"}
EVAL_CONFIG_KEYS = {"timeout_seconds", "max_tool_calls", "max_time_minutes", "max_turns"}
TIME_LIMIT_MINUTES = 12 * 60
# Files we silently leave out of the zip instead of failing on them.
JUNK_NAMES = {".DS_Store", "Thumbs.db"}
JUNK_DIRS = {"__pycache__", ".ipynb_checkpoints"}


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


# --------------------------------------------------------------------------------------
# YAML loading with the competition's `!include` tag
# --------------------------------------------------------------------------------------


class IncludeError(ValueError):
    pass


def load_yaml(path: Path, root: Path, depth: int = 0) -> Any:
    """Load a YAML file, resolving `!include` relative to the including file."""
    if depth > 10:
        raise IncludeError(f"!include nesting deeper than 10 at {path}")

    class Loader(yaml.SafeLoader):
        pass

    def include(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
        rel = loader.construct_scalar(node)
        if rel.startswith("/") or ".." in Path(rel).parts:
            raise IncludeError(f"{path.relative_to(root)}: !include {rel} escapes the submission")
        target = (path.parent / rel).resolve()
        if not target.is_relative_to(root.resolve()):
            raise IncludeError(f"{path.relative_to(root)}: !include {rel} escapes the submission")
        if not target.is_file():
            raise IncludeError(f"{path.relative_to(root)}: !include {rel} not found")
        if target.suffix in (".yaml", ".yml"):
            return load_yaml(target, root, depth + 1)
        return target.read_text(encoding="utf-8")

    Loader.add_constructor("!include", include)
    with path.open(encoding="utf-8") as fh:
        return yaml.load(fh, Loader=Loader)  # noqa: S506 - SafeLoader subclass


# --------------------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------------------


def is_junk(rel: Path) -> bool:
    return rel.name in JUNK_NAMES or any(part in JUNK_DIRS for part in rel.parts)


def check_files(sub: Path, report: Report) -> list[Path]:
    """Check file types, symlinks and size. Returns the files that go into the zip."""
    files: list[Path] = []
    total = 0
    for path in sorted(sub.rglob("*")):
        rel = path.relative_to(sub)
        if is_junk(rel):
            continue
        if path.is_symlink():
            report.errors.append(f"{rel}: symlinks are not allowed")
            continue
        if path.is_dir():
            continue
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            report.errors.append(
                f"{rel}: extension '{path.suffix}' is not allowed "
                f"(allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))})"
            )
            continue
        total += path.stat().st_size
        files.append(path)
    if total >= MAX_TOTAL_BYTES:
        report.errors.append(f"submission is {total / 1024**3:.2f} GiB; the limit is 3 GiB")
    report.info.append(f"{len(files)} files, {total / 1024:.1f} KiB unpacked")
    return files


def find_root_config(sub: Path, report: Report) -> Path | None:
    roots = [sub / name for name in ROOT_CONFIG_NAMES if (sub / name).is_file()]
    if len(roots) != 1:
        found = ", ".join(r.name for r in roots) or "none"
        report.errors.append(f"need exactly one root config ({' / '.join(ROOT_CONFIG_NAMES)}); found {found}")
        return None
    return roots[0]


def template_placeholders(text: str) -> list[str]:
    """Placeholders ADK would try to fill from session state (mirrors ADK's own regex)."""
    names = []
    for match in re.finditer(r"{+[^{}]*}+", text):
        name = match.group().lstrip("{").rstrip("}").strip()
        bare = name.removesuffix("?")
        parts = bare.split(":")
        valid = (len(parts) == 1 and bare.isidentifier()) or (
            len(parts) == 2 and parts[0] in {"app", "user", "temp"} and parts[1].isidentifier()
        )
        if valid:
            names.append(name)
    return names


def check_instruction(where: str, text: str, report: Report) -> None:
    for name in template_placeholders(text):
        bare = name.removesuffix("?")
        if name.endswith("?"):
            continue
        if bare in STATE_KEYS:
            continue
        if bare in OPTIONAL_STATE_KEYS:
            report.errors.append(
                f"{where}: '{{{bare}}}' crashes on tasks where it is not set; write '{{{bare}?}}'"
            )
        else:
            report.errors.append(
                f"{where}: '{{{name}}}' would be read from session state and raise KeyError. "
                "Avoid curly braces around single words in prompts."
            )
    check_heredocs(where, text, report)


HEREDOC_START = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?\s*$")


def check_heredocs(where: str, text: str, report: Report) -> None:
    """Heredoc examples the model will copy must work verbatim: bash only ends a heredoc at a line
    that is exactly the delimiter, so an indented closing line (or an indented Python body) breaks
    the command. Three drafted prompts had this bug (docs/research-2026-10-02.md)."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        start = HEREDOC_START.search(lines[i])
        if not start:
            i += 1
            continue
        delimiter, opener = start.group(1), i
        if lines[i] != lines[i].lstrip():
            report.errors.append(f"{where}:{i + 1}: heredoc example is indented; start it at column 0")
        i += 1
        while i < len(lines) and lines[i].strip() != delimiter:
            i += 1
        if i == len(lines):
            report.errors.append(f"{where}:{opener + 1}: heredoc '{delimiter}' is never closed")
        elif lines[i] != delimiter:
            report.errors.append(f"{where}:{i + 1}: heredoc closing '{delimiter}' must be alone at column 0")
        else:
            body = lines[opener + 1:i]
            if "python" in lines[opener] and body and all(b.startswith((" ", "\t")) for b in body if b):
                report.errors.append(f"{where}:{opener + 2}: Python heredoc body is indented (IndentationError)")
        i += 1


def check_generation_config(where: str, cfg: Any, report: Report) -> None:
    if cfg is None:
        report.warnings.append(f"{where}: no generate_content_config; defaults are 16,384 output tokens, thinking on")
        return
    if not isinstance(cfg, dict):
        report.errors.append(f"{where}: generate_content_config must be a mapping")
        return
    forbidden = {"tools", "system_instruction", "http_options", "safety_settings", "response_schema"} & cfg.keys()
    if forbidden:
        report.errors.append(f"{where}: generate_content_config may not set {sorted(forbidden)}")
    max_out = cfg.get("max_output_tokens")
    if max_out is not None and not 1 <= int(max_out) <= MAX_TOKENS:
        report.errors.append(f"{where}: max_output_tokens must be 1..{MAX_TOKENS}")
    thinking = cfg.get("thinking_config") or {}
    budget = thinking.get("thinking_budget")
    if budget is not None and not 0 <= int(budget) <= MAX_TOKENS:
        report.errors.append(f"{where}: thinking_budget must be 0..{MAX_TOKENS}")
    if "thinking_level" in thinking:
        report.warnings.append(
            f"{where}: thinking_level is mapped to reasoning_effort over vLLM; "
            "the harness README recommends include_thoughts + thinking_budget instead"
        )


def check_agent(path: Path, sub: Path, report: Report, seen: set[Path]) -> None:
    """Check one agent YAML and recurse into its sub-agents and agent tools."""
    if path in seen:
        return
    seen.add(path)
    rel = path.relative_to(sub)
    try:
        cfg = load_yaml(path, sub)
    except (IncludeError, yaml.YAMLError, OSError) as exc:
        report.errors.append(f"{rel}: {exc}")
        return
    if not isinstance(cfg, dict):
        report.errors.append(f"{rel}: top level must be a mapping")
        return

    agent_class = cfg.get("agent_class", "LlmAgent")
    name = cfg.get("name")
    if not name:
        report.errors.append(f"{rel}: missing 'name'")
    where = f"{rel} ({name or '?'})"

    if agent_class == "LlmAgent":
        model = str(cfg.get("model", "")).split("/")[-1]
        if model != ALLOWED_MODEL:
            report.errors.append(f"{where}: model must be '{ALLOWED_MODEL}', got '{cfg.get('model')}'")
        instruction = cfg.get("instruction", "")
        if not isinstance(instruction, str):
            report.errors.append(f"{where}: instruction must be text")
        else:
            check_instruction(where, instruction, report)
        check_generation_config(where, cfg.get("generate_content_config"), report)

        adapter = cfg.get("adapter")
        if adapter:
            adapter_dir = sub / "adapters" / adapter
            for needed in ("adapter_config.json", "adapter_model.safetensors"):
                if not (adapter_dir / needed).is_file():
                    report.errors.append(f"{where}: adapter '{adapter}' is missing adapters/{adapter}/{needed}")

        for tool in cfg.get("tools") or []:
            if isinstance(tool, str):
                if tool not in BUILTIN_TOOLS:
                    report.errors.append(f"{where}: unknown tool '{tool}'")
            elif isinstance(tool, dict) and "agent_tool" in tool:
                child = tool["agent_tool"].get("config_path", "")
                check_child(path, child, sub, where, report, seen)
            else:
                report.errors.append(f"{where}: unsupported tool entry {tool!r}")

        for skill in cfg.get("skills") or []:
            skill_md = sub / skill / "SKILL.md"
            if not skill_md.is_file():
                report.errors.append(f"{where}: skill '{skill}' has no SKILL.md")

    for child_ref in cfg.get("sub_agents") or []:
        child = child_ref.get("config_path", "") if isinstance(child_ref, dict) else ""
        check_child(path, child, sub, where, report, seen)


def check_child(parent: Path, child: str, sub: Path, where: str, report: Report, seen: set[Path]) -> None:
    if not child or child.startswith("/") or ".." in Path(child).parts:
        report.errors.append(f"{where}: bad config_path '{child}'")
        return
    # Like adk_submission: try the referencing file's directory first, then the submission root.
    target = parent.parent / child
    if parent.parent == sub or not target.exists():
        target = sub / child
    if not target.is_file():
        report.errors.append(f"{where}: config_path '{child}' not found")
        return
    check_agent(target, sub, report, seen)


def check_eval_config(sub: Path, report: Report, num_tasks: int, overhead_minutes: float) -> None:
    path = sub / "eval_config.yaml"
    if not path.is_file():
        report.warnings.append(
            "no eval_config.yaml: the scorer then allows 60 min per task, which blows the 12 h limit "
            "if tasks run long"
        )
        return
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        report.errors.append(f"eval_config.yaml: {exc}")
        return
    section = raw.get("evaluation", raw)
    unknown = set(section) - EVAL_CONFIG_KEYS
    if unknown:
        report.warnings.append(f"eval_config.yaml: keys {sorted(unknown)} are ignored by the scorer")
    minutes = float(section.get("max_time_minutes", 60.0))
    worst = num_tasks * (minutes + overhead_minutes)
    report.info.append(
        f"budget: {minutes} min/task x {num_tasks} tasks (+{overhead_minutes} min overhead each) "
        f"= {worst / 60:.1f} h worst case (limit 12 h)"
    )
    if worst > TIME_LIMIT_MINUTES:
        report.errors.append(
            f"worst-case runtime {worst / 60:.1f} h exceeds the 12 h limit; lower max_time_minutes"
        )
    elif worst > TIME_LIMIT_MINUTES - 60:
        report.warnings.append(f"worst-case runtime {worst / 60:.1f} h leaves under 1 h of slack")
    if int(section.get("timeout_seconds", 300)) < 120:
        report.warnings.append("timeout_seconds < 120 may cut off slow test runs")


def check_submission(sub: Path, num_tasks: int = 120, overhead_minutes: float = 1.0) -> tuple[Report, list[Path]]:
    report = Report()
    if not sub.is_dir():
        report.errors.append(f"{sub} is not a directory")
        return report, []
    files = check_files(sub, report)
    root = find_root_config(sub, report)
    if root is not None:
        check_agent(root, sub, report, seen=set())
    check_eval_config(sub, report, num_tasks, overhead_minutes)
    return report, files


# --------------------------------------------------------------------------------------
# Optional: the organizers' compiler
# --------------------------------------------------------------------------------------


def _dummy_tools() -> dict[str, Any]:
    """Stand-ins with the real signatures, so ADK can build tool declarations."""

    def run_command(command: str) -> str:
        """Run a shell command in /workspace."""
        return ""

    def submit_patch() -> str:
        """Submit the current git diff as the answer."""
        return ""

    def get_status() -> str:
        """Report budget use."""
        return ""

    def read_file(filepath: str, start_line: int | None = None, end_line: int | None = None) -> str:
        """Read a file."""
        return ""

    def edit_file(filepath: str, old_string: str, new_string: str, allow_multiple: bool = False) -> str:
        """Replace text in a file."""
        return ""

    def write_file(filepath: str, content: str) -> str:
        """Write a file."""
        return ""

    def get_code_neighbors(node: str, edge_type: str | None = None, max_neighbors: int = 50) -> str:
        """Graph neighbours of a symbol."""
        return ""

    def search_similar_code(query: str, k: int = 10) -> str:
        """Embedding search by symbol name."""
        return ""

    def get_code_subgraph(nodes: list[str]) -> str:
        """Induced subgraph of symbols."""
        return ""

    return {fn.__name__: fn for fn in (
        run_command, submit_patch, get_status, read_file, edit_file, write_file,
        get_code_neighbors, search_similar_code, get_code_subgraph,
    )}


def official_compile(sub: Path, report: Report) -> None:
    try:
        from adk_submission import ModelRegistry, compile_submission, discover_adapters
        from adk_submission.limits import GenerationConstraints, NumericRange, SubmissionLimits
    except ImportError:
        report.warnings.append(
            "official compiler not installed; skipped (see README -> 'Local validation with the official compiler')"
        )
        return

    # Same values as swegemma.config.build_submission_limits() in the scoring harness.
    limits = SubmissionLimits(
        max_total_size_bytes=MAX_TOTAL_BYTES,
        max_yaml_size_bytes=50 * 1024 * 1024,
        max_skill_size_bytes=50 * 1024 * 1024,
        max_file_count=10_000,
        max_yaml_files=1_000,
        max_instruction_chars=1_000_000,
        max_total_instruction_chars=10_000_000,
        max_agents=500,
        max_sub_agent_depth=50,
        max_skills=1_000,
        max_loop_iterations=500,
        allowed_file_extensions=frozenset(ALLOWED_EXTENSIONS),
        adapter_extensions=frozenset({".safetensors"}),
    )
    constraints = GenerationConstraints(
        allowed_fields=None,
        max_output_tokens=NumericRange(1, MAX_TOKENS),
        thinking_budget=NumericRange(0, MAX_TOKENS),
        defaults={"max_output_tokens": 16384, "thinking_config": {"thinking_budget": 4096}},
    )
    models = ModelRegistry()
    models.register(ALLOWED_MODEL, ALLOWED_MODEL)
    try:
        adapters = discover_adapters(str(sub), adapter_extensions={".safetensors"})
        agent = compile_submission(
            submission_dir=str(sub),
            tool_registry=_dummy_tools(),
            model_registry=models,
            limits=limits,
            generation_constraints=constraints,
            adapter_manifest=adapters,
            adapter_resolver_fn=lambda base, info: base,
        )
    except Exception as exc:  # noqa: BLE001 - report whatever the compiler rejects
        report.errors.append(f"official compiler rejected the submission: {type(exc).__name__}: {exc}")
        return

    def describe(node: Any, depth: int = 0) -> list[str]:
        tools = [getattr(t, "name", None) or getattr(t, "__name__", type(t).__name__)
                 for t in getattr(node, "tools", []) or []]
        pad = "  " * depth
        lines = [f"{pad}- {node.name} ({type(node).__name__})", f"{pad}    tools: {', '.join(tools)}"]
        instruction = getattr(node, "instruction", None)
        if isinstance(instruction, str):
            lines.append(f"{pad}    instruction: {len(instruction):,} chars")
        gen = getattr(node, "generate_content_config", None)
        if gen is not None:
            shown = gen.model_dump(exclude_none=True) if hasattr(gen, "model_dump") else gen
            lines.append(f"{pad}    generate_content_config: {shown}")
        for child in getattr(node, "sub_agents", []) or []:
            lines += describe(child, depth + 1)
        return lines

    report.info.append("official compiler: OK\n" + "\n".join(describe(agent)))


# --------------------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------------------


def write_zip(sub: Path, files: list[Path], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(files):
            # Fixed timestamp so the same input always produces the same zip bytes.
            info = zipfile.ZipInfo(path.relative_to(sub).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission-dir", type=Path, default=Path("submission"))
    parser.add_argument("--out", type=Path, default=Path("dist/submission.zip"))
    parser.add_argument("--check-only", action="store_true", help="validate without writing the zip")
    parser.add_argument("--num-tasks", type=int, default=120, help="hidden test set size used for the time check")
    parser.add_argument("--overhead-minutes", type=float, default=1.0, help="sandbox setup time per task")
    parser.add_argument("--skip-official", action="store_true", help="do not run the official compiler")
    args = parser.parse_args(argv)

    sub = args.submission_dir
    report, files = check_submission(sub, args.num_tasks, args.overhead_minutes)
    if report.ok and not args.skip_official:
        official_compile(sub, report)

    for line in report.info:
        print(f"info:    {line}")
    for line in report.warnings:
        print(f"warning: {line}")
    for line in report.errors:
        print(f"ERROR:   {line}")

    if not report.ok:
        print(f"\n{len(report.errors)} error(s); not building.")
        return 1
    if args.check_only:
        print("\nAll checks passed.")
        return 0
    write_zip(sub, files, args.out)
    print(f"\nWrote {args.out} ({args.out.stat().st_size / 1024:.1f} KiB). Upload it with:")
    print(f'  kaggle competitions submit -c gemma-4-developer-agent -f {args.out} -m "<what changed>"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
