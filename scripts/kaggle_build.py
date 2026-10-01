#!/usr/bin/env python3
"""Assemble a self-contained Kaggle notebook from kaggle/<name>/ and optionally push it.

Kaggle runs one script file, so the build:
  - pastes kaggle/common/kaggle_common.py in place of the `from kaggle_common import ...` line,
  - embeds the agent config files (`SUBMISSION_FILES = {}  # @embed submission`),
  - embeds the task ids to run     (`EMBEDDED_TASK_IDS = []  # @embed task_ids`),
and writes build/kaggle/<name>/ with the script and its kernel-metadata.json. Each pushed version
is therefore a complete record of exactly what ran.

Usage:
    python scripts/kaggle_build.py grader_check --push
    python scripts/kaggle_build.py agent_eval --submission submission --split dev --push
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".yaml", ".yml", ".md", ".txt", ".py", ".json"}


def embed_submission(sub: Path) -> dict[str, str]:
    files = {}
    for path in sorted(sub.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith("."):
            if path.suffix not in TEXT_SUFFIXES:
                raise SystemExit(f"{path}: only text files can be embedded (adapters need a Kaggle dataset)")
            files[path.relative_to(sub).as_posix()] = path.read_text(encoding="utf-8")
    if "agent.yaml" not in files:
        raise SystemExit(f"{sub} has no agent.yaml")
    return files


def build(name: str, submission: Path | None, split: str | None, splits_file: Path) -> Path:
    src_dir = ROOT / "kaggle" / name
    meta = json.loads((src_dir / "kernel-metadata.json").read_text())
    code = (src_dir / meta["code_file"]).read_text(encoding="utf-8")
    common = (ROOT / "kaggle" / "common" / "kaggle_common.py").read_text(encoding="utf-8")

    out_lines = []
    for line in code.splitlines():
        if line.startswith("from kaggle_common import"):
            out_lines.append("# ---- begin kaggle/common/kaggle_common.py (inlined by scripts/kaggle_build.py) ----")
            out_lines.extend(common.splitlines())
            out_lines.append("# ---- end kaggle/common/kaggle_common.py ----")
        elif line.rstrip().endswith("# @embed submission"):
            if submission is None:
                raise SystemExit(f"{name} embeds a submission: pass --submission")
            out_lines.append(f"SUBMISSION_FILES = {embed_submission(submission)!r}")
        elif line.rstrip().endswith("# @embed task_ids"):
            if split is None:
                raise SystemExit(f"{name} embeds task ids: pass --split")
            ids = json.loads(splits_file.read_text())[split]
            out_lines.append(f"EMBEDDED_TASK_IDS = {ids!r}  # split '{split}' from {splits_file.name}")
        else:
            out_lines.append(line)

    out_dir = ROOT / "build" / "kaggle" / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / meta["code_file"]).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    compile((out_dir / meta["code_file"]).read_text(), meta["code_file"], "exec")  # fail fast on syntax errors
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="folder under kaggle/, e.g. grader_check or agent_eval")
    parser.add_argument("--submission", type=Path, help="agent config folder to embed")
    parser.add_argument("--split", help="which list from the splits file to embed (dev, train, ...)")
    parser.add_argument("--splits-file", type=Path, default=ROOT / "eval" / "splits.json")
    parser.add_argument("--push", action="store_true", help="push to Kaggle after building")
    args = parser.parse_args()

    out_dir = build(args.name, args.submission, args.split, args.splits_file)
    print(f"built {out_dir}")
    if args.push:
        return subprocess.run(["kaggle", "kernels", "push", "-p", str(out_dir)]).returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
