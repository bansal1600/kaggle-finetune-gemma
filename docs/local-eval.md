# Local evaluation (Step 2)

Our own test bench: run the agent on public tasks whose answers we know, read exactly what it
did, and check changes before spending a daily submission. Concepts behind it are in
concepts.md, sections 6 ("Running experiments well") and the Step 2 entries.

**Expect local scores to be higher than the leaderboard and not to predict it closely.**
Other teams saw 0.18–0.24 locally vs 0.05–0.12 on the leaderboard (see "Data contamination").
Use the bench mainly to understand failures and catch breakage.

## Pieces

| Step | What | Where | GPU |
|---|---|---|---|
| 2b | Grader check: the official verifier runs every task with the real fix (must pass) and with no fix (must fail) | `kaggle/grader_check/` → `eval/grader_check.csv` | No |
| 2a | Split healthy tasks into dev (newest, ~33) and train | `scripts/make_split.py` → `eval/splits.json` | No |
| 2c | Run an agent config on the dev set with the scorer's model, hardware and budgets | `kaggle/agent_eval/` (next) | 4×L4 |
| 2d | Failure analysis of the run | `eval/runs/<run>/` | No |

## How a Kaggle run works

Each folder under `kaggle/` is one Kaggle notebook: a script plus `kernel-metadata.json`
(attached data, machine type, the organizers' pinned Docker image). From this repo:

```bash
kaggle kernels push -p kaggle/grader_check                          # start a run on Kaggle
kaggle kernels status guaravbansal/gemma-agent-grader-check         # QUEUED / RUNNING / COMPLETE / ERROR
kaggle kernels output guaravbansal/gemma-agent-grader-check -p eval/raw/grader_check   # download outputs
```

You can also watch it under *Your Work → Code* on Kaggle.

## Matching the scorer's environment

In notebook mode the official harness uses a "subprocess" sandbox: no Docker, one private Python
environment per task. Out of the box, that sandbox skips the scorer's dependency setup and lets
tasks import whatever the notebook happens to have installed. `use_scorer_environment()` in
`kaggle/grader_check/grader_check.py` fixes that:

1. It installs the scorer's package set: the newest version of each package in the competition's
   `wheels/` folder (the same rule as the scorer), built for this notebook's Python.
2. Each task's environment sees only that set, not the notebook's own packages.
3. The editable install and `sandbox/setup.py` step that the scorer runs, run here too.

Remaining differences from the scorer: Python 3.12 (scorer: 3.13) and the exact pytest version.

## Local testing before pushing

The scripts read their paths from environment variables, so they can be tried here on a few tasks
first (needs Python 3.12 with the harness wheels; see the README for the wheels):

```bash
DATA_DIR=data/comp WHEELHOUSE=/nonexistent OUT_DIR=/tmp/gc TASK_IDS=rich_3718 \
  python kaggle/grader_check/grader_check.py
```

`data/comp/` holds `tasks.jsonl`, `wheels/`, `sandbox/setup.py` and the needed `snapshots/*.tgz`
(downloaded with `kaggle competitions download -f <file>`). It is git-ignored.
