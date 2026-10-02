# Experiment log

One row per Kaggle submission. Change **one thing** at a time (see concepts.md →
"One change at a time"), and note the git commit so every score maps to exact files.

Rough scale: the public leaderboard has ~60 tasks, so 1 task ≈ 0.017 and differences of ≤ 0.03
are mostly noise.

| # | Date (UTC) | Commit | Change vs previous | Public score | Notes |
|---|---|---|---|---|---|
| v1 | 2026-10-01 20:50 | 41ab46b | First submission: single agent, our prompt, thinking off, temp 0.2, 4.5 min / 40 calls / 80 turns | **0.12** | Scored 2026-10-02 (~8 h after submitting). Dev set: 6/33 (18.2%). Kaggle ref 56758822; zip sha256 369b1c64…. Validated with adk-submission 0.2.12 + google-adk 1.36.1. Expected 0.06–0.12 from comparable public configs |

## Local dev runs (Step 2c)

Our own bench: 33 dev tasks (`eval/splits.json`), scorer's model and budgets on Kaggle 4×L4.
Analysis with `python scripts/analyze_run.py eval/runs/<run>`. Expect these to read higher than
the leaderboard (see concepts.md → "Data contamination"); use them to compare our versions.

| Run | Date (UTC) | Config | Dev score | Failure breakdown (27 fails for v1) |
|---|---|---|---|---|
| v1_dev | 2026-10-02 02:20 | v1 (git 41ab46b `submission/`) | **6/33 = 18.2%** (1.1 h) | loop 15, wrong_fix 6, no_patch 4, touched_tests 1, budget_out 1 |

| v2 batch | 2026-10-02 | 4 configs in one notebook (agent_eval v2), each in `experiments/`: | _running_ | |
| ↳ v2c | | anti-loop prompt **and** Gemma sampling (temp 1.0, top_p 0.95, top_k 64) | | |
| ↳ v2a | | Gemma sampling only (v1 prompt) | | |
| ↳ v2b | | anti-loop prompt only (v1 sampling, temp 0.2) | | |
| ↳ v2d | | anti-loop prompt + temp 0.7 (top_p 0.95, top_k 64) | | |

Reading the v2 batch: v2a and v2b each change one thing vs v1, v2c both, v2d checks whether
temp 1.0 is too random. With 33 tasks, ±2 tasks is within noise (concepts.md → "Noise").

v1 findings:
- **Loops are the #1 failure (15/27).** Same tool call repeated 5–44 times in a row: re-running an
  identical `grep`/`python -c` (10 tasks), or re-sending an `edit_file` call that failed, often
  because its arguments were malformed (`fastapi_14986`, `rich_3944`). 3 of them never called
  `submit_patch`.
- **no_patch (4):** explored for 30–40 calls, never edited, then said the issue was too vague.
- **wrong_fix (6):** a real attempt that fails the hidden tests, mostly with few calls (4–13): it
  submitted without checking enough.
- Patches often include stray `repro.py` files written in the repo root.

## Ideas queue (one per submission)

- [ ] **Anti-loop (from v1 dev analysis):** Gemma's recommended sampling (temp 1.0, top_p 0.95, top_k 64) + "never repeat a call" prompt rule
- [ ] v2: add a read-only analyzer AgentTool (`skip_summarization: false`) to save context in the main agent
- [ ] Prompt variants: stronger "reproduce first" vs "edit first"
- [ ] Thinking on with a small budget (e.g. 1024), keeping everything else fixed
- [ ] Graph tools on vs off
- [ ] LoRA adapter trained on our own successful trajectories (Steps 4–5)
