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

| v2 batch | 2026-10-02 11:55–16:20 | 4 configs in one notebook (agent_eval v2), each in `experiments/` | | ~4.5 h incl. ~9 h queue before start |
| ↳ v2c_both | | anti-loop prompt **and** Gemma sampling (temp 1.0, top_p 0.95, top_k 64) | 7/33 = 21.2% (1.26 h) | loop 6, wrong_fix 16, timeout 2, budget_out 2 |
| ↳ v2a_sampling | | Gemma sampling only (v1 prompt) | **8/33 = 24.2%** (1.03 h) | loop 3, wrong_fix 14, no_patch 4, budget_out 4 |
| ↳ v2b_prompt | | anti-loop prompt only (v1 sampling, temp 0.2) | 6/33 = 18.2% (1.09 h) | loop 14, wrong_fix 9, no_patch 2, budget_out 2 |
| ↳ v2d_both_t07 | | anti-loop prompt + temp 0.7 (top_p 0.95, top_k 64) | 7/33 = 21.2% (1.05 h) | loop 8, wrong_fix 12, timeout 2, no_patch 2, budget_out 2 |

v2 findings:
- **Sampling fixes the loops, the prompt does not.** Loop tasks: v1 15 → v2a 3, v2c 6, v2d 8, but v2b
  (prompt rules at temp 0.2) still 14. Temperature is what breaks the repetition.
- **But the score barely moved** (6 → 6–8, all within noise). The tasks that stopped looping now fail
  as wrong fixes. Loops were a symptom; the real limit is fix quality.
- **Pass matrix over 5 runs:** 5 tasks pass every time (fastapi_14786, fastapi_14794, requests_7315,
  rich_3894, rich_3905), 4 pass sometimes (fastapi_14873, fastapi_15589, rich_3882, rich_4077), and
  **24 never pass**. The union is 9/33: picking the best config alone cannot go much higher.
- Time is not the constraint: the mean task takes ~2 min of the 4.5 allowed; the 40-call budget runs
  out first.

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
