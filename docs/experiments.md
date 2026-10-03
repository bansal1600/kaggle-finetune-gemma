# Experiment log

One row per Kaggle submission. Change **one thing** at a time (see concepts.md →
"One change at a time"), and note the git commit so every score maps to exact files.

Rough scale: the public leaderboard has ~60 tasks, so 1 task ≈ 0.017 and differences of ≤ 0.03
are mostly noise.

| # | Date (UTC) | Commit | Change vs previous | Public score | Notes |
|---|---|---|---|---|---|
| v1 | 2026-10-01 20:50 | 41ab46b | First submission: single agent, our prompt, thinking off, temp 0.2, 4.5 min / 40 calls / 80 turns | **0.12** | Scored 2026-10-02 (~8 h after submitting). Dev set: 6/33 (18.2%). Kaggle ref 56758822; zip sha256 369b1c64…. Validated with adk-submission 0.2.12 + google-adk 1.36.1. Expected 0.06–0.12 from comparable public configs |
| v3d_full | 2026-10-02 18:28 | e0a5646 | On v2a sampling: script edits via run_command, issue text pinned with {problem_description?}, list-source-files / reproduce / verify rules, get_code_subgraph declared, 50 tool calls (see docs/research-2026-10-02.md) | _pending_ | Kaggle ref 56780076; zip sha256 1de2fbdd…. Dev 8/33 (v1 6/33). Garbled calls 68 vs v1 143 |
| v3b_noedit | 2026-10-03 01:44 | e0a5646 | On v2a sampling: edit_file and write_file removed, every edit through a Python script in run_command; tool-hygiene prompt; get_code_subgraph declared; 40 tool calls | _pending_ | Kaggle ref 56786866; zip sha256 3c017a16…. Dev 11, 10, 8 over 3 runs (mean 9.7), core 6/6 every run, 0–1 garbled calls |

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

| v3 run A | 2026-10-02 16:47–18:26 | `experiments/v3d_full`: all research fixes on v2a (see docs/research-2026-10-02.md) | **8/33 = 24.2%** (1.52 h) | budget_out 10, wrong_fix 6, no_patch 5, loop 2, timeout 2. Submitted |
| v3 run B | 2026-10-02 16:47–21:30 | notebook `gemma-agent-eval-b` | | |
| ↳ v3a_hyg | | edit/tool mechanics only (script edits preferred, edit_file kept) | 6/33 (1.21 h) | garbled calls 12, but lost 1 core task |
| ↳ v3c_loc | | localize / reproduce / verify rules only | 8/33 (1.27 h) | garbled calls 166 (edit_file advice unchanged) |
| ↳ v3b_noedit | | v3a with `edit_file` and `write_file` removed | **11/33** (1.03 h) | **garbled calls 1**, repeats 3.3%, 0 stray files, core 6/6, fixable 5, 0 undeclared-tool calls |
| ↳ v2a_rerun | | v2a again (noise control) | **11/33** (1.13 h) | same config scored 8/33 on 10-02 morning |

v3d_full vs v2a vs v1 (`scripts/compare_runs.py`): same 8/33 as v2a, but it solves 3 of the 12
fixable tasks (v2a 2, v1 0): fastapi_15589, requests_7427 and rich_4077 (requests_7427 had never
passed before). It lost one core task (rich_3882). Mechanisms moved as intended: garbled calls
68 (v2a 91, v1 143), repeats 7.0% (11.6%, 33.9%), 32 script edits (0 before), with 2 failed
asserts. Costs: mean 165 s per task (v2a 112 s), so a 120-task run projects to ~7 h of the 12 h;
budget_out rose to 10, as the extra verify steps use calls. The union of passes over all six runs
is 10/33.

Run B findings:
- **Noise is large.** The identical v2a config scored 8/33 and then 11/33. Part of the gap is the
  editable-install bench fix (requests_7427 became passable for every post-fix run); most is
  temperature-1.0 luck. Single-run gaps of ±3 tasks are noise, so decide on repeated runs and on
  mechanism metrics.
- **Removing edit_file is the one change that clearly works mechanically** (v3b): garbled calls
  1 vs 125 for the v2a rerun, repeats 3.3%, 0 stray files, fastest mean (112 s). The model never
  tried to call the removed tools. Its score ties the best.
- Mechanics with edit_file still available (v3a) barely moved the model: it kept using edit_file
  for most edits.
- Union of passes across all 9 runs: 13/33.
- Next: batch C runs v3e (v3d_full without edit_file/write_file) and v3b again, twice each.

| Run | Date (UTC) | Config | Dev score | Notes |
|---|---|---|---|---|
| batch C | 2026-10-02 21:51 – 10-03 00:30 | two notebooks in parallel | | |
| ↳ v3e_full_noedit | | v3d_full without edit_file/write_file | 9/33 (1.25 h) | core 6/6, garbled 1 |
| ↳ v3b_r2 | | v3b_noedit again | 10/33 (1.07 h) | core 6/6, garbled 0 |
| ↳ v3e_r2 | | v3e again | 8/33 (1.23 h) | core 6/6, garbled 0 |
| ↳ v3b_r3 | | v3b_noedit again | 8/33 (1.09 h) | core 6/6, garbled 0 |

Pooled over repeated runs (all after the editable-install bench fix):

| Config | Runs | Scores | Mean | Core kept | Fixable solved | Garbled calls | Mean s/task |
|---|---|---|---|---|---|---|---|
| **v3b_noedit** | 3 | 11, 10, 8 | **9.7** | 6/6 in every run | 5, 4, 2 | 1, 0, 0 | ~116 |
| v3e_full_noedit | 2 | 9, 8 | 8.5 | 6/6 in every run | 3, 2 | 1, 0 | ~135 |
| v2a | 1 | 11 | 11 | 6/6 | 5 | 125 | 123 |
| v3d_full | 1 | 8 | 8 | 5/6 | 3 | 68 | 165 |

Batch C findings:
- **v3b is the best-supported config.** It never lost a core task in 3 runs, has no garbled calls,
  no stray files, and is the fastest. Its run-to-run range (8–11) shows the noise directly.
- **The extra rules did not help on top of removing edit_file.** v3e (v3b plus the localize /
  verify rules, issue pin and 50 calls) averaged 8.5 against v3b's 9.7, with a longer mean time.
  The difference is within noise, but nothing suggests the rules add value. Shorter prompts did at
  least as well, which matches other teams' reports.
- v2a's single post-fix run (11) equals v3b's best. On the hidden set, v2a's 125 garbled calls
  per 33 tasks are a risk v3b does not carry.
- Union of passes across all post-fix runs: 14/33.

From v3 on, the dev bench installs the task repo in editable mode like the scorer, which matters
for requests (src/ layout). Compare with `python scripts/compare_runs.py name=dir ...`, which
also shows core kept / fixable solved (`eval/fairness.json`) and the mechanism metrics.

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
