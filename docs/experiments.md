# Experiment log

One row per Kaggle submission. Change **one thing** at a time (see concepts.md →
"One change at a time"), and note the git commit so every score maps to exact files.

Rough scale: the public leaderboard has ~60 tasks, so 1 task ≈ 0.017 and differences of ≤ 0.03
are mostly noise.

| # | Date (UTC) | Commit | Change vs previous | Public score | Notes |
|---|---|---|---|---|---|
| v1 | 2026-10-01 20:50 | 41ab46b | First submission: single agent, our prompt, thinking off, temp 0.2, 4.5 min / 40 calls / 80 turns | _pending_ | Kaggle ref 56758822; zip sha256 369b1c64…. Validated with adk-submission 0.2.12 + google-adk 1.36.1. Expected 0.06–0.12 from comparable public configs |

## Ideas queue (one per submission)

- [ ] v2: add a read-only analyzer AgentTool (`skip_summarization: false`) to save context in the main agent
- [ ] Prompt variants: stronger "reproduce first" vs "edit first"
- [ ] Thinking on with a small budget (e.g. 1024), keeping everything else fixed
- [ ] Graph tools on vs off
- [ ] LoRA adapter trained on our own successful trajectories (Steps 4–5)
