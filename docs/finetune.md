# Fine-tuning plan (Step 4)

The plan was designed on 2026-10-13 by a research workflow. It checked the harness, vLLM and
training-library source code, then went through two adversarial reviews. The full
machine-readable version, with source references, is `docs/plans/finetune-plan-2026-10-13.json`.
The ideas it uses are explained in concepts.md §7 (QLoRA, train/serve mismatch, loss masking,
paired comparison).

## Decisions

| Question | Decision | Why |
|---|---|---|
| Thinking | **On**: budget 4,096, thoughts kept | 129-task public sweep: 23 resolved off vs 48–51 on. Every public 0.17–0.18 config uses it. Training data must use the deployment setting, so this comes first |
| Model | 31B (`gemma-4-31b-it-qat-w4a16-ct`) | 12B/E4B/E2B are allowed since 10-11 but weaker; 31B has the full analysis behind it |
| Training base | `gemma-4-31b-it-qat-q4_0-unquantized` (bf16) | The checkpoint the served 4-bit model was made from (weights 99.8% aligned) |
| Method | QLoRA: base in NF4, rank 16, alpha 32, all 7 projections of the language model | Fits 4×L4. vLLM can apply LoRA to exactly these layers |
| Data | Our own agent's successful runs on the 72 train tasks (rejection sampling), exact vLLM token ids | Stronger teacher models barely beat 31B here (forum 747168). Exact tokens avoid formatting mismatch |
| Decision | Paired adapter-vs-base dev runs, pre-registered GO rule (McNemar p ≤ 0.05 and ≥ +1.5 tasks/run) | Single runs vary ±2–3 tasks |

## Stages and gates

Each stage has a check. If it fails, we stop the fine-tuning track and put the GPU hours back into
prompt work.

| # | Stage | Where | 4×L4 hours | Gate |
|---|---|---|---|---|
| 0 | Build: capture proxy, adapter tools, FT-0 notebook, paired stats, base-shard check | CPU | 0 | Fake-model test captures every call |
| 1 | **FT-0**: dummy adapter on the scorer's exact vLLM. Does it start, keep its memory, change outputs? | Kaggle | 0.75 | Fail → stop the FT track |
| 2 | **T0**: thinking-on (`v3t_think`) on all 33 dev tasks, recording data | Kaggle | 2.9 | Finished, 0 overflows, ≥ 5/6 core, ≤ 10.5 h projected → candidate submission |
| 3 | Data: 72 train tasks, pass 1 (+ pass 2 on solved tasks) | Kaggle ×2 | 5.9 + 3.6 | < 15 distinct solved tasks → stop the FT track |
| 4 | Convert recordings into loss-masked training examples, plus quality checks | CPU | 0 | ≥ 40 trajectories from ≥ 15 tasks |
| 5 | Train the rank-16 QLoRA | Kaggle (or Lightning) | 3.25 | Validation loss drops |
| 6 | FT-0R + E0: the trained adapter served by vLLM keeps ≥ 50% of its training gain | Kaggle | 0.5 | Fail → stop |
| 7 | E1/E2: 4 paired adapter-vs-base dev runs | Kaggle | 4.2 + 4.2 | GO rule (see below) |
| 8 | Submit the adapter only on GO; keep an adapter-free config as one final pick | scorer | 0 | |

**GO rule (fixed in advance).** The adapter must:
- gain at least +1.5 tasks per run on average;
- reach an exact one-sided McNemar p ≤ 0.05;
- keep the always-solved tasks (≥ 23 of 24 cells);
- make ≤ 2 garbled calls per run;
- have ≤ 6% repeats and 0 context overflows;
- project to ≤ 10.5 h for the hidden set.

## Budget

`kaggle quota` (2026-10-13): 30 GPU-hours per week, refreshing Saturdays 00:00 UTC. A 4×L4
session probably counts double, so that is about 15 machine-hours a week.

| Week | Stages | 4×L4 hours |
|---|---|---|
| to 10-17 | 1, 2, 3, 4 | ~13.2 (+1.9 reserve) |
| to 10-24 | 5, 6, 7 | ~12.2 (+2.9 reserve) |

Submissions queue for about 33 h, and only one can be pending at a time. The final-selection
deadline is 2026-12-02.

## Honest expectations

- **No public proof yet.** No team has publicly shown a LoRA submission beating its own base.
  The one published trained-LoRA test was a NO-GO. The forum claim that "LoRA teams dominate"
  is unproven.
- **Small, hard-to-detect gain.** The expected gain is 0 to +2 dev tasks, about what four
  paired runs can just detect.
- **The sure gain is elsewhere.** Thinking-on (Stage 2) is the step most likely to raise the
  leaderboard score. The sure gain from fine-tuning is a working, measured pipeline and
  understanding how each stage works.
