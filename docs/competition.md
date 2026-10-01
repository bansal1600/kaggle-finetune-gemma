# Competition digest

Our own summary of the rules and the scoring harness, collected 2026-10-01. Primary sources: the
competition [overview](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview),
[data](https://www.kaggle.com/competitions/gemma-4-developer-agent/data) and
[rules](https://www.kaggle.com/competitions/gemma-4-developer-agent/rules) pages, plus
`HARNESS_README.md` from the data tab (download it after accepting the rules). Findings marked
*(reported)* come from other teams' public notes, not from the organizers.

## What we submit

`submission.zip` with `agent.yaml` at the root (< 3 GiB unpacked):

```
agent.yaml            required root agent config (Google ADK Agent Config, YAML only)
eval_config.yaml      optional per-task budgets
configs/*.yaml        e.g. sampling parameters, pulled in with !include
prompts/*.md          system prompts, pulled in with !include
sub_agents/*.yaml     extra agents (sub_agents or agent_tool)
adapters/<name>/      LoRA: adapter_config.json + adapter_model.safetensors
skills/<name>/        SKILL.md (+ scripts/ run in the sandbox, resources/)
```

- Allowed file types: `.yaml .yml .md .txt .py .json .safetensors`. No symlinks, no `..` paths,
  no Python entry points (only skill scripts, which run inside the sandbox).
- Every agent must use the model `gemma-4-31b-it-qat-w4a16-ct`.
- LoRA adapters: rank ≤ 128, up to 8 adapters, and each agent may use a different one.
- `generate_content_config` may set `temperature`, `top_p`, `top_k`, `max_output_tokens`
  (≤ 32,768), penalties, `stop_sequences`, `seed`, and `thinking_config`.

## How it's scored

1. Kaggle serves the model with vLLM on 4 × L4 GPUs (context window 32,768 tokens).
2. For each of ~120 hidden tasks (private repos), **sequentially**:
   - **Container A** (offline, 4 GB RAM, 2 CPUs): the repo snapshot is placed at `/workspace` and
     committed as a baseline. Our agent gets a task message and works until it calls
     `submit_patch`, a budget runs out, or it stops calling tools 3 times in a row.
   - The patch is `git diff` of `/workspace`, taken even if the agent never submits.
   - **Container B**: fresh snapshot → apply our patch → reset any test or config files we touched
     → apply the hidden tests → `pytest`. The task is resolved if pytest exits 0 and the
     required tests passed.
3. Score = resolved / total. The public leaderboard uses about half the tasks and the private
   leaderboard the other half.
4. **Hard limit: 12 hours for all tasks** (sandbox setup included, verification excluded).
   Exceeding it fails the whole submission. The organizers say per-task budgets in
   `eval_config.yaml` are how to stay under it.

## The task message (written by the harness)

The issue text, hints (if any), the budget, environment rules (300 s per command by default,
5,000-character output cap, `read_file` ≤ 150 lines / 10,000 characters, offline), standard
instructions, a note about the code-graph tools, and the first 150 entries of the repo's file
tree.

## Tools

| Tool | Counts toward budget | Notes |
|---|---|---|
| `run_command(command)` | yes | `bash -c` in `/workspace`; output truncated to 5,000 chars |
| `read_file(filepath, start_line, end_line)` | yes | 1-indexed, inclusive; max 150 lines / 10k chars |
| `edit_file(filepath, old_string, new_string, allow_multiple)` | yes | exact match, then whitespace-tolerant, then regex; fails on 0 or > 1 matches |
| `write_file(filepath, content)` | yes | creates or overwrites |
| `get_status()` | **no** | budget left, patch status |
| `submit_patch()` | **no** | records `git diff`; the task ends when the current turn ends |
| `search_similar_code(query, k)` | yes | `query` must be a **symbol name**, not a sentence |
| `get_code_neighbors(node, edge_type, max_neighbors)` | yes | callers, callees, imports of a symbol |
| `get_code_subgraph(nodes)` | yes | edges among a set of symbols |

## `eval_config.yaml`

```yaml
evaluation:
  max_time_minutes: 4.5   # default 60 without this file
  max_tool_calls: 40      # default 100
  max_turns: 80           # default 500
  timeout_seconds: 180    # per command, default 300
```

The organizers' sample uses 1 minute / 10 calls. That's only for testing, and it scores near 0.

## Public training data (129 tasks)

fastapi 67, rich 48, requests 13, httpx 1. Each task has a repo snapshot (`snapshots/*.tgz`,
~20 GB total), a code graph and embeddings, the issue text, the reference patch and the tests.
No task has hints. The hidden test set comes from **different, private** repos.

## Checked directly on Kaggle (2026-10-01)

- Competition settings from Kaggle's API: 1 submission per day, 50% of the hidden tasks on the
  public leaderboard, required file `submission.zip`, 720-minute run limit, 1,200 teams.
- `HARNESS_README.md` on the data tab is unchanged since 2026-09-25, byte-identical to the copy
  this digest was first written from.
- **Scoring packages updated 2026-09-30**: `adk-submission` 0.2.12, `google-adk` 1.36.1,
  `swegemma` 0.2.7 (Kaggle dataset `metric/gemma-4-developer-agent-wheelhouse`). Our v1 passes the
  0.2.12 compiler.
- **GPU outage on the night of Sep 30 to Oct 1**: many submissions failed with "Notebook Threw
  Exception". The admin says it's resolved. A failed run still uses up that day's submission.
- Fixed in the current packages, per an admin: `read_file` with line ranges crashing
  (`'>' not supported between 'int' and 'str'`), and thinking-mode reasoning being dropped between
  tool calls. That makes "thinking on" worth re-testing later.
- **Context overflow scores 0.** We read `swegemma/harness/agent_runner.py` (0.2.7): when the prompt
  outgrows the context window, the error skips patch extraction entirely. Even a patch already sent
  with `submit_patch` is lost, not just unsaved edits. One team saw 12–33% of tasks overflow. The
  only defense is keeping tool outputs short and finishing early.
- **LoRA adapters currently shrink the KV cache to ~7,600 tokens** on 4×L4, so long tasks stall.
  An admin is patching it to size LoRA buffers from the submission. Re-check before Step 5.
- Training on other models' outputs: the admin says it's allowed if you follow that model's
  license terms. Many closed-model terms forbid using outputs to train other models, so check
  before doing it.
- Notebooks attached to this competition can use **4×L4 machines**, the same hardware as the
  scorer, at double the GPU-quota rate and with internet disabled.

## Lessons from other teams (reported, up to 2026-09-30)

- Every public config scoring 0.10–0.12 uses **thinking off**, temperature 0.15–0.2 and an
  output cap of 4,096–8,192, and most add a read-only "analyzer" AgentTool. None of them use LoRA.
- Per-task time: 3.5 min → 0.06, 4.5 min → 0.10, 5.0 min → 0.08, 5.5 min → exceeded 12 h.
  Differences of 1–2 tasks are within run-to-run noise.
- The context window fills up after ~20 large tool outputs; nothing useful is compacted mid-task.
- An AgentTool with `skip_summarization: true` ends the parent's turn after each call and
  triggers a harness nudge; `false` behaves better.
- Running the 31B model on Kaggle's free T4 × 2 only works through llama.cpp, and slowly (vLLM's
  INT4 kernels need newer GPUs). The organizers' starter notebook uses the `NvidiaL4` machine.
- vLLM LoRA on this W4A16 model: rank 8 on some layers worked; rank 32 on all layers was reported
  unstable. Test before training a big adapter.
- The organizers allow training on data generated by `gemma-4-31b` itself (other models: see
  the 2026-10-01 update above).

Public notes we read: [happyc0der/gemma-swe-agent](https://github.com/happyc0der/gemma-swe-agent),
[LogosTopos/gemma-4-developer-agent](https://github.com/LogosTopos/gemma-4-developer-agent),
[emiliodavola/kaggle-gemma-agent](https://github.com/emiliodavola/kaggle-gemma-agent),
[rishaviitd/kaggle.gemma.coding.agent](https://github.com/rishaviitd/kaggle.gemma.coding.agent).
