# Concepts

A running notebook of every concept this project uses, in plain language. Each entry says what
the idea is, why it matters **in this competition**, and where it shows up in the repo.

New concepts get added as we use them. The **Used in** tag says which step introduced them, so you can
read along with the [roadmap in the README](README.md#roadmap).

---

## Contents

1. [The competition](#1-the-competition)
2. [The model](#2-the-model)
3. [Talking to the model](#3-talking-to-the-model)
4. [Agents](#4-agents)
5. [Code intelligence: graphs and embeddings](#5-code-intelligence-graphs-and-embeddings)
6. [Running experiments well](#6-running-experiments-well)
7. [Fine-tuning (preview, coming in later steps)](#7-fine-tuning-preview)

---

## 1. The competition

### SWE-bench-style task
**Used in:** Step 1

A real GitHub issue ("calling `foo(None)` crashes") plus a snapshot of the repository *just
before* it was fixed. The agent must produce the fix. The name comes from
[SWE-bench](https://www.swebench.com/), the standard benchmark for coding agents.

Each task in `tasks.jsonl` has: `instance_id`, `repo`, `base_commit`, `problem_statement` (the
issue text), `hints_text`, and, in the public training set only, the reference `patch` and the
`test_patch` that checks it.

### Patch / unified diff
**Used in:** Step 1

A text file that describes changes to files (`-` lines removed, `+` lines added). It's what
`git diff` prints. Our agent's answer **is** a diff: when it calls `submit_patch`, the harness
runs `git diff` inside the sandbox and saves the result.

This has a gotcha. *Any* file the agent leaves in `/workspace` (a scratch `repro.py`, say)
becomes part of the patch. That's why our prompt says to put scratch files in `/tmp`.

### Fail-to-pass and pass-to-pass tests
**Used in:** Step 1

How a fix is graded. The hidden `test_patch` adds tests that:
- **fail** on the original code (proving they detect the bug), and
- **pass** once a correct fix is applied, while the old tests keep passing.

Our patch goes into a *fresh* container, the hidden tests are added, and `pytest` must exit
cleanly. Any test files our agent edited are reset first, so editing tests can't game the score.

### Underspecified tasks (why scores are low)
**Used in:** Step 1 (reading `tasks.jsonl`)

Many issues don't fully say what the hidden test checks. Real example, `fastapi_14258`:

- The issue only says: *"Show a clear error on attempt to include router into itself."*
- The hidden test requires this **exact** text:
  `pytest.raises(AssertionError, match="Cannot include the same APIRouter instance into itself. Did you mean to include a different router?")`

A fix that raises a *different* clear error is correct in spirit, but it fails. No agent can
reliably guess wording like that, which is part of why even the best scores are around 0.15.
Our prompt's rule "match the exact names and messages in the issue" helps when the issue states
them; when it doesn't, following the repo's existing message style is the best bet.

### Resolution rate (the metric)
**Used in:** Step 1

`resolved tasks / total tasks`. A score of 0.10 means 10% of the issues were fixed. Every task is
pass/fail; there's no partial credit.

### Hidden test set, public vs private leaderboard
**Used in:** Step 1

We only see 129 *training* tasks (from fastapi, rich, requests and httpx). Scoring uses about 120
*hidden* tasks from **private repositories** we have never seen. Half of them drive the public
leaderboard; the other half (the private leaderboard) decide the final ranking.

**Why it matters:** anything that only works for fastapi won't help. Our prompts stay
general, and we shouldn't chase tiny public-leaderboard gains (see
[Overfitting to the leaderboard](#overfitting-to-the-public-leaderboard)).

### Sandbox / container
**Used in:** Step 1

An isolated Linux environment (Docker) where the agent's commands run. Here it is **offline** (no
internet, no `pip install`), with 4 GB RAM and 2 CPUs, and the repo sits at `/workspace`. Two
separate containers are used per task: A for the agent to work in, and B for clean verification.

---

## 2. The model

### Large language model (LLM) and tokens
**Used in:** Step 1

An LLM predicts the next **token** (a chunk of text, about 3–4 characters of English or code),
over and over. Everything is measured in tokens: input length, output length, and the context
window. Rough conversion: **1,000 tokens ≈ 3,500 characters**. A tool output capped at 5,000
characters is therefore about 1,300–1,700 tokens.

### Parameters and model size
**Used in:** Step 1

The model's learned numbers ("weights"). `gemma-4-31b` has about **31 billion** of them. Memory
needed just to hold the weights is roughly `parameters × bytes per parameter`:

| Precision | Bytes per weight | 31B model |
|---|---|---|
| bfloat16 (16-bit) | 2 | ~62 GB |
| int4 (4-bit) | 0.5 | ~16–18 GB |

### Instruction-tuned ("it") model
**Used in:** Step 1

A **base** model only continues text. An **instruction-tuned** model (the `-it` in the name) has
been further trained to follow instructions, chat, and call tools. We use the `-it` variant
because we need it to follow our system prompt and call tools.

### Quantization: W4A16, QAT, compressed-tensors
**Used in:** Step 1

Decoding the model name `gemma-4-31b-it-qat-w4a16-ct`:

- **Quantization** stores weights with fewer bits to save memory and speed up inference, at
  some cost in quality.
- **W4A16**: **W**eights in **4**-bit integers, **A**ctivations (the intermediate values computed
  while running) kept in **16**-bit. The 4-bit weights are expanded on the fly during the math.
- **QAT (Quantization-Aware Training)**: the model was trained while *simulating* 4-bit weights,
  so it learned to work well despite them. That's better than squashing a finished model
  afterwards ("post-training quantization").
- **ct (compressed-tensors)**: the file format the quantized weights are stored in. vLLM can load
  it directly.

**Why it matters:** it's the only model allowed. Later, when we fine-tune, the adapter we train has
to work on top of *this* quantized model (see [LoRA](#lora-low-rank-adaptation)).

### Inference server (vLLM) and tensor parallelism
**Used in:** Step 1

**vLLM** is software that loads a model on GPUs and serves it behind an **OpenAI-compatible HTTP
API**, so any client that speaks the OpenAI chat format can use it. The scorer runs it on **4 ×
NVIDIA L4 GPUs** (24 GB each).

**Tensor parallelism** (`tensor_parallel_size=4`) splits every layer's weight matrices across the
4 GPUs, so they compute each token together. It's used when a model, plus its working memory, is
too big or too slow for one GPU.

### GPU choice: memory and generation
**Used in:** Step 1 (planning compute for Steps 2 and 5)

Two questions decide whether a GPU can run our model:
1. **Is there enough memory?** You need the weights (~17–18 GB for our 4-bit 31B model) **plus**
   the [KV cache](#kv-cache) for a 32k context **plus** some overhead. Training needs much more:
   gradients, optimizer state, and activations saved for the backward pass.
2. **Is it a new enough generation?** GPUs have a "compute capability". vLLM's fast kernels for
   4-bit weights need **Ampere (2020) or newer**: A10G, A100, L4, L40S, H100. The older T4 can't
   use them, which is why other teams fell back to the slower llama.cpp on Kaggle's free T4s.

### KV cache
**Used in:** Step 1

While generating, the model stores intermediate "key" and "value" vectors for every token already
in the conversation, so it doesn't recompute them for each new token. That memory grows with
context length, which is one reason the scorer caps context at 32,768 tokens.

### Context window
**Used in:** Step 1 (it drives several decisions in `submission/`)

The maximum number of tokens the model can see at once: system prompt, task, the whole
conversation so far, and the reply it's writing. Here it is **32,768 tokens**.

Three facts make it tight in this competition:
1. The harness's first message (task, rules, file listing) already uses a few thousand tokens.
2. Every tool output stays in the history for the rest of the task. Other teams found the
   history is not usefully compressed mid-task, so it only grows.
3. vLLM rejects a request if `prompt tokens + max_output_tokens > 32,768`, so a big output cap
   *shrinks* the room left for history.

A task can die from a full context after roughly 20 large tool outputs. **In our config:**
`max_output_tokens: 4096`, and the prompt insists on short outputs (`| head -20`,
`sed -n '120,180p'`).

---

## 3. Talking to the model

### System prompt vs user message
**Used in:** Step 1 (`submission/prompts/system.md`)

Chat models receive a list of messages with roles:
- **system** (ADK calls it `instruction`): standing instructions on how to behave. *We write this.*
- **user**: the request. *The harness writes this*: the issue text, the budget, environment rules
  and a file listing.
- **assistant**: the model's replies and tool calls.
- **tool**: results of tool calls.

Our only lever in Step 1 is the system prompt and the settings around it. That is **prompt
engineering**.

### Prompt templating and the curly-brace trap
**Used in:** Step 1 (`scripts/build_submission.py` checks for it)

ADK replaces `{name}` in an instruction with `session.state["name"]`. The harness provides
`{problem_description}`, and `{hints}` *only when a task has hints*. Writing `{hints}`, or
`{anything_else}` around a single word, makes the agent crash with a `KeyError` at run time.
`{hints?}` (with a question mark) means "optional". Our build script refuses prompts containing
these traps.

### Sampling parameters
**Used in:** Step 1 (`submission/configs/sampling.yaml`)

At each step the model gives a probability for every possible next token. **Sampling** picks one.

- **temperature**: rescales the probabilities. `0` means always take the most likely token
  (deterministic); higher values give more variety and more mistakes. We use `0.2`, because code
  edits need precision.
- **top_k**: only consider the *k* most likely tokens (we use 40).
- **top_p** ("nucleus sampling"): only consider the smallest set of tokens whose probabilities add
  up to *p* (we use 0.95).
- **max_output_tokens**: the cap on the length of one reply (we use 4096; see
  [Context window](#context-window)).
- **seed**: makes sampling repeatable. We leave it unset.

Another team found temperature 0.2 vs 0.7 made no measurable difference here, so this setting
isn't where the points are.

### Thinking / reasoning tokens
**Used in:** Step 1

Gemma 4 can "think" (write hidden reasoning) before answering. It often helps on hard problems,
but it costs **time** (each task has only minutes) and **context** (thoughts can fill the window).
Every public config scoring 0.10–0.12 turns it off, and so do we: `include_thoughts: false`.
In the compiler this becomes `enable_thinking: false` sent to vLLM. It's a good candidate for a
controlled experiment later.

---

## 4. Agents

### Agent = LLM + tools + loop
**Used in:** Step 1

A plain LLM only produces text. An **agent** runs a loop:

```
repeat:
    model reads the conversation -> decides to call a tool (or to stop)
    harness runs the tool -> appends the result to the conversation
until the model stops or a budget runs out
```

Our agent's "hands" are the 9 tools the harness provides (`run_command`, `read_file`,
`edit_file`, `submit_patch`, …). It cannot do anything else.

### Tool calling (function calling)
**Used in:** Step 1 (`submission/agent.yaml` -> `tools:`)

The model is shown each tool's name, description and parameters. To use one, it emits a
structured call such as `edit_file(filepath="src/x.py", old_string="...", new_string="...")`.
The server parses that out of the text (vLLM's `tool_call_parser="gemma4"`), the harness runs it,
and the JSON result is fed back.

A common failure: a huge `edit_file` gets cut off by `max_output_tokens` before the call is
finished, so it never runs. That's why our prompt asks for small edits.

### Google ADK and declarative agent configs
**Used in:** Step 1

The **Agent Development Kit (ADK)** is Google's framework for building agents. Normally you
write Python. This competition only accepts **declarative YAML** (`agent.yaml`) so nobody can run
arbitrary code on the scoring machine. The organizers' `adk-submission` package compiles the
YAML into a real ADK agent, and our build script runs that same compiler locally when it's
installed.

`!include path` pastes another file in: `.md` files become text and `.yaml` files are parsed.
That's how `agent.yaml` pulls in `prompts/system.md` and `configs/sampling.yaml`.

### Multi-agent: sub-agents and AgentTool
**Used in:** introduced in Step 1 research; planned for experiment v2

You can split work between agents:
- **AgentTool** (`agent_tool:` in `tools:`): the parent calls a child agent *like a function*
  ("find where X is defined") and gets back only the child's final answer. The child's file
  reads never enter the parent's context, which saves our precious
  [context window](#context-window).
- **sub_agents**: the parent *hands over* the conversation to a child agent.

Most public 0.10–0.12 solutions use a "coder" plus a read-only "analyzer" AgentTool. We start
with a single agent (simpler to reason about) and test the analyzer as a separate experiment.

### Budgets
**Used in:** Step 1 (`submission/eval_config.yaml`)

The whole hidden run (~120 tasks, **one after another**) must finish within **12 hours**, or the
submission fails ("Notebook Exceeded Allowed Compute"). Per-task limits are ours to set:

| Setting | Ours | Meaning |
|---|---|---|
| `max_time_minutes` | 4.5 | agent wall-clock time per task |
| `max_tool_calls` | 40 | `submit_patch` and `get_status` are free |
| `max_turns` | 80 | model calls per task |
| `timeout_seconds` | 180 | per shell command |

Worst case: 120 × (4.5 + ~1 min sandbox setup) ≈ 11 h. Another team measured 3.5 min → 0.06,
4.5 min → 0.10, and 5.5 min → over the 12 h limit. `scripts/build_submission.py` does this
arithmetic for us.

### Nudges, `submit_patch`, and the fallback diff
**Used in:** Step 1

- If the model replies with text but no tool call, the harness "nudges" it to continue (at most 3
  times in a row).
- `submit_patch` records `git diff` as the answer. Once the model's current turn ends after a
  submit, the task is over.
- `submit_patch` does **not** end the task by itself. The task ends when the model next replies
  with plain text. So the agent can submit, keep checking, and submit again; the last submit
  counts.
- If the agent runs out of time or turns without submitting, the harness takes whatever diff is in
  `/workspace` anyway.
- But if the conversation outgrows the [context window](#context-window), the run crashes and
  the task scores 0. Even a patch already submitted is lost (we checked the harness code). The
  only protection is keeping tool outputs short, which is why our prompt is strict about
  `| head` and line ranges.

---

## 5. Code intelligence: graphs and embeddings

### Code graph (AST, call graph)
**Used in:** Step 1 (tools `get_code_neighbors`, `get_code_subgraph`)

The organizers parsed each repo into an **abstract syntax tree (AST)**, the program's structure as
a tree, and from it built a **graph**. **Nodes** are modules, classes and functions (for example
`fastapi.routing.APIRoute`). **Edges** are relationships such as "calls" or "imports".
`get_code_neighbors("Client.send")` answers "what calls this, and what does it call?" without
reading files.

### Embeddings and cosine similarity
**Used in:** Step 1 (tool `search_similar_code`)

An **embedding** is a list of numbers (here 256 of them) representing the meaning of a piece of
code, so that similar code gets similar vectors. **Cosine similarity** measures how closely two
vectors point in the same direction (1 means identical direction).

Gotcha: the sandbox is offline, so it can't embed a new English sentence. `search_similar_code`
only accepts an existing **symbol name** (such as `parse_header`) and returns code similar to that
symbol. Our prompt says so explicitly.

---

## 6. Running experiments well

### Baseline
**Used in:** Step 1

The simplest reasonable version, used as the reference point. Our `v1` (single agent, tuned
prompt, no fine-tuning) is the baseline that every later idea must beat.

### One change at a time (ablation)
**Used in:** Step 1 onwards (`docs/experiments.md`)

If you change the prompt *and* the temperature *and* add a sub-agent, and the score moves, you
don't know which change mattered. Change one thing per submission and log it. An **ablation** is
the reverse: remove one component to see how much it was contributing.

### Noise
**Used in:** Step 1

The public leaderboard has about 60 tasks, so **one task ≈ 0.017**. And with temperature > 0 the
same submission can solve different tasks on different runs. Another team saw 0.10, 0.08 and 0.10
from near-identical configs. **Differences of 1–2 tasks (≈0.02–0.03) are probably luck.**

### Overfitting to the public leaderboard
**Used in:** Step 1

Picking whatever scored best on the public ~60 tasks partly picks *luck*, and that luck doesn't
carry over to the private ~60. Prefer changes with a clear reason behind them, confirmed on our
own local evaluation once we have one (Step 2).

### Train / validation split
**Used in:** Step 2a (`scripts/make_split.py` → `eval/splits.json`)

Hold back some of the 129 public tasks and never tune on them, so they give an honest estimate of
how the agent does on unseen issues. This matters even more once we fine-tune, because training on
a task and then testing on it measures memorization, not skill. Our **dev** set (~33 tasks) is
only for testing; the **train** set is where Step 4's training data will come from.

Two kinds of **leakage** can quietly break a split:
- **Shared inputs:** two tasks built on the same repo snapshot must sit on the same side.
- **Time:** if training data is newer than test data, the model "knows the future". Within each
  repo, our train tasks are all older than our dev tasks.

### Data contamination
**Used in:** Step 2a

A model may have seen a benchmark's answers during its original training: these public issues and
their fixes are on GitHub. Then a high score can mean *remembering* instead of *solving*. The hidden
tasks come from private repos, so they're uncontaminated. That's one reason other teams' local
scores (0.18–0.24) beat their leaderboard scores (0.05–0.12). We reduce the problem by testing on
the **newest** public issues, the least likely to be in Gemma's training data.

### Controls (gold and none)
**Used in:** Step 2b (`kaggle/grader_check/`)

Before trusting a measuring instrument, check it on cases where you already know the answer:
- **Gold control:** apply the task's *real* fix. The grader must say "pass".
- **None control:** apply *nothing*. The grader must say "fail".

A task that fails either control can't be scored fairly, no matter how good the agent is, so we
exclude it. Example: `requests_6589` fails even with the real fix, because its tests need a pytest
plugin (`pytest-httpbin`) that isn't installed in the offline sandbox.

### Environment fidelity
**Used in:** Step 2b

The tests' result depends on the *environment* (Python version, package versions), not just on the
code. The official scorer gives every task one fixed set of packages: the newest of each in the
competition's `wheels/` folder. In notebook mode the harness instead let tasks use whatever the
notebook had installed. So we rebuild the scorer's package set and give each task only that. Two
small differences remain: Python 3.12 instead of 3.13, and pytest's exact version. When local and
leaderboard results disagree, environment differences are the first suspect.

### Failure analysis (error analysis)
**Used in:** Step 2d (`scripts/analyze_run.py`)

A score tells you *how much* fails; only reading the transcripts tells you *why*. Sort every failed
task into one category (loop, ran out of time, no patch, wrong fix, ...), count them, and fix the
biggest category first. Our v1 baseline: 15 of its 27 failures were one category (loops), which
no amount of prompt tweaking aimed elsewhere would have fixed.

### Repetition loops (degeneration)
**Used in:** Step 2d finding, Step 3

A language model can get stuck repeating itself: it runs the same `grep` again, gets the same
output, and the most likely next step is... the same `grep` again. Low temperature makes this
worse (it always picks the single most likely continuation), and so does an identical error
message that the model keeps "retrying" against. v1 (temperature 0.2) looped in 15 of 33 dev
tasks, up to 44 identical calls in a row. Usual remedies: the model's recommended sampling
settings (Gemma: temperature 1.0, top_p 0.95, top_k 64), a repetition/frequency penalty, and an
explicit rule in the prompt ("never repeat a call; if something fails twice, change approach").

### Malformed tool calls
**Used in:** Step 2d finding

The model writes tool calls as text, which a parser turns into function arguments. With long
code strings full of quotes and newlines it sometimes produces broken arguments (in v1, keys like
`"filepath": "fastapi/applications.py`,new_string:"`). The tool then rejects the call, and the
model often repeats the same broken call. Smaller edits (short `old_string`) break less often.

---

## 7. Fine-tuning (preview)

> These concepts appear in the competition rules (the `adapters/` folder), so they're defined
> here. We'll expand each one when we actually use it in Steps 3–4.

### Prompt engineering vs fine-tuning
Prompt engineering changes the *instructions*. Fine-tuning changes the *model's weights*.
Prompting is fast and free to iterate. Fine-tuning can teach behaviors that prompting can't
reliably get, such as consistent tool-call formats or better debugging habits, but it needs
data, GPUs and care. We start with prompting to get a working baseline and a data pipeline,
then fine-tune.

### Supervised fine-tuning (SFT)
Show the model examples of the behavior you want (inputs plus ideal outputs) and train it to
reproduce them. For an agent, an example is a whole **trajectory**: the issue, every tool call and
result, and the final patch.

### PEFT
**Parameter-Efficient Fine-Tuning**: instead of updating all 31B weights (huge memory, huge file),
train a small number of extra parameters. LoRA is the most popular PEFT method.

### LoRA (Low-Rank Adaptation)
Keep the original weight matrix `W` frozen and learn a small correction `ΔW = B × A`, where `A`
and `B` are thin matrices of **rank r** (for example 16). Only `A` and `B` are trained, often under
1% of the parameters.
- **rank (r)**: capacity of the update. The competition allows up to 128.
- **alpha**: scales the update (effective scale is `alpha / r`).
- **target modules**: which layers get adapters (such as `q_proj`, `v_proj`, `gate_proj`).

The result is an **adapter**: `adapter_config.json` plus `adapter_model.safetensors`, which go in
`adapters/<name>/`. Each agent in `agent.yaml` can name a different adapter (`adapter: main_lora`),
with up to 8 adapters and 3 GiB total.

### safetensors
A weight file format that stores only numbers. The older `.bin`/`.pt` formats use Python
"pickle", which can run code when loaded, so the competition rejects them.

### Trajectories and rejection sampling
Planned data recipe: run our agent many times on the 129 training tasks, **keep only the runs
whose patch passes the tests**, and fine-tune on those. This is called rejection sampling or
"expert iteration". The model learns from its own successes. Training on data generated by `gemma-4-31b` itself is
allowed. Outputs from other models are allowed only if that model's license permits it, and many
closed-model terms forbid training other models on their outputs.

### Reinforcement learning (stretch goal)
Instead of copying good examples, let the model try, score the attempt (did the tests pass?), and
nudge the weights toward higher-scoring behavior. GRPO and PPO are common algorithms. This is
powerful but expensive; a stretch goal for later.
