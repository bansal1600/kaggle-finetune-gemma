# kaggle-finetune-gemma

Our entry for the Kaggle competition
[**Google – The Gemma 4 Developer Agent Competition**](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview).

**Two goals:**
1. Submit a competitive entry: an autonomous coding agent built on Gemma 4 that fixes real GitHub
   issues.
2. Learn fine-tuning and AI engineering along the way. Every concept we use is explained in
   **[concepts.md](concepts.md)**. Read it alongside the code.

## The competition in one paragraph

We give Kaggle a small zip containing an **agent config** (YAML), not predictions. Kaggle runs our
agent with the `gemma-4-31b-it-qat-w4a16-ct` model on ~120 hidden GitHub issues from private
Python repositories. For each issue, the agent explores the repo in an offline sandbox, edits
code, and submits a patch. A patch scores if the hidden tests pass. Score = fraction of issues
fixed. We may also include **LoRA adapters** (fine-tuned weights) in the zip, and that's where
fine-tuning comes in. Full details: [docs/competition.md](docs/competition.md).

| Date (23:59 UTC) | Deadline |
|---|---|
| 2026-11-12 | Optional research-paper track |
| **2026-11-25** | **Entry deadline: accept the rules on Kaggle before this** |
| **2026-12-02** | **Final submission** |

## Roadmap

| Step | What we build | What we learn | Status |
|---|---|---|---|
| 0 | Read the rules, harness and other teams' public findings | SWE-bench tasks, agents, tokens, context windows, quantization | ✅ done |
| 1 | **v1 baseline:** single agent, tuned prompt, safe budgets; local validator + zip builder | Prompt engineering, sampling, tool calling, budgets | ✅ built, ⏳ submit and record the score |
| 2 | Local evaluation on a Kaggle GPU notebook with the official harness, on a held-out set of the 129 public tasks | Evaluation methodology, noise, reading agent trajectories, failure analysis | next |
| 3 | Scaffold experiments, one change at a time: analyzer sub-agent, prompt variants, skills | Multi-agent design, ablations | |
| 4 | Fine-tuning dataset from our agent's own successful runs | Trajectories, rejection sampling, chat templates, train/val splits | |
| 5 | Train a LoRA adapter, check it loads on the quantized model in vLLM, evaluate, submit | SFT, LoRA (rank, alpha, target modules), overfitting | |
| 6 | Stretch: RL (GRPO), several specialized adapters, paper track | Reinforcement learning | |

### Compute plan

Step 1 needs no GPU: Kaggle runs the model when it scores a submission. GPUs come in from Step 2.

| Resource | What we have | Use it for |
|---|---|---|
| Kaggle notebooks | Free weekly GPU quota; the organizers' starter notebook uses the `NvidiaL4` machine | Step 2 runs with the official harness, close to the real scorer |
| [Lightning.ai](https://lightning.ai/) Studios | 80 free GPU hours | Step 2 evaluation runs and Step 5 LoRA training |

Picking a GPU for this model (see concepts.md → "GPU choice"):
- **vLLM serving** (Steps 2–5): needs an Ampere-or-newer GPU (A10G, L4, L40S, A100, H100), **not
  T4**. The 31B model's weights alone take ~17–18 GB, so plan on **≥ 48 GB of GPU memory** (one
  L40S or A100, or several smaller GPUs) for the full 32k context.
- **LoRA training** (Step 5): the most memory-hungry step; plan on an A100 80 GB or H100.
- Free hours are limited: stop the Studio when a run finishes, and test the pipeline on a few
  tasks before launching a long run.

Where the leaderboard stood (late September 2026): top public score **0.15**; most good public
configs (prompt-only, no fine-tuning) score **0.10–0.12**. One public task ≈ 0.017, so differences
of 0.02 are mostly noise.

## Repository layout

```
concepts.md                   every concept we use, explained (start here)
docs/competition.md           rules, harness, tools and budgets, in our own words
docs/experiments.md           log of every submission: what changed, what it scored
submission/                   exactly what goes into submission.zip
  agent.yaml                    root agent (model, tools, prompt, sampling)
  eval_config.yaml              per-task budgets (time, tool calls, turns)
  configs/sampling.yaml         temperature, top_p/top_k, output cap, thinking off
  prompts/system.md             the system prompt: our main lever in Step 1
scripts/build_submission.py   validates submission/ against the rules, then builds dist/submission.zip
tests/                        tests for the build script
```

## How to submit

1. **Join the competition once.** On the [competition page](https://www.kaggle.com/competitions/gemma-4-developer-agent),
   click *Join Competition* and accept the rules (required before 2026-11-25).
2. **Build and validate:**
   ```bash
   pip install -r requirements.txt
   python scripts/build_submission.py        # checks rules, writes dist/submission.zip
   python -m pytest -q                       # optional: tests for the build script
   ```
   The script refuses to build if anything breaks a competition rule.
3. **Upload** `dist/submission.zip`, either:
   - on the website: competition page → *Submit Prediction* → upload the zip, or
   - with the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) (needs an API token from
     Kaggle → Settings → API → *Create New Token*, saved as `~/.kaggle/kaggle.json`):
     ```bash
     kaggle competitions submit -c gemma-4-developer-agent -f dist/submission.zip -m "v1 baseline"
     ```
4. **Wait** about 6–10 hours for scoring (the agent really runs on ~120 issues). You get **1
   submission per day**.
5. **Record the result** in [docs/experiments.md](docs/experiments.md).

### Local validation with the official compiler (optional, recommended)

The build script always runs our own rule checks. If the organizers' compiler (`adk-submission`)
is installed, it also compiles the agent exactly as the scorer does. That catches schema mistakes
before you spend a daily submission. The package isn't on PyPI; it ships in the Kaggle dataset
[`metric/gemma-4-developer-agent-wheelhouse`](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse):

```bash
kaggle datasets download metric/gemma-4-developer-agent-wheelhouse -f adk_submission-0.2.11-py3-none-any.whl -p vendor/
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt "google-adk>=1.34,<2" vendor/adk_submission-0.2.11-py3-none-any.whl
python scripts/build_submission.py        # now also prints "official compiler: OK"
```

(If the file name differs, list the dataset with `kaggle datasets files metric/gemma-4-developer-agent-wheelhouse`.)
