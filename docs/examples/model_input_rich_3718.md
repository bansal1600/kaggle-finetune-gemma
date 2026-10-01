# What the model actually receives (task rich_3718)

Captured by sending one real task through the official harness with our v1 config and a fake model (`scripts/mock_llm.py`), so this is exactly what Gemma gets on its first turn. See concepts.md → "System prompt vs user message" and "Tool calling".

## Generation settings sent with every request

```json
{
  "temperature": 0.2,
  "top_p": 0.95,
  "top_k": 40.0,
  "max_completion_tokens": 4096,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

## 1. System message (written by us: `submission/prompts/system.md`)

Our prompt from [submission/prompts/system.md](../../submission/prompts/system.md), sent unchanged, followed by one sentence that ADK appends from `agent.yaml`:

```text
You are an agent. Your internal name is "swe_coder". The description about you is "Fixes a GitHub issue in a Python repository with a small, verified source patch.".
```

## 2. User message (written by the harness)

```text
You are evaluating a software engineering task for repository Textualize/rich.

Problem Statement:
fix(panel): fix title missing panel background

 Fix `Panel` title missing the panel background style.
    
This really just reverts the change in 7a38204. There's a history of issues related to the panel title styles, so I was careful to run all the examples in those issues to ensure there wasn't any regression.
    
Fixes #3569

## Type of changes

- [x] Bug fix
- [ ] New feature
- [ ] Documentation / docstrings
- [ ] Tests
- [ ] Other

## Checklist

- [x] I've run the latest [black](https://github.com/psf/black) with default args on new code.
- [x] I've updated CHANGELOG.md and CONTRIBUTORS.md where appropriate.
- [x] I've added tests for new code.
- [x] I accept that @willmcgugan may be pedantic in the code review.


## Task Budget (Session terminates when any budget is exhausted)
- Time allowance: 4.5 minutes
- Tool calls allowance: 40 calls
- Max loop iterations: 80 turns

## Execution Environment Rules
- Single command timeout: 180 seconds (commands exceeding this fail without ending the session)
- Command output limit: 5000 characters
- File view limit: 150 lines per read_file call
- File character limit: 10000 characters per read_file call
- Environment is offline (no network/PyPI access). All repository and test dependencies are ALREADY pre-installed. Do NOT attempt to run pip install or download packages.

## Instructions:
0. All source code is under `/workspace`. Do NOT search outside `/workspace` (e.g. `/usr/`, `/wheels/`, system site-packages). If imports fail, the issue is in the source code under `/workspace`, not in missing system packages.
1. Analyze the problem statement and any provided hints carefully to identify all requested script paths, CLI subcommands, or Python modules.
2. Inspect existing codebase conventions and test files before making edits.
3. Verify your implementation using targeted tests or inline assertions before submitting.
4. Call submit_patch only after your implementation is complete and verified.
5. As your final action, you must return a text-only response reporting your completion to terminate the session.
## Workspace Layout
The repository is located at `/workspace`. Here is the directory tree (up to 3 levels):
```
.
./.coveragerc
./.faq
./.faq/FAQ.md
./.faq/suggest.md
./.git
./.github
./.github/FUNDING.yml
./.github/ISSUE_TEMPLATE
./.github/ISSUE_TEMPLATE/bug_report.md
./.github/ISSUE_TEMPLATE/feature_request.md
./.github/dependabot.yml
./.github/pull_request_template.md
./.github/workflows
./.github/workflows/codeql.yml
./.github/workflows/codespell.yml
./.github/workflows/comment.yml
./.github/workflows/newissue.yml
./.github/workflows/pythonpackage.yml
./.github/workflows/readmechanged.yml
./.gitignore
./.pre-commit-config.yaml
./.readthedocs.yml
./CHANGELOG.md
./CODE_OF_CONDUCT.md
./CONTRIBUTING.md
./CONTRIBUTORS.md
./FAQ.md
./LICENSE
./Makefile
./README.cn.md
./README.de-ch.md
./README.de.md
./README.es.md
./README.fa.md
./README.fr.md
./README.hi.md
./README.id.md
./README.it.md
./README.ja.md
./README.kr.md
./README.md
./README.pl.md
./README.pt-br.md
./README.ru.md
./README.sv.md
./README.tr.md
./README.zh-tw.md
./SECURITY.md
./assets
./assets/logo.ai
./assets/logo.svg
./assets/logo.txt
./asv.conf.json
./asvhashfile
./benchmarks
./benchmarks/README.md
./benchmarks/__init__.py
./benchmarks/benchmarks.py
./benchmarks/results
./benchmarks/results/benchmarks.json
./benchmarks/results/darrenburns-2022-mbp
./benchmarks/snippets.py
./conftest.py
./docs
./docs/Makefile
./docs/images
./docs/images/box.svg
./docs/images/svg_export.svg
./docs/make.bat
./docs/requirements.txt
./docs/source
./docs/source/appendix
./docs/source/appendix.rst
./docs/source/columns.rst
./docs/source/conf.py
./docs/source/console.rst
./docs/source/group.rst
./docs/source/highlighting.rst
./docs/source/index.rst
./docs/source/introduction.rst
./docs/source/layout.rst
./docs/source/live.rst
./docs/source/logging.rst
./docs/source/markdown.rst
./docs/source/markup.rst
./docs/source/padding.rst
./docs/source/panel.rst
./docs/source/pretty.rst
./docs/source/progress.rst
./docs/source/prompt.rst
./docs/source/protocol.rst
./docs/source/reference
./docs/source/reference.rst
./docs/source/style.rst
./docs/source/syntax.rst
./docs/source/tables.rst
./docs/source/text.rst
./docs/source/traceback.rst
./docs/source/tree.rst
./examples
./examples/README.md
./examples/attrs.py
./examples/bars.py
./examples/columns.py
./examples/cp_progress.py
./examples/downloader.py
./examples/dynamic_progress.py
./examples/exception.py
./examples/export.py
./examples/file_progress.py
./examples/fullscreen.py
./examples/group.py
./examples/group2.py
./examples/highlighter.py
./examples/jobs.py
./examples/justify.py
./examples/justify2.py
./examples/layout.py
./examples/link.py
./examples/listdir.py
./examples/live_progress.py
./examples/log.py
./examples/overflow.py
./examples/padding.py
./examples/print_calendar.py
./examples/rainbow.py
./examples/recursive_error.py
./examples/repr.py
./examples/save_table_svg.py
./examples/screen.py
./examples/spinners.py
./examples/status.py
./examples/suppress.py
./examples/table.py
./examples/table_movie.py
./examples/top_lite_simulator.py
./examples/tree.py
./faq.yml
./imgs
./imgs/columns.png
./imgs/downloader.gif
./imgs/features.png
./imgs/hello_world.png
./imgs/inspect.png
./imgs/log.png
./imgs/logging.png
./imgs/logo.svg
./imgs/markdown.png
./imgs/print.png
```

```

## 3. Tools offered (names and descriptions; parameter schemas omitted)

- **`run_command`**(command): Execute a shell command inside the repository sandbox.
- **`edit_file`**(filepath, old_string, new_string, allow_multiple): Replace a contiguous block of text in an existing file.
- **`write_file`**(filepath, content): Create or overwrite a file in the workspace.
- **`read_file`**(filepath, start_line, end_line): Read the contents of a file from the repository workspace.
- **`get_status`**(): Check active budget details (time, tool calls) and active patch details.
- **`submit_patch`**(): Capture the current working tree modifications as the agent's submission.
- **`search_similar_code`**(query, k): Find semantically similar code functions and classes using graph vector embeddings.
- **`get_code_neighbors`**(node, edge_type, max_neighbors): Discover structural code graph neighbors (callers, callees, definitions) for a given code node.

## 4. How the conversation grows

Each later request repeats everything above plus every step so far. The 4th request of this run had these roles:

`system → user → assistant → tool → assistant → tool → assistant → tool`

This is why tool outputs must stay short: nothing is ever removed within a task (concepts.md → "Context window").

