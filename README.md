# Cozmo: A Coding Agent for Your Terminal

Cozmo is a coding agent that runs in your terminal, inside your own repository. It finds the right code with hybrid search, reads only the lines it needs, asks before it writes a file or runs a command, and stops itself before it burns through your budget. The agent loop is written from scratch. There is no LangChain.

[![CI](https://github.com/ShashwatXD/cozmo/actions/workflows/ci.yml/badge.svg)](https://github.com/ShashwatXD/cozmo/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cozmo-agent.svg)](https://pypi.org/project/cozmo-agent/)
[![Python](https://img.shields.io/badge/Python-3.11%20|%203.12%20|%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

![Cozmo running in a terminal](https://github.com/user-attachments/assets/640dc42d-e9b0-42b7-a5d3-b981a5ce7e8f)

```bash
pipx install cozmo-agent
cd ~/your-project
cozmo
```

## Overview

Most coding agents either hide the loop behind a framework or paste whole files into the prompt until the context window fills up. Cozmo takes the opposite approach. **You own the loop**, and the harness decides how many bytes go back into the model.

It is built around three rules:

1. **Find before you read.** Exact names go through ripgrep. Vague questions go through hybrid semantic search. Files are read in line ranges, not whole.
2. **Nothing mutating runs without consent.** Writes and shell commands show a preview first, and you choose *allow once*, *always allow* or *deny*.
3. **Every run has hard limits.** Steps, tool calls, dollars and wall-clock time are all capped, and a stuck detector breaks repeating tool loops.

## Core Features

| Feature | Description |
|---|---|
| **Owned ReAct Loop** | A plain Python agent loop: model → tool calls → observations → repeat, with a typed stop reason on every run. |
| **Hybrid Code Search** | BM25 and embeddings fused with Reciprocal Rank Fusion, then a lexical reranker. Benchmarked below. |
| **Multi-Model Routing** | A strong *orchestrator* model for planning and summaries, and an optional cheaper *worker* model for tool calls. |
| **Subagents and Teams** | `run_subtask` sends out a read-only child agent that returns a compact evidence pack. `run_team` runs a director → researcher/coder/reviewer → synthesizer pipeline. |
| **Permission Gate** | Diff preview before writes, the full command before shell. Rules are saved per project in `.cozmo/permissions.json`. |
| **Guardrails** | Hard stops on steps, tool calls, cost (USD) and session time. Context is compacted automatically at about 70% of budget. |
| **Stuck Detector** | Spots repeated identical calls, repeated identical errors and A/B ping-pong, then pushes the model to change course. |
| **Modes** | `/ask` (read-only explain), `/plan` (explore, then propose) and `/agent` (full tools). |
| **Sessions** | Every turn is logged as JSONL. `/continue` resumes a session, `/export md\|json` writes an audit trail, and `search_history` searches past turns. |
| **MCP Client** | Connects any Model Context Protocol server over stdio. Its tools appear as `mcp__{server}__{tool}`. |
| **Groundedness Eval** | Checks every claim and `path:line` citation in an answer against the code that was actually retrieved. |
| **Any Provider** | OpenAI, Anthropic, OpenRouter, Ollama, or any OpenAI-compatible `base_url`. Bring your own key. |

## How It Works

```mermaid
graph TD
    A[You: question or task] --> B{Over context budget?};
    B -- Yes --> C[Orchestrator compacts older turns];
    C --> D;
    B -- No --> D[Worker model plans the next step];
    D --> E{Hard limit hit? steps / tools / $ / time};
    E -- Yes --> Z[Stop with a typed reason];
    E -- No --> F{Tool calls?};
    F -- No --> Y[Final answer + citations];
    F -- Yes --> G{Mutating tool?};
    G -- Yes --> H[Preview → allow / always / deny];
    H --> I;
    G -- No --> I[Run tool inside the workspace sandbox];
    I --> J[Trim output to the size cap];
    J --> K{Stuck pattern?};
    K -- Yes --> L[Inject a one-time course-correction hint];
    L --> D;
    K -- No --> D;
```

**Retrieval** (the `semantic_search` tool):

1. **Index.** Files are split into overlapping chunks of about 600 characters on line boundaries, embedded in batches that span files, and stored in Chroma. Only files whose content hash changed are re-embedded.
2. **Recall.** A query runs through BM25 (inverted index with identifier sub-word splitting, so `detect_stuck` also matches "stuck") and cosine vector search. The two lists are fused with Reciprocal Rank Fusion.
3. **Rerank.** The fused rank is kept as the main signal. It is refined with exact-match, content-word and path evidence.
4. **Expand.** Each hit is padded with surrounding source lines and returned as `path:start-end`.

## Benchmarks

All numbers below are reproducible with `python benchmarks/run_retrieval.py`. The corpus is Cozmo's own source at a pinned commit (97 files, 634 chunks), searched with **43 hand-labeled queries**: 28 natural-language questions and 15 identifier lookups. A query counts as a hit at *k* if a correct file appears in the top *k* distinct files. Embeddings come from local Ollama `nomic-embed-text`.

**Production pipeline (`semantic_search`), before and after the engineering pass:**

| Metric | Before | After |
|---|---|---|
| Recall@1 | 0.56 | **0.79** |
| Recall@5 | 0.93 | **0.98** |
| MRR@10 | 0.708 | **0.868** |
| `semantic_search` latency, p50 | 64.4 ms | **2.5 ms** |
| Embedding requests to index the corpus | 99 | **10** |
| Full index time (local Ollama, uncached) | 25.6 s | **21.6 s** |

**By strategy (after):**

| Strategy | R@1 | R@5 | MRR@10 | p50 |
|---|---|---|---|---|
| BM25 only | 0.44 | 0.98 | 0.664 | 0.16 ms |
| Vector only | 0.72 | 0.93 | 0.811 | 0.07 ms |
| Hybrid (RRF) | 0.74 | 0.93 | 0.834 | 0.53 ms |
| **Hybrid + rerank** | **0.79** | **0.98** | **0.868** | 2.5 ms |

What changed:

- **Reranker.** The old reranker multiplied the fused score by 0.15. RRF scores are around 0.03, so the hybrid ranking was effectively thrown away and the final order was purely lexical. The pipeline scored *below* plain hybrid search. The rerank now keeps the fused rank as its prior.
- **Latency.** BM25 used to be rebuilt over the whole repository on every tool call. It is now an inverted index cached per store version. Vector search was a pure-Python loop and is now a single numpy matrix-vector product.
- **Indexing.** Embedding requests are batched across files instead of one or more per file. A failing file is retried on its own so it cannot break the rest.

Raw reports are in [`benchmarks/results/`](benchmarks/results). CI runs the same benchmark offline (with a deterministic stub embedder) as a quality gate, so ranking regressions fail the build.

## Tech Stack

- **Language**: Python 3.11+
- **CLI**: Typer, Rich, questionary
- **LLMs**: OpenAI SDK (OpenAI / OpenRouter / Ollama / compatible), Anthropic Messages API over httpx
- **Retrieval**: Chroma (HNSW), numpy, a custom BM25 inverted index, Reciprocal Rank Fusion
- **Reliability**: tenacity retries with `retry-after` support, pydantic-settings
- **Tooling**: ripgrep, Model Context Protocol (`mcp`)
- **Quality**: pytest (188 tests), ruff, mypy, a coverage gate, and GitHub Actions on Python 3.11–3.13

## Project Structure

```
cozmo/
├── src/cozmo/
│   ├── domain/              # pure rules and types: messages, guardrails, events, ports
│   ├── app/                 # use-cases: agent loop, compaction, subagents, teams,
│   │                        #   permissions, stuck detector, groundedness, evals
│   ├── search/              # hybrid BM25 + vector search, rerank, context expansion
│   ├── infra/
│   │   ├── llm/             # OpenAI-compatible + Anthropic adapters, retries, model catalog
│   │   ├── rag/             # chunking, incremental indexer, Chroma / JSON vector stores
│   │   ├── tools/           # tool registry, sandbox, ripgrep search, permission store
│   │   ├── history/         # JSONL session store + history search
│   │   ├── mcp/             # MCP client manager and tool registration
│   │   └── telemetry/       # structured traces
│   ├── cli/                 # Typer entrypoint, REPL, setup wizard
│   └── prompts/             # versioned system prompts per mode and role
├── benchmarks/              # labeled retrieval queries, runner, saved results
├── tests/                   # unit, integration (real MCP server) and CLI tests
└── website/                 # project site
```

Dependencies point inward: `cli → app → domain ← infra`. The domain layer never imports a provider SDK.

## Getting Started

### Prerequisites

- Python 3.11 or newer
- An API key for OpenAI, Anthropic or OpenRouter, **or** [Ollama](https://ollama.com) running locally
- Optional: [ripgrep](https://github.com/BurntSushi/ripgrep) for faster exact search (Cozmo falls back to Python without it)

### 1. Install

```bash
pipx install cozmo-agent              # recommended
# or: pip install cozmo-agent
pip install "cozmo-agent[mcp]"        # optional: MCP tool servers
```

The package on PyPI is `cozmo-agent`. The command is `cozmo`.

### 2. Configure

```bash
cozmo setup
```

Pick a provider, paste your key and choose a model. Settings go to `~/.cozmo/config.json`, which is readable only by you. You can also set `COZMO_PROVIDER`, `COZMO_MODEL` and `COZMO_API_KEY` as environment variables.

### 3. Run

```bash
cd ~/your-project
cozmo                       # interactive agent
cozmo -m "why does add() return the wrong value?"   # one shot
cozmo --ask                 # read-only explain mode
cozmo --plan                # explore, then propose
```

The first launch indexes the repository into `.cozmo/`. Later launches only re-embed files that changed.

## Usage

**REPL commands**

| Command | Action |
|---|---|
| `/ask` `/plan` `/agent` | Switch mode |
| `/sessions` | List recent sessions |
| `/continue [id]` | Resume a session (latest if no id) |
| `/export md\|json` | Export the session audit trail |
| `/compact` | Summarize older turns now |
| `/clear` | Reset conversation memory |
| `/config` `/setup` | Show config path / re-run setup |
| `/exit` | Quit |

Each turn ends with a footer showing tokens, estimated cost, `ctx≈N/budget`, steps and the stop reason.

**Commands**

| Command | Description |
|---|---|
| `cozmo` / `cozmo agent` | Interactive agent |
| `cozmo chat` | Plain chat, no tools |
| `cozmo index` | Rebuild the code index |
| `cozmo eval` | Golden agent tasks + groundedness / citation eval |
| `cozmo doctor` | Show the effective settings |
| `cozmo config --show` | Print config (keys masked) |

<details>
<summary><b>Configuration reference</b></summary>

Precedence, highest first: CLI flags → `COZMO_*` env → `.env` → `<repo>/.cozmo/config.json` → `~/.cozmo/config.json` → defaults.

```json
{
  "provider": "openai",
  "model": "gpt-4o",
  "worker_model": "gpt-4o-mini",
  "allow_write": true,
  "allow_shell": false,
  "max_agent_steps": 8,
  "max_tool_calls_per_session": 40,
  "max_cost_usd": 0.50,
  "session_timeout_s": 600,
  "context_token_budget": 24000,
  "vector_backend": "chroma"
}
```

| Path | Contents |
|---|---|
| `<repo>/.cozmo/chroma/`, `index.json` | Code vector index |
| `<repo>/.cozmo/history/*.jsonl` | Session events |
| `<repo>/.cozmo/permissions.json` | Saved allow / deny rules |
| `<repo>/.cozmo/traces.jsonl` | LLM and tool telemetry |

</details>

<details>
<summary><b>Connecting MCP servers</b></summary>

Add to `~/.cozmo/config.json` or `<repo>/.cozmo/config.json`:

```json
"mcp_servers": [
  {
    "name": "filesystem",
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed"]
  }
]
```

A server that fails to start is skipped with a warning, and the others still connect. MCP tools are hidden in `/ask` and `/plan`. Set `COZMO_MCP_DISABLED=1` to turn MCP off.
</details>

## Development

```bash
git clone https://github.com/ShashwatXD/cozmo.git
cd cozmo
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,mcp]"

pytest --cov=cozmo                     # 188 tests, offline, coverage gate at 75%
ruff check . && ruff format --check .  # lint + format
mypy                                   # type-check
cozmo eval                             # golden agent eval
python benchmarks/run_retrieval.py --embedder ollama   # retrieval benchmark
```

Set `COZMO_DEBUG_CURL=1` to print each LLM request as a ready-to-run `curl` command (keys redacted).

## Privacy

- Your code index, sessions and permissions stay in `.cozmo/` inside your repository. Your key stays in `~/.cozmo/config.json`, readable only by you.
- Tools cannot read or write outside the workspace root. Shell access is off until you turn it on.
- Code is sent only to the model provider you configured. With Ollama, nothing leaves your machine.

## Contributing

Issues and pull requests are welcome. Please run `pytest`, `ruff check .` and `mypy` before opening a pull request.

## License

MIT. Named after Anki's Cozmo robot. This is an unaffiliated fan project.
