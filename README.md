# Cozmo

**Local-first CLI coding agent** for your repository.

Owned ReAct loop, multi-model routing, hybrid RAG, permission-gated tools, and production-style guardrails — without a LangChain dependency.

[![PyPI](https://img.shields.io/pypi/v/cozmo-agent.svg)](https://pypi.org/project/cozmo-agent/)
[![Python](https://img.shields.io/pypi/pyversions/cozmo-agent.svg)](https://pypi.org/project/cozmo-agent/)
[![License](https://img.shields.io/pypi/l/cozmo-agent.svg)](./LICENSE)

```bash
pipx install cozmo-agent
cd ~/your-project
cozmo
```

Package name on PyPI is `cozmo-agent`; the CLI binary is `cozmo`. Requires **Python 3.11+**.

---

## Why Cozmo

| Concern | Approach |
|---------|----------|
| Control | You own the agent loop — steps, tools, memory, stop reasons |
| Cost / quality | Orchestrator + worker models; optional cheap worker for tool calls |
| Safety | Workspace sandbox; writes and shell off until enabled |
| Context | Auto compaction + hard budgets (steps, tools, cost, time) |
| Retrieval | Hybrid BM25 + embeddings, lexical rerank, symbol / graph tools |
| Ops | Session history + traces under `.cozmo/`; `doctor` / `config` |

---

## Install

**Recommended (pipx):**

```bash
brew install pipx && pipx ensurepath   # macOS / Homebrew Python
pipx install cozmo-agent
```

**Virtualenv:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install cozmo-agent
```

Optional ANN backend:

```bash
pip install "cozmo-agent[vector]"   # chromadb
# then: COZMO_VECTOR_BACKEND=chroma
```

---

## Quick start

```bash
cozmo                 # interactive agent in cwd
cozmo setup           # provider, API key, model
cozmo -m "…"          # one-shot (no REPL)
cozmo doctor          # effective settings
```

First launch walks you through provider setup, writes `~/.cozmo/config.json`, indexes the repo into `.cozmo/`, then opens the REPL.

**REPL**

| Command | Action |
|---------|--------|
| `/help` | Commands |
| `/clear` | Reset conversation memory |
| `/compact` | Summarize older turns now |
| `/history` | Recent session ids |
| `/config` | Config path |
| `/setup` | Re-run wizard |
| `/exit` | Quit |

Each turn footer reports token usage estimate, `steps`, and `stop` reason (`completed`, `max_iterations`, …).

---

## Architecture

```text
cli/  →  app/  →  domain/  ←  infra/
                 ↑
            search / indexer
```

```text
user
  │
  ├─► guardrails (compact | kill on budgets)
  ├─► orchestrator (synthesis / compaction)
  ├─► worker (ReAct tool loop)
  │      ├─ read / write / search / git / symbols
  │      ├─ semantic_search → hybrid → rerank → expand
  │      └─ run_subtask → scoped subagent → JSON summary
  └─► .cozmo/history/*.jsonl   (+ traces.jsonl for telemetry)
```

Layers stay fixed: **view → view-model → domain ← infrastructure**. Domain never imports provider SDKs.

---

## Capabilities

### Multi-model

Set a strong `model` and an optional cheaper `worker_model`:

```bash
export COZMO_MODEL=claude-sonnet-4-20250514
export COZMO_WORKER_MODEL=gpt-4o-mini
```

| Role | Typical use |
|------|-------------|
| Orchestrator | Compaction, hard reasoning (when routed) |
| Worker | Tool loop |
| Verifier | Optional; omitted unless `verifier_model` is set |

### Guardrails

| Knob | Default | Behavior |
|------|---------|----------|
| `max_agent_steps` | 8 | Hard stop after N LLM steps |
| `max_messages_before_compact` | 30 | Soft: summarize older turns |
| `context_token_budget` | 24000 | Soft: compact at ~70% |
| `max_tool_calls_per_session` | 40 | Hard stop |
| `max_cost_usd` | unset | Optional hard stop |
| `session_timeout_s` | 600 | Optional hard stop |
| `max_subagent_depth` / `steps` | 1 / 4 | Nested agent caps |

### Retrieval

1. Chunk files (~600 chars, line-aware) → embed → vector store  
2. Query: hybrid BM25 + vector recall (~50) → lexical rerank (~10) → ± line context  
3. Code intel: `symbol_search`, `find_references`, `get_codebase_graph`

Default store is JSON under `.cozmo/index.json`. Chroma is optional (`vector_backend=chroma`).

### Subagents

The worker may call `run_subtask` with a goal. A child agent runs with read/search tools only, tighter step limits, and returns a **JSON summary** — not a full transcript — so the parent context stays small.

### Permissions

| Setting | Default |
|---------|---------|
| `allow_write` | `true` |
| `allow_shell` | `false` |

Tools cannot leave the workspace root (`WorkspaceGuard`).

---

## Configuration

**Precedence (highest wins):** CLI flags → `COZMO_*` env → cwd `.env` → `<repo>/.cozmo/config.json` → `~/.cozmo/config.json` → defaults.

| Path | Contents |
|------|----------|
| `~/.cozmo/config.json` | Global provider, model, key, budgets |
| `$XDG_CONFIG_HOME/cozmo/config.json` | Alternate global (if XDG set) |
| `<repo>/.cozmo/config.json` | Project overrides |
| `<repo>/.cozmo/index.json` | Vector index (or `chroma/`) |
| `<repo>/.cozmo/history/` | Session event JSONL |
| `<repo>/.cozmo/traces.jsonl` | Low-level LLM/tool telemetry |

```bash
cozmo config
cozmo config --show    # secrets masked
```

<details>
<summary>Example config</summary>

```json
{
  "provider": "openai",
  "model": "gpt-4o-mini",
  "worker_model": "gpt-4o-mini",
  "api_key": "sk-...",
  "allow_write": true,
  "allow_shell": false,
  "max_agent_steps": 8,
  "max_messages_before_compact": 30,
  "history_enabled": true,
  "vector_backend": "json",
  "trace_enabled": true
}
```

</details>

**Providers:** `openai` · `anthropic` · `openrouter` · `ollama` · `stub` (tests) · any OpenAI-compatible `base_url`.

---

## CLI reference

| Command | Description |
|---------|-------------|
| `cozmo` | Interactive agent (default) |
| `cozmo agent` | Same as default |
| `cozmo chat` | LLM only (no tools) |
| `cozmo setup` | Interactive configure |
| `cozmo config` | Paths / show JSON |
| `cozmo index` | Force re-index |
| `cozmo doctor` | Diagnose effective settings |
| `cozmo eval` | Golden cases against fixture / live |
| `cozmo --version` | Print version |

Common flags: `-w / --workdir`, `-p / --provider`, `--model`, `-m / --message`, `--no-index`.

---

## Development

```bash
git clone https://github.com/ShashwatXD/cozmo.git
cd cozmo
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

```bash
pip install -e ".[vector]"          # optional Chroma
cd website && npm install && npm run dev   # marketing site
```

---

## License

MIT. Named after [Anki Cozmo](https://www.digitaldreamlabs.com/pages/cozmo) — unaffiliated fan project.
