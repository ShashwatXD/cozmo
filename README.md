# Cozmo

Production-style **CLI coding agent** in Python: provider-swappable LLM gateway, ReAct tool loop, sandboxed tools, conversation memory, RAG, retries/cost tracking, JSONL traces, and a golden eval suite.

Inspired by systems like Cursor - focused on the **agent runtime**, not an IDE UI.

## Features

| Area | What shipped |
|------|----------------|
| **LLM gateway** | Unified `LLMClient` port - **OpenAI / Ollama / stub** via config or `--provider` |
| **Tool calling** | JSON schemas, registry, executor, error feedback to the model |
| **ReAct agent** | Plan/act/observe loop with max-steps guard |
| **Sandbox** | Workspace path allowlist; write/shell gated by flags |
| **Tools** | `read_file`, `write_file`, `search_repo`, `semantic_search`, `git_status`, `git_diff`, `run_shell` |
| **Memory** | Sliding-window conversation memory (multi-turn REPL) |
| **RAG** | Chunk → embed → cosine retrieval; embedders: **hash / OpenAI / Ollama** |
| **Reliability** | Timeouts, exponential backoff retries (`tenacity`) |
| **Cost** | Token usage + estimated USD by model |
| **Observability** | JSONL traces (LLM latency/tokens, tool calls) under `.cozmo/traces.jsonl` |
| **Evaluation** | `cozmo eval` golden tasks (stub CI-safe; `--live` optional) |
| **Architecture** | Clean layers: `cli` → `app` → `domain` ← `infra` (ports & adapters) |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# edit .env - ollama or openai

# optional: local model
ollama serve
# ollama pull qwen2.5:3b

cozmo index -w tests/fixtures/tiny_repo
cozmo agent -w tests/fixtures/tiny_repo -m "Find the bug in math_utils.py"
cozmo eval -w tests/fixtures/tiny_repo
pytest -q
```

### Provider switch

```bash
# .env
COZMO_PROVIDER=openai
COZMO_MODEL=gpt-4o-mini
COZMO_OPENAI_API_KEY=sk-...

# or CLI override (no .env edit)
cozmo agent -p openai --model gpt-4o-mini -w . -m "Explain src/cozmo/app/agent.py"
cozmo chat -p ollama --model qwen2.5:3b -m "Hello"
```

### RAG with real embeddings

```bash
# OpenAI embeddings
COZMO_EMBEDDER=openai COZMO_EMBEDDING_MODEL=text-embedding-3-small cozmo index -w .

# or Ollama: ollama pull nomic-embed-text
COZMO_EMBEDDER=ollama COZMO_EMBEDDING_MODEL=nomic-embed-text cozmo index -w .
```

## Architecture

```
CLI (typer)
  → ChatUseCase / AgentRunner          # application / orchestration
      → LLMClient (Protocol)           # port
          → OpenAICompatible / Stub    # adapters (+ RetryingLLMClient)
      → ToolRegistry / ToolExecutor
      → ConversationMemory
      → VectorStore + Embedder         # RAG
      → Tracer                         # JSONL observability
```

Dependency rule: **domain never imports openai**. Infra implements ports.

## Commands

| Command | Purpose |
|---------|---------|
| `cozmo chat` | Streaming chat (+ REPL) |
| `cozmo agent` | Coding agent with tools |
| `cozmo index` | Build `.cozmo/index.json` |
| `cozmo eval` | Golden regression suite |
| `cozmo --version` | Version |

## Resume / skills demonstrated

Use these as bullet inspiration (edit for honesty):

- Built a **provider-agnostic LLM gateway** (OpenAI + Ollama) with **retries, timeouts, and cost estimation**
- Implemented a **ReAct agent runtime** with **function/tool calling**, schema validation, and tool error recovery
- Designed **clean architecture** (ports & adapters / dependency inversion) for swappable models and tools
- Added **RAG**: chunking, pluggable embeddings, vector similarity search, semantic retrieval tool
- Built **conversation memory** with context-window pruning for multi-turn sessions
- Enforced **tool sandboxing** (workspace boundaries, capability flags for write/shell)
- Added **observability** (structured JSONL traces: latency, tokens, tool calls) and **`cozmo eval` golden-task regression**
- Delivered a production-minded **Python CLI** (Typer, Pydantic Settings, pytest)

## Project layout

```
src/cozmo/
  domain/     # messages, tools, ports, memory, cost, rag types
  app/        # ChatUseCase, AgentRunner, eval_runner
  infra/      # LLM adapters, tools, RAG, telemetry
  cli/        # Typer entrypoint
  prompts/    # versioned system prompts
tests/
docs/
```

