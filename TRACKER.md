# Cozmo tracker

Checklist of planned work. Mark an item `[x]` when it is complete.

Related: [`PAINPOINTS.md`](./PAINPOINTS.md)

**Design rule:** Take useful primitives from peer agents. Do not treat peer feature lists as the success criteria. Keep an owned ReAct loop. Do not adopt LangChain as the core runtime.

---

## Already shipped (baseline)

- [x] Owned ReAct agent loop (`AgentRunner`)
- [x] Guardrails, budgets, and compaction
- [x] Hybrid BM25 and vector RAG with `semantic_search`
- [x] `run_subtask` subagent
- [x] Workspace sandbox (`WorkspaceGuard`)
- [x] Session history JSONL and traces under `.cozmo/`
- [x] Multi-provider setup (BYOK)
- [x] REPL basics (`/help`, `/clear`, `/compact`, `/history`, `/setup`, and related)

---

## Phase 1: Context foundation

Prerequisite for a strong context and retrieval layer.

- [x] **H1** Ripgrep-backed `search_repo` (respect `.gitignore`; return `path:line:snippet`; cap hit count)
- [x] **H2** Ranged `read_file` (`start_line` / `end_line`, or around-line plus window)
- [x] **H3** Tool-result shaping and hard size caps (the harness controls bytes re-injected into the model)
- [x] **H4** Finder policy: exact identifiers use search; conceptual queries use semantic search; always read before answering
- [x] **H5** Incremental and resilient RAG indexing (do not fail the session on embed errors such as HTTP 402; skip unchanged files)
- [x] **H6** Ignore hygiene (`.gitignore` plus skip lists for both search and indexing)
- [x] **H7** Subagent compact evidence packs (paths, claims, and open questions only)

**Exit criteria:** Locate code, perform a windowed read, and answer with citations, without loading many full files into the prompt.

---

## Phase 2: Trust and visibility

- [x] **M1** Plan mode (explore, then approve, then edit)
- [x] **M2** Mutating-tool permissions for **write and shell**: preview, then Allow once / Always allow / Deny, with project-level persistence
- [x] **M4** Live context economy UI (approximate token usage and compaction status)

**Exit criteria:** Mutating tools do not run without explicit permission UX. Context budget is visible in the REPL.

---

## Phase 3: Modes and sessions

### Sessions

- [x] Show sessions (`/sessions`: id, time, model, preview)
- [x] Continue session (`/continue` or `/resume <id>`)
- [x] Continue last session (no id selects the most recent session in the workdir)
- [x] Export session (`/export md|json`)
- [ ] Name or pin session (optional titles) — skipped for now

### Modes

- [x] **Agent mode** (default: full tools subject to permissions).
- [x] **Ask mode**: read and search only; no write, shell, or mutating git (`/ask`, `cozmo --ask`)
- [x] **Plan mode**: same capability as M1; expose via `/plan` and a CLI flag
- [ ] **Review mode** (later): diff or PR oriented, critique first

### Permissions UX

- [x] `write_file`: show diff preview, then Allow once / Always allow / Deny
- [x] `run_shell`: show full command and cwd, then Allow once / Always allow pattern / Deny
- [x] Persist allow and deny rules under `.cozmo/permissions.json` (or equivalent)
- [ ] Gated git mutations (later): same permission pattern

### REPL command targets

- [x] `/sessions`
- [x] `/continue [id]`
- [x] `/export md|json`
- [x] `/ask`
- [x] `/plan`
- [x] `/agent`
- [x] `/compact`
- [x] `/clear`

**Exit criteria:** Move from ask to plan to approved agent work; leave and resume later with `/sessions` and `/continue` without losing the investigation trail.

---

## Phase 4: Amplifiers

Ship items that support the thesis. Skip cosmetic work that does not improve context or trust.

- [ ] **M3** Project memory file (`COZMO.md`)
- [ ] **M5** Structure-aware chunking
- [ ] **M6** Check loop (`run_checks` returning `file:line` diagnostics)
- [ ] **M7** `@path` and `@path:range` mentions
- [x] **M8** Session list, continue, and export (tracked primarily under Phase 3; pin/name deferred)
- [ ] **M9** Richer read-only git tools (blame, log); mutations remain gated
- [ ] **M10** Optional MCP hooks
- [ ] **M11** Web search tool (`web_search`): query the public web for docs, errors, and APIs; return titled snippets plus URLs; cap result size; optional or keyed provider
- [ ] **M12** Web fetch tool (`web_fetch`): fetch a URL and return truncated readable text for follow-up after search; same size caps as other tools

---

## Phase 5: Leapfrog features (after foundation)

Differentiation after Phase 1 is solid. Not a peer parity list.

- [ ] **Evidence graph**: session graph linking file, claim, and open question; subsequent tools prefer expanding that graph
- [ ] **Confidence-gated answers**: weak citations trigger further investigation or an explicit refusal to invent
- [ ] **Dual-index fusion**: ripgrep and semantic hits as one ranked evidence stream, enforced by the harness
- [ ] **Context compiler**: before each LLM call, pin the goal and citations; compress prior tool output to fingerprints (example: `agent.py:40-90`)
- [ ] **Eval gate**: golden find and read tasks; harness regressions fail CI

**Exit criteria:** Model input is compiled evidence under harness control, with evals that detect retrieval and context regressions.

---

## Key features by milestone

| After | Capability delivered |
|-------|----------------------|
| **Phase 1** | Fast exact find, narrow reads, reliable RAG. Core repository intelligence. |
| **Phase 2** | Permission prompts for write and shell. Visible context budget. |
| **Phase 3** | Ask, Plan, and Agent modes. List and continue sessions. |
| **Phase 4** | Project memory, checks, path ranges, richer git, web search and fetch. Daily-driver polish. |
| **Phase 5** | Context compiler, dual-index fusion, confidence gating, eval gates. |

### Target user flows

- [ ] "Where is X?": ripgrep hits, ranged read, answer with `path:line`
- [ ] "How does Y work?": semantic search, ranged reads, cited explanation
- [x] Ask mode explains code without write or shell risk
- [x] Plan, approve, then agent applies patches under permission prompts
- [x] Shell commands always display and require approval unless an allow pattern matches
- [x] `/sessions` and `/continue` resume a prior investigation
- [x] Context meter reflects usage; compaction preserves task continuity
- [ ] Weak evidence causes further investigation rather than invented paths
- [ ] Docs or unfamiliar API questions can use `web_search`, then `web_fetch` on a chosen URL, with capped tool output

---

## Out of scope

Do not treat the following as project success criteria:

- [ ] Cloning another agent role fleet or TUI checklist
- [ ] Rewriting the core runtime onto LangChain
- [ ] Shipping LSP, IDE, or cloud surfaces before the context compiler
- [ ] Restoring incomplete symbol graphs only for marketing bullets

---

## Suggested order

```text
Phase 1 (H1 through H7)
  then Phase 2 (M1, M2, M4)
  then Phase 3 (modes and sessions)
  then Phase 5 bets in parallel with Phase 4 amplifiers
```

---

## Principle

Harness owns the context budget. The model owns judgment inside that budget. Peer agents inspire primitives. Cozmo sets its own ceiling.
