```
          :@@@@@@@@@@@@@=-
         @.              @@@%-=
      @@@                    @@
    @@+  =+  @@@@@@@@@@@+=+: .
   *@@=@@@@%@#%%#****##@@@@@@@*-  @@:=
  @ @@           *@@@@@@#++ ::=*#%  @-=
  @-                 +@@@%*   +*%%* .:-
   @@@@@@@@@@@@@@@@@  @  @@@#*+#%@*+ @.=
 = @%+@@@@@@@@@@@@@@@@@   @#%=+%@+@@@@.+=
 @ @#@@@@@@@@@@@@@@@@@@@ . %#+#@+ #+:@==.
@@ @*@@      @@@@@@@@@@@@-=@#%%@ @ +@+%
@% @=@@      @@@@@@@@@@@@.=@%%@@@@:%-@ %%
%@ @=@@      @@      @@@@ =@%#@*@%@*%@ @%
 - @=@@      @@      @@@@ +@%#@@@@@*%@ @@
   @+@#%%@@@@@@@@@@@@@@@@ +@%%%@.@-%@@@%
   @#@@@@@#@@%%%@%@@@@@@@ +@%%%#.=@#@#
   @@@%%%@%%%@%%@%@@@@%@@-=%%#@+% +=+
      @@@@@@@@@@@@@@@@@@   %%@@*%#
    =@    @@@@@@@@@@@@@  @@@@@*=
      @@@@@*+         -@@@@@@%#  %
      -=#@@@@@@@@@@@@@@@@@@-.  @@@

  ██████╗ ██████╗ ███████╗███╗   ███╗ ██████╗
 ██╔════╝██╔═══██╗╚══███╔╝████╗ ████║██╔═══██╗
 ██║     ██║   ██║  ███╔╝ ██╔████╔██║██║   ██║
 ██║     ██║   ██║ ███╔╝  ██║╚██╔╝██║██║   ██║
 ╚██████╗╚██████╔╝███████╗██║ ╚═╝ ██║╚██████╔╝
  ╚═════╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝ ╚═════╝
```

# Cozmo

CLI coding agent for your repo. One command. Provider setup. Then it indexes and works.

```bash
pip install cozmo-agent
cd ~/your-project
cozmo
```

Requires Python 3.11+.

---

## What happens

1. **Install** `cozmo-agent` (CLI binary is still `cozmo`)
2. Run `cozmo` in any project
3. First run: pick provider → paste key → pick model
4. Cozmo writes config, indexes the repo, opens an interactive session

Type questions at the prompt. Exit with `Ctrl+C` or an empty quit.

```bash
cozmo                # interactive agent (normal use)
cozmo setup          # change provider / model / key
```

---

## Config (where files land)

| Path | What |
|------|------|
| `~/.cozmo/config.json` | Global settings + API key (created on first run) |
| `$XDG_CONFIG_HOME/cozmo/config.json` | Same, if `XDG_CONFIG_HOME` is set |
| `<repo>/.cozmo/` | Index, traces, code graph |
| `<repo>/.cozmo/config.json` | Optional project overrides |

```bash
cozmo config         # print exact paths
cozmo config --show  # print JSON (key masked)
cozmo doctor         # effective settings
```

<details>
<summary>Example <code>~/.cozmo/config.json</code></summary>

```json
{
  "provider": "openai",
  "model": "gpt-4o-mini",
  "api_key": "sk-...",
  "allow_write": true,
  "allow_shell": false,
  "max_agent_steps": 8
}
```

Load order: CLI flags → `COZMO_*` env → cwd `.env` → project config → global config → defaults

</details>

---

## Commands

| | |
|--|--|
| `cozmo` | Interactive agent (default) |
| `cozmo setup` | Interactive reconfigure |
| `cozmo config` | Config paths |
| `cozmo index` | Force re-index |
| `cozmo chat` | LLM only, no tools |
| `cozmo doctor` | Diagnose settings |
| `cozmo -m "…"` | Optional one-shot (no REPL) |

---

## Stack (short)

- Owned ReAct loop (no LangChain)
- Tools: read/write, search, semantic search, git, symbols, graphs (shell gated)
- Hybrid RAG: BM25 + embeddings
- Providers: OpenAI, OpenRouter, Ollama, any OpenAI-compatible API

```
cozmo → config → index → AgentRunner
              LLM ↔ tools ↔ your files
```

---

## Dev

```bash
git clone https://github.com/ShashwatXD/cozmo.git
cd cozmo
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

---

**Fun fact:** Named after [Anki Cozmo](https://www.digitaldreamlabs.com/pages/cozmo), the robot pet I always wanted. Learning project, not affiliated. PyPI package is `cozmo-agent`.

MIT
