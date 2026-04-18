# aiwolf-jsai-agent

[日本語 README](/README.md)

LLM agent for the AIWolf Competition (Natural Language Division) — JSAI 2026 edition.

- **multi-turn / single-turn modes** switchable via config
- **Split LangChain**: use separate models and separate `llm_message_history` for talk-group (talk/whisper) vs action-group (vote/divine/guard/attack) requests
- **Prompt blocks**: reusable Jinja2 fragments under `prompts/{jp,en}/` composed with `{{ block('...') }}` (switch via config's `lang`), with an optional `markdown` / `xml` heading toggle
- **Cost tracking**: per-call USD cost written to `log/<game>/cost_summary.{json,md}` in real time

## Contents

- [Quick start](#quick-start)
- [Config files](#config-files)
- [Modes: multi-turn / single-turn](#modes-multi-turn--single-turn)
- [Split LangChain (talk group vs action group)](#split-langchain-talk-group-vs-action-group)
- [Prompt blocks](#prompt-blocks)
- [Cost tracking](#cost-tracking)
- [scripts/](#scripts)
- [Development](#development)

## Quick start

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
# 1) Clone the repo
git clone https://github.com/aiwolfdial/aiwolf-nlp-agent-llm.git aiwolf-jsai-agent
cd aiwolf-jsai-agent

# 2) Create .env from template (fill in API keys afterwards)
cp config/.env.example config/.env

# 3) Copy the 3-file config set (main + multi_turn child + single_turn child)
#    Swap .en for .jp if you prefer Japanese prompts (the main config's configs: block already points to the .en children)
cp config/config.main.en.yml.example         config/config.main.en.yml
cp config/config.multi_turn.en.yml.example   config/config.multi_turn.en.yml
cp config/config.single_turn.en.yml.example  config/config.single_turn.en.yml

# 4) Install dependencies
uv sync
```

After setting API keys (`OPENAI_API_KEY` / `GOOGLE_API_KEY` / `CLAUDE_API_KEY`, whichever you use) in `config/.env`, run:

```bash
# Launch agents (defaults to ./config/config.main.jp.yml; the matching child config is auto-merged based on `mode`)
uv run python src/main.py

# Use -c to switch to English prompts
uv run python src/main.py -c ./config/config.main.en.yml

# ...or to run multiple configs in parallel
uv run python src/main.py -c './config/*.main.*.yml'
```

> Without `uv`: `python -m venv .venv && source .venv/bin/activate && pip install -e .`, then `python src/main.py`.

## Config files

Configuration is split across three files. The main config references the child configs.

| File | Role |
|---|---|
| `config/config.main.{jp,en}.yml` | Mode (`mode: multi_turn` / `single_turn`), WebSocket, agent, log settings |
| `config/config.multi_turn.{jp,en}.yml` | LLM settings and prompts for multi-turn mode |
| `config/config.single_turn.{jp,en}.yml` | LLM settings and prompts for single-turn mode |

The main config's `configs:` block maps each mode to a child config path. At load time the matching child config is merged on top of the main config (child wins on key collision). Keeping the `.jp` / `.en` language suffix lets both sets coexist in the same directory.

```yaml
# config.main.en.yml (excerpt)
mode: multi_turn
configs:
  multi_turn: ./config.multi_turn.en.yml
  single_turn: ./config.single_turn.en.yml
```

## Modes: multi-turn / single-turn

| Mode | Behavior | LLM input |
|---|---|---|
| **multi-turn** | Conversation history (`llm_message_history`) is kept by LangChain | Full history sent on every request |
| **single-turn** | No `llm_message_history`; full context is embedded into every prompt | Single `HumanMessage` per call |

### Single-turn specifics
- `initialize` / `daily_initialize` / `daily_finish` are **not sent to the LLM**. Their payloads are snapshotted inside the agent (`day_events`).
- On talk / whisper / divine / etc., `day_events` and the full `talk_history` / `whisper_history` are rendered directly into the prompt body.

Switch modes simply by changing the `mode` field in the main config (`config.main.jp.yml` / `config.main.en.yml`).

## Split LangChain (talk group vs action group)

Set `llm.separate_langchain: true` in a child config (`config.multi_turn.{jp,en}.yml` or `config.single_turn.{jp,en}.yml`) to use separate LangChain instances and separate `llm_message_history` per request group.

```yaml
llm:
  type: google               # single-model fallback when separate_langchain=false
  sleep_time: 3
  separate_langchain: true
  talk:
    type: google             # used for talk / whisper
  action:
    type: claude             # used for vote / divine / guard / attack
```

- **Shared requests** (`initialize` / `daily_initialize` / `daily_finish`) are sent to both models so both histories stay in sync.
- **Talk-group** requests (`talk` / `whisper`) only update `llm.talk`'s history.
- **Action-group** requests (`vote` / `divine` / `guard` / `attack`) only update `llm.action`'s history.
- When `false`, the single `llm.type` model/history is used (legacy behavior).

## Prompt blocks

Five reusable Jinja2 blocks live under `prompts/jp/` and `prompts/en/` respectively. The `lang: jp` / `lang: en` field in the main config selects which directory to load. Reference the blocks from `prompt.<request>` via `{{ block('<name>') }}`.

| Block | Purpose | Key variables |
|---|---|---|
| `identity.jinja` | Name / role / profile | `info.agent`, `role.value`, `info.profile` |
| `history.jinja` | Talk/whisper history loop (switched via `history_source` / `history_start`) | `talk_history`, `whisper_history` |
| `event.jinja` | Daily-event list (prefers `day_events`, falls back to `info`) | `day_events`, `info` |
| `instruction.jinja` | Minimal per-request instructions | `request_key` |
| `constraints.jinja` | Output format + length caps (pulled from server `setting`) | `request_key`, `setting` |

Usage:

```jinja
{% set history_source = talk_history %}
{% set history_start = sent_talk_count %}
{{ block('history') }}
{{ block('instruction') }}
{{ block('constraints') }}
```

`block('<name>')` renders `prompts/<lang>/<name>.jinja` with the caller's context (same result as a plain `{% include %}`). When the **heading toggle** (see below) is on, it also prepends a heading line.

Both jp and en blocks + configs ship out of the box. To add another language, create `prompts/<lang>/` with the same 5 files and `config/config.<mode>.<lang>.yml.example`, then set `lang: <lang>` in the main config and point its `configs:` block at the `<lang>`-suffixed children.

### Heading toggle (`headings`)

The `headings` section in the main config controls whether each block is prefixed with a heading. This is useful when you want the LLM to see explicit block boundaries.

```yaml
# config.main.en.yml (excerpt)
headings:
  enabled: false     # true to prepend headings
  style: markdown    # markdown | xml
```

| style | Output (lang=en) | Output (lang=jp) |
|---|---|---|
| `markdown` | `### history` followed by the body | `### 履歴` followed by the body |
| `xml` | `<history>` … body … `</history>` | `<履歴>` … body … `</履歴>` |

- The heading text defaults to the **block's filename stem** (e.g. `history` for `history.jinja`). Japanese labels are defined in `prompts/jp/_labels.yml`; any unlabeled block falls back to the stem.
- `prompts/en/` ships no `_labels.yml`, so English headings are always the filename stem.
- To add a new block `foo.jinja`, simply drop the jinja files in place; optionally add one line (`foo: ○○`) to `prompts/jp/_labels.yml` for a Japanese label. No Python changes needed.
- With `enabled: false` (the default) blocks are concatenated with no headings, preserving the previous behavior.

## Cost tracking

Each LLM call has its `AIMessage.usage_metadata` extracted and priced against `data/model_cost/*.csv`. Results are appended to `log/<game>/cost_summary.json` with `fcntl` locking in real time, and `cost_summary.md` is rendered on game finish.

### Output layout

Under `log/<YYYYMMDDHHmmssSSS>/` (same naming as `agent_logger`):

```
log/20260418033529578/
  kanolab1.log
  kanolab2.log
  ...
  cost_summary.json    # rewritten per LLM call
  cost_summary.md      # generated at finish
```

### What is counted

- Tokens are split into **input / cached_input / output / thinking** (OpenAI reasoning, Anthropic extended thinking, Google cached content).
- Multi-turn cumulative input (full-history `input_tokens`) is fully included.
- Pricing tables: `data/model_cost/openai.csv`, `anthropic.csv`, `google.csv`.
- Select `pricing_mode` per provider via `<provider>.pricing_mode` in config (default `standard`; `batch` etc. also supported).
- `ollama` is free (zero cost). Models missing from the CSV are logged with a warning and flagged `unknown_pricing`.

### Updating the price table

After adding rows to a provider CSV, regenerate the reference table with `uv run python scripts/generate_models_md.py` to refresh `data/models.md`.

## scripts/

| Script | Purpose |
|---|---|
| `scripts/preview_prompt.py` | Read `data/sample_packet.yml` and render all requests for 4 targets (jp × {multi_turn, single_turn} + en × {multi_turn, single_turn}) into `preview.md` |
| `scripts/generate_models_md.py` | Build `data/models.md` (the human-readable model / pricing reference) from `data/model_cost/*.csv` |

Run:

```bash
uv run python scripts/preview_prompt.py       # regenerate preview.md
uv run python scripts/generate_models_md.py   # regenerate data/models.md
```

## Development

```bash
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pyright          # type check (strict)
```

### Directory layout

```
aiwolf-jsai-agent/
├── config/                       # Config examples
├── data/
│   ├── model_cost/               # Per-provider pricing CSVs
│   ├── models.md                 # Generated model reference
│   └── sample_packet.yml         # Sample for preview
├── prompts/
│   ├── jp/                       # Japanese Jinja2 blocks (5 files + _labels.yml)
│   └── en/                       # English Jinja2 blocks (5 files)
├── scripts/                      # Utility scripts
├── src/
│   ├── agent/                    # Agent base + role subclasses
│   ├── utils/
│   │   ├── agent_logger.py       # Per-game log output
│   │   ├── cost_utils.py         # Pricing table + cost math
│   │   ├── cost_logger.py        # cost_summary.{json,md} writer
│   │   └── ...
│   ├── main.py
│   └── starter.py
└── preview.md                    # (generated)
```

## References

- [aiwolf-nlp-agent](https://github.com/aiwolfdial/aiwolf-nlp-agent) — reference implementation (protocol details, etc.)
- [aiwolf-nlp-server](https://github.com/aiwolfdial/aiwolf-nlp-server) — game server
