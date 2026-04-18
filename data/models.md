# Available Models

- Generated: 2026-04-18T22:59:27
- Source: `data/model_cost/*.csv` (text-chat models only)

このファイルは `scripts/generate_models_md.py` によって自動生成される参考資料です. 
config ファイル (`config.multi_turn.*.yml` / `config.single_turn.*.yml`) で `<provider>.model` に 
下表の `Model ID` を指定してください. 料金計算は `data/model_cost/*.csv` と `src/utils/cost_utils.py` が担当します.

## Config Providers

| `llm.type` | Pricing source | Notes |
|---|---|---|
| `openai` | OpenAI | `OPENAI_API_KEY` required |
| `google` | Google | `GOOGLE_API_KEY` required (Gemini API) |
| `vertexai` | Google | Uses Google Cloud ADC (`gcloud auth application-default login`); shares Google pricing |
| `anthropic` | Anthropic | `ANTHROPIC_API_KEY` required |
| `ollama` | (local) | No cost. Any model available on the local Ollama server |

`<provider>.pricing_mode` を config で明示すると `standard` 以外の料金モード (batch など) を使える. 
未指定時は `standard` が適用される.

---

## OpenAI

`llm.type: openai` — `<provider>.pricing_mode` 未指定時は `standard` の short_context 料金が適用される.
モデルによっては `standard-long_context` / `batch` / `flex` も選択可能 (下表 Extra 列参照).

| Model ID | Status | Group | Input ($/M) | Cached Input | Output ($/M) | Thinking | Extra modes | Notes |
|---|---|---|---:|---:|---:|---|---|---|
| `gpt-4o-mini` | legacy | legacy_general | $0.15 | $0.075 | $0.6 | no | — | Included because it still has a public model page with pricing, even though it is not on the current main pricing table. |
| `gpt-5.3-chat-latest` | current | specialized | $1.75 | $0.175 | $14 | unknown | — | Listed under specialized models / ChatGPT. |
| `gpt-5.3-codex` | current | specialized | $1.75 | $0.175 | $14 | unknown | — | Codex specialized model. |
| `gpt-5.4` | current | frontier | $2.5 | $0.25 | $15 | yes | standard-long $5.0/22.5 ; batch $1.25/7.5 | Regional processing endpoints are +10%. |
| `gpt-5.4-mini` | current | frontier | $0.75 | $0.075 | $4.5 | yes | batch $0.375/2.25 | Regional processing endpoints are +10%. |
| `gpt-5.4-nano` | current | frontier | $0.2 | $0.02 | $1.25 | yes | batch $0.1/0.625 | Regional processing endpoints are +10%. |
| `gpt-5.4-pro` | current | frontier | $30 | — | $180 | yes | standard-long $60.0/270.0 ; batch $15.0/90.0 | Regional processing endpoints are +10%. |

## Anthropic

`llm.type: anthropic` — `standard` 列を既定で使用. `anthropic.pricing_mode: batch` で batch 料金に切替可能.

| Model ID | Status | Family | Input ($/M) | Cache Read | Cache Write 5m | Cache Write 1h | Output ($/M) | Batch In | Batch Out | Thinking | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `claude-3-5-haiku` | current | Haiku | $0.8 | $0.08 | $1 | $1.6 | $4 | — | — | unknown | no batch pricing listed in excerpt |
| `claude-3-haiku` | current | Haiku | $0.25 | $0.03 | $0.3 | $0.5 | $1.25 | — | — | unknown |  |
| `claude-haiku-4-5` | current | Haiku | $1 | $0.1 | $1.25 | $2 | $5 | $0.5 | $2.5 | extended | Models overview indicates extended thinking but no adaptive thinking. |
| `claude-3-opus` | deprecated | Opus | $15 | $1.5 | $18.75 | $30 | $75 | — | — | unknown | Explicitly marked deprecated on pricing page. |
| `claude-opus-4` | current | Opus | $15 | $1.5 | $18.75 | $30 | $75 | $7.5 | $37.5 | unknown | deprecated |
| `claude-opus-4-1` | current | Opus | $15 | $1.5 | $18.75 | $30 | $75 | $7.5 | $37.5 | unknown |  |
| `claude-opus-4-5` | current | Opus | $5 | $0.5 | $6.25 | $10 | $25 | $2.5 | $12.5 | unknown |  |
| `claude-opus-4-6` | current | Opus | $5 | $0.5 | $6.25 | $10 | $25 | $2.5 | $12.5 | adaptive | Models overview indicates adaptive thinking; pricing page also lists batch pricing. |
| `claude-opus-4-7` | current | Opus | $5 | $0.5 | $6.25 | $10 | $25 | $2.5 | $12.5 | adaptive | Current latest Opus; pricing page plus models overview indicate adaptive thinking and 1M context. |
| `claude-3-7-sonnet` | deprecated | Sonnet | $3 | $0.3 | $3.75 | $6 | $15 | $1.5 | $7.5 | unknown | Explicitly marked deprecated on pricing page. |
| `claude-sonnet-4` | current | Sonnet | $3 | $0.3 | $3.75 | $6 | $15 | $1.5 | $7.5 | unknown |  |
| `claude-sonnet-4-5` | current | Sonnet | $3 | $0.3 | $3.75 | $6 | $15 | $1.5 | $7.5 | unknown |  |
| `claude-sonnet-4-6` | current | Sonnet | $3 | $0.3 | $3.75 | $6 | $15 | $1.5 | $7.5 | extended+adaptive | Models overview indicates both extended and adaptive thinking; pricing page also lists batch pricing. |

## Google (Gemini / Vertex AI)

`llm.type: google` or `vertexai` — どちらも同じ Google 価格表を参照する. 既定は `standard`.
同じ model_id でも `prompt_size_band` が複数ある場合は `all` (もしくは最短 band) を代表表示.

| Model ID | Status | Family | Input ($/M) | Cached Input | Output ($/M) | Thinking | Other modes | Notes |
|---|---|---|---:|---:|---:|---|---|---|
| `gemini-2.0-flash` | deprecated | Gemini 2.0 | $0.1 | $0.025 | $0.4 | unknown | batch | Deprecated; shutdown June 1, 2026. |
| `gemini-2.0-flash-lite` | deprecated | Gemini 2.0 | $0.075 | — | $0.3 | unknown | batch | Deprecated; shutdown June 1, 2026. |
| `gemini-2.5-flash` | current | Gemini 2.5 | $0.3 | $0.03 | $2.5 | yes | batch, flex, priority | Output pricing explicitly includes thinking tokens. |
| `gemini-2.5-flash-lite` | current | Gemini 2.5 | $0.1 | $0.01 | $0.4 | yes | batch, flex, priority | Output pricing explicitly includes thinking tokens. |
| `gemini-2.5-flash-lite-preview-09-2025` | preview | Gemini 2.5 | $0.1 | $0.01 | $0.4 | yes | batch | Preview variant. |
| `gemini-2.5-pro` | current | Gemini 2.5 | $1.25 | $0.125 | $10 | yes | batch, flex, priority | Output pricing explicitly includes thinking tokens. |
| `gemini-3-flash-preview` | preview | Gemini 3 | $0.5 | $0.05 | $3 | yes | batch, flex, priority | Output pricing explicitly includes thinking tokens. |
| `gemini-3.1-flash-lite-preview` | preview | Gemini 3.1 | $0.45 | $0.045 | $2.7 | yes | batch, flex | Output pricing explicitly includes thinking tokens. |
| `gemini-3.1-pro-preview` | preview | Gemini 3.1 | $2 | $0.2 | $12 | yes | batch, priority | Output pricing explicitly includes thinking tokens. |
