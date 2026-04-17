# aiwolf-jsai-agent

[README in English](/README.en.md)

人狼知能コンテスト（自然言語部門）向けの LLM エージェント実装（JSAI 2026 版）。

- **multi-turn / single-turn モード** を config で切替
- **LangChain 分離**: 発話系（talk/whisper）とアクション系（vote/divine/guard/attack）で別モデル・別 `llm_message_history` を使用可能
- **プロンプトブロック**: `prompts/{jp,en}/*.jinja` を `{% include %}` で再利用（日英切替は config の `lang`）
- **コストトレース**: 呼び出しごとに `log/<game>/cost_summary.{json,md}` をリアルタイム生成

## 目次

- [クイックスタート](#クイックスタート)
- [設定ファイル構成](#設定ファイル構成)
- [モード: multi-turn / single-turn](#モード-multi-turn--single-turn)
- [LangChain 分離 (発話系 / アクション系)](#langchain-分離-発話系--アクション系)
- [プロンプトブロック](#プロンプトブロック)
- [コストトレース](#コストトレース)
- [scripts/](#scripts)
- [開発](#開発)

## クイックスタート

Python 3.11 以上 + [uv](https://docs.astral.sh/uv/) を推奨します。

```bash
# 1) リポジトリ取得
git clone https://github.com/aiwolfdial/aiwolf-nlp-agent-llm.git aiwolf-jsai-agent
cd aiwolf-jsai-agent

# 2) API キー用の .env をテンプレから作成 (編集は後述)
cp config/.env.example config/.env

# 3) 3分割configを example からコピー (メイン + multi_turn子 + single_turn子)
#    英語プロンプトを使うなら .jp を .en に置き換える
cp config/config.main.jp.yml.example         config/config.main.yml
cp config/config.multi_turn.jp.yml.example   config/config.multi_turn.yml
cp config/config.single_turn.jp.yml.example  config/config.single_turn.yml

# 4) 依存インストール
uv sync
```

`config/.env` に API キー（`OPENAI_API_KEY` / `GOOGLE_API_KEY` / `CLAUDE_API_KEY` のうち使うもの）を設定後、実行:

```bash
# エージェントを起動 (既定で ./config/config.main.yml を読み, モードに応じて子configを自動マージ)
uv run python src/main.py

# 明示的に指定する場合 / 複数 config を並列実行する場合のみ -c を使う
uv run python src/main.py -c ./config/my_config.main.yml
uv run python src/main.py -c './config/*.main.yml'
```

> `uv` を使わない場合: `python -m venv .venv && source .venv/bin/activate && pip install -e .` の後、`python src/main.py` で実行。

## 設定ファイル構成

設定は 3 ファイルに分割されています。メイン config が子 config を参照する構造です。

| ファイル | 役割 |
|---|---|
| `config/config.main.yml` | モード (`mode: multi_turn` / `single_turn`)、WebSocket、agent、log 等の共通設定 |
| `config/config.multi_turn.yml` | multi-turn 時の LLM 設定・プロンプト定義 |
| `config/config.single_turn.yml` | single-turn 時の LLM 設定・プロンプト定義 |

メイン config の `configs:` セクションで子 config のパスを指定。`mode` に応じて対応する子 config がロード時にマージされます（キー衝突時は子 config 優先）。

```yaml
# config.main.yml (抜粋)
mode: multi_turn
configs:
  multi_turn: ./config.multi_turn.yml
  single_turn: ./config.single_turn.yml
```

## モード: multi-turn / single-turn

| モード | 特徴 | LLM への入力 |
|---|---|---|
| **multi-turn** | 会話履歴 (`llm_message_history`) を LangChain で保持 | 毎リクエスト、履歴全体を送信 |
| **single-turn** | `llm_message_history` を使わず毎回フルコンテキストを埋め込み | `HumanMessage` 単発のみ |

### single-turn の動作
- `initialize` / `daily_initialize` / `daily_finish` は **LLM に送信せず**、agent 内部 (`day_events`) にスナップショット保持
- talk / whisper / divine 等の各リクエストで、`day_events` とフル `talk_history` / `whisper_history` をプロンプト本文に埋め込み

モード切替は `config.main.yml` の `mode` フィールドを変更するだけです。

## LangChain 分離 (発話系 / アクション系)

`config.multi_turn.yml` または `config.single_turn.yml` の `llm.separate_langchain` を `true` にすると、リクエスト種別ごとに LangChain インスタンスと `llm_message_history` を分離できます。

```yaml
llm:
  type: google               # separate_langchain=false 時の単一モデル
  sleep_time: 3
  separate_langchain: true   # ← true で分離
  talk:
    type: google             # talk / whisper はこちら
  action:
    type: claude             # vote / divine / guard / attack はこちら
```

- **共通リクエスト** (`initialize` / `daily_initialize` / `daily_finish`) は両方のモデルに送信され、両系統の履歴に情報共有される
- **発話系** (`talk` / `whisper`) は `llm.talk` の履歴のみ更新
- **アクション系** (`vote` / `divine` / `guard` / `attack`) は `llm.action` の履歴のみ更新
- `false` の場合は `llm.type` で指定した単一モデル・単一履歴で従来通り動作

## プロンプトブロック

`prompts/jp/` と `prompts/en/` の配下にそれぞれ 5 つの再利用可能 Jinja2 ブロックがあります。`config.main.yml` の `lang: jp` / `lang: en` で参照先を切替えます。config の `prompt.<request>` から `{% include %}` で参照してください。

| ブロック | 役割 | 主な変数 |
|---|---|---|
| `identity.jinja` | 名前・役職・プロフィール | `info.agent` / `role.value` / `info.profile` |
| `history.jinja` | 発言履歴ループ（`history_source` / `history_start` で切替） | `talk_history` / `whisper_history` |
| `event.jinja` | 日次イベント一覧（`day_events` 優先、無ければ `info`） | `day_events` / `info` |
| `instruction.jinja` | リクエスト別の最低限の指示文 | `request_key` |
| `constraints.jinja` | 出力形式と文字数制限（サーバ `setting` 参照） | `request_key` / `setting` |

使用例:

```jinja
{% set history_source = talk_history %}
{% set history_start = sent_talk_count %}
{% include 'history.jinja' %}
{% include 'instruction.jinja' %}
{% include 'constraints.jinja' %}
```

jp / en の両言語のブロックと config が揃っています。新しい言語を追加する場合は `prompts/<lang>/` と `config/config.<mode>.<lang>.yml.example` の両方を用意し、`config.main.yml` の `lang` を書き換えてください。

## コストトレース

LLM 呼び出しごとに、`AIMessage.usage_metadata` からトークン使用量を抽出し `data/model_cost/*.csv` を参照して USD 換算。`log/<game>/cost_summary.json` に fcntl ロック付きでリアルタイム追記、ゲーム終了時に `cost_summary.md` を生成します。

### 出力先

`log/<YYYYMMDDHHmmssSSS>/` 配下（`agent_logger` と同じ命名規則）に以下が並びます。

```
log/20260418033529578/
  kanolab1.log
  kanolab2.log
  ...
  cost_summary.json    # 呼び出しごとに上書き
  cost_summary.md      # finish時に生成
```

### 集計対象

- **input / cached_input / output / thinking** トークンを分離して集計（OpenAI reasoning / Anthropic extended thinking / Google cached content に対応）
- **multi-turn** の累積入力（履歴込みの `input_tokens`）も漏れなく計上
- **プロバイダ別料金表**: `data/model_cost/openai.csv` / `anthropic.csv` / `google.csv`
- 各プロバイダの `pricing_mode`（`standard` / `batch` 等）を config の `<provider>.pricing_mode` で切替可能（既定 `standard`）
- `ollama` は無料計上、`models.csv` に無いモデルは警告 + `unknown_pricing` フラグ付与

### 料金表の更新

モデル追加時は該当プロバイダの CSV に行を追加し、`uv run python scripts/generate_models_md.py` で `data/models.md` を再生成してください。

## scripts/

| Script | 説明 |
|---|---|
| `scripts/preview_prompt.py` | `data/sample_packet.yml` を読み、jp × {multi_turn, single_turn} + en × {multi_turn, single_turn} の計4ターゲット全リクエストをレンダリングして `preview.md` に上書き出力 |
| `scripts/generate_models_md.py` | `data/model_cost/*.csv` から `data/models.md`（使用可能モデル一覧と料金）を生成 |

実行:

```bash
uv run python scripts/preview_prompt.py       # preview.md を再生成
uv run python scripts/generate_models_md.py   # data/models.md を再生成
```

## 開発

```bash
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pyright          # 型チェック (strict)
```

### ディレクトリ構成

```
aiwolf-jsai-agent/
├── config/                       # 設定ファイル (example)
├── data/
│   ├── model_cost/               # プロバイダ別料金 CSV
│   ├── models.md                 # 自動生成されたモデル一覧
│   └── sample_packet.yml         # preview 用サンプル
├── prompts/
│   ├── jp/                       # Jinja2 ブロック 日本語 (5ファイル)
│   └── en/                       # Jinja2 ブロック 英語 (5ファイル)
├── scripts/                      # ユーティリティスクリプト
├── src/
│   ├── agent/                    # Agent 実装 (役職別含む)
│   ├── utils/
│   │   ├── agent_logger.py       # ゲーム別ログ出力
│   │   ├── cost_utils.py         # 料金テーブル読込・コスト計算
│   │   ├── cost_logger.py        # cost_summary.{json,md} 書込
│   │   └── ...
│   ├── main.py
│   └── starter.py
└── preview.md                    # (自動生成)
```

## 参考

- [aiwolf-nlp-agent](https://github.com/aiwolfdial/aiwolf-nlp-agent) — リファレンス実装（プロトコル詳細など）
- [aiwolf-nlp-server](https://github.com/aiwolfdial/aiwolf-nlp-server) — ゲームサーバ
