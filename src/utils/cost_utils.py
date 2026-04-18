"""Cost utilities: pricing table and per-call cost calculation.

料金テーブルのロードおよび LLM 呼び出しごとのコスト計算ユーティリティ.

usage_metadata (LangChain BaseMessage.usage_metadata) や response_metadata から
input / cached_input / output / thinking の各トークン数を抽出し,
USD コストに換算する.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# config の provider キー (llm.type) から models.csv の provider 名へのマッピング.
# vertexai は Google 価格表を参照する. ollama はローカル実行で無課金扱い.
_PROVIDER_ALIAS: dict[str, str] = {
    "openai": "OpenAI",
    "google": "Google",
    "vertexai": "Google",
    "anthropic": "Anthropic",
}
_FREE_PROVIDERS: set[str] = {"ollama"}

_MILLION = 1_000_000


@dataclass(frozen=True)
class PricingRow:
    """Pricing entry for a (provider, model_id, pricing_mode) triple.

    料金表の1行を表す. 価格は "USD per 1M tokens" 単位で保持する.
    """

    provider: str
    model_id: str
    pricing_mode: str
    input_price_usd: float | None
    cached_input_price_usd: float | None
    output_price_usd: float | None
    thinking_support: str
    status: str
    notes: str = ""


@dataclass
class CostRecord:
    """Token usage and cost for a single LLM call.

    1回のLLM呼び出し分のトークン使用量とコスト.
    """

    provider: str
    model_id: str
    pricing_mode: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cost_usd: float = 0.0
    unknown_pricing: bool = False
    details: dict[str, Any] = field(default_factory=dict)


# OpenAI / Google はひとつの model_id に対し context_band / prompt_size_band で
# 複数行を持つ. デフォルトは短コンテキスト側を採用する.
_OPENAI_DEFAULT_CONTEXT_BAND = "short_context"
# prompt_size_band の優先順位. "all" があれば採用, なければ "<=200K" 相当.
_GOOGLE_PROMPT_BAND_PRIORITY = ("all", "<=200K", "<=128K", "<=32K")
# 非チャット用途のモデルは Agent のコスト計算からは除外する.
# OpenAI: product_group 列, Google: family 列で判定する.
_OPENAI_EXCLUDED_PRODUCT_GROUPS = {
    "image_generation",
    "realtime_and_audio_generation",
    "transcription",
    "video_generation",
    "fine_tuning",
}
_GOOGLE_EXCLUDED_FAMILIES = {
    "Embeddings",
    "Imagen 4",
    "Lyria 3",
    "Robotics-ER",
    "Veo 2",
    "Veo 3",
    "Veo 3.1",
}


def load_pricing_table(
    root_dir: Path,
) -> dict[tuple[str, str, str], PricingRow]:
    """Load all provider pricing CSVs under `root_dir` into a unified table.

    `data/model_cost/<provider>.csv` を一括で読み込み,
    (provider_display, model_id, pricing_mode) -> PricingRow の辞書にまとめる.

    各プロバイダのスキーマが異なるため, それぞれ専用ローダーで正規化する.
    openai の context_band / google の prompt_size_band は単一スライスを代表として採用する
    (デフォルトは OpenAI=short_context, Google=all or 最短 band).

    Args:
        root_dir (Path): Directory containing provider CSVs / プロバイダ別CSVの格納先

    Returns:
        dict[tuple[str, str, str], PricingRow]: Unified pricing table / 統合料金テーブル
    """
    table: dict[tuple[str, str, str], PricingRow] = {}
    openai_csv = root_dir / "openai.csv"
    anthropic_csv = root_dir / "anthropic.csv"
    google_csv = root_dir / "google.csv"
    if openai_csv.exists():
        table.update(_load_openai_pricing(openai_csv))
    if anthropic_csv.exists():
        table.update(_load_anthropic_pricing(anthropic_csv))
    if google_csv.exists():
        table.update(_load_google_pricing(google_csv))
    return table


def _load_openai_pricing(csv_path: Path) -> dict[tuple[str, str, str], PricingRow]:
    """Load data/model_cost/openai.csv.

    同じ model_id に対し context_band が複数ある場合, 短コンテキスト (short_context) を
    デフォルトとして採用する. 非チャット用途 (embeddings / image / tts など) はスキップ.
    """
    table: dict[tuple[str, str, str], PricingRow] = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            product_group = (row.get("product_group") or "").strip()
            if product_group in _OPENAI_EXCLUDED_PRODUCT_GROUPS:
                continue
            model_id = (row.get("model_id") or "").strip()
            pricing_mode = (row.get("pricing_mode") or "").strip()
            context_band = (row.get("context_band") or "").strip()
            if not model_id or not pricing_mode:
                continue
            if context_band and context_band != _OPENAI_DEFAULT_CONTEXT_BAND:
                # 非デフォルトバンドはサフィックス付きで別キーに保持 (例: standard-long_context)
                effective_mode = f"{pricing_mode}-{context_band}"
            else:
                effective_mode = pricing_mode
            key = ("OpenAI", model_id, effective_mode)
            table[key] = PricingRow(
                provider="OpenAI",
                model_id=model_id,
                pricing_mode=effective_mode,
                input_price_usd=_to_float(row.get("input_price_usd")),
                cached_input_price_usd=_to_float(row.get("cached_input_price_usd")),
                output_price_usd=_to_float(row.get("output_price_usd")),
                thinking_support=(row.get("thinking_support") or "unknown").strip(),
                status=(row.get("status") or "").strip(),
                notes=(row.get("notes") or "").strip(),
            )
    return table


def _load_anthropic_pricing(csv_path: Path) -> dict[tuple[str, str, str], PricingRow]:
    """Load data/model_cost/anthropic.csv.

    1行1モデル. standard / batch の2モードをそれぞれ別キーで保存する.
    cache_read を cached_input_price_usd として採用する (cache_write は未採用).
    """
    table: dict[tuple[str, str, str], PricingRow] = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_id = (row.get("model_id") or "").strip()
            if not model_id:
                continue
            thinking = (row.get("thinking_mode") or "unknown").strip()
            status = (row.get("status") or "").strip()
            notes = (row.get("notes") or "").strip()
            # standard
            table[("Anthropic", model_id, "standard")] = PricingRow(
                provider="Anthropic",
                model_id=model_id,
                pricing_mode="standard",
                input_price_usd=_to_float(row.get("standard_base_input_usd_per_mtok")),
                cached_input_price_usd=_to_float(row.get("standard_cache_read_usd_per_mtok")),
                output_price_usd=_to_float(row.get("standard_output_usd_per_mtok")),
                thinking_support=thinking,
                status=status,
                notes=notes,
            )
            # Batch-mode row is registered only when batch columns are populated.
            batch_in = _to_float(row.get("batch_input_usd_per_mtok"))
            batch_out = _to_float(row.get("batch_output_usd_per_mtok"))
            if batch_in is not None or batch_out is not None:
                table[("Anthropic", model_id, "batch")] = PricingRow(
                    provider="Anthropic",
                    model_id=model_id,
                    pricing_mode="batch",
                    input_price_usd=batch_in,
                    cached_input_price_usd=None,
                    output_price_usd=batch_out,
                    thinking_support=thinking,
                    status=status,
                    notes=notes,
                )
    return table


def _load_google_pricing(csv_path: Path) -> dict[tuple[str, str, str], PricingRow]:
    """Load data/model_cost/google.csv.

    同じ (model_id, pricing_mode) に対し prompt_size_band が複数ある場合,
    優先度 (all > <=200K > ...) に従って代表行を選ぶ. 非チャット用途はスキップ.
    """
    # 先に (model_id, pricing_mode) 単位で候補行を集め, 優先度で1行選ぶ.
    buckets: dict[tuple[str, str], list[dict[str, str]]] = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            family = (row.get("family") or "").strip()
            if family in _GOOGLE_EXCLUDED_FAMILIES:
                continue
            model_id = (row.get("model_id") or "").strip()
            pricing_mode = (row.get("pricing_mode") or "").strip()
            if not model_id or not pricing_mode:
                continue
            buckets.setdefault((model_id, pricing_mode), []).append(row)

    table: dict[tuple[str, str, str], PricingRow] = {}
    for (model_id, pricing_mode), rows in buckets.items():
        chosen = _pick_google_row(rows)
        if chosen is None:
            continue
        table[("Google", model_id, pricing_mode)] = PricingRow(
            provider="Google",
            model_id=model_id,
            pricing_mode=pricing_mode,
            input_price_usd=_to_float(chosen.get("input_price_usd")),
            cached_input_price_usd=_to_float(chosen.get("cached_input_price_usd")),
            output_price_usd=_to_float(chosen.get("output_price_usd")),
            thinking_support=(chosen.get("thinking_support") or "unknown").strip(),
            status=(chosen.get("status") or "").strip(),
            notes=(chosen.get("notes") or "").strip(),
        )
    return table


def _pick_google_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    """Pick a representative row by prompt_size_band priority.

    複数行候補から優先度に従って1つ選ぶ. 優先度不明の行は最低優先.
    """
    if not rows:
        return None
    def rank(r: dict[str, str]) -> int:
        band = (r.get("prompt_size_band") or "").strip()
        if band in _GOOGLE_PROMPT_BAND_PRIORITY:
            return _GOOGLE_PROMPT_BAND_PRIORITY.index(band)
        return len(_GOOGLE_PROMPT_BAND_PRIORITY) + 1
    return sorted(rows, key=rank)[0]


def _to_float(value: str | None) -> float | None:
    """Parse optional float string. Empty / whitespace returns None.

    空文字列や空白はNone, その他はfloatに変換する.
    """
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def resolve_pricing_row(
    table: dict[tuple[str, str, str], PricingRow],
    provider_key: str,
    model_id: str,
    pricing_mode: str = "standard",
) -> PricingRow | None:
    """Resolve a PricingRow for a given (config provider, model_id, mode).

    config の provider キー (llm.type) と model_id から該当する PricingRow を解決する.
    ollama など無料扱いのプロバイダは None を返す (呼び出し側で無料計上する).

    Args:
        table (dict): Pricing table / 料金テーブル
        provider_key (str): Config provider key / config の provider キー
        model_id (str): Model identifier / モデルID
        pricing_mode (str): Pricing mode / 料金モード (既定: standard)

    Returns:
        PricingRow | None: Matching row or None if free / 該当行. 無料プロバイダはNone.
    """
    key = provider_key.lower().strip()
    if key in _FREE_PROVIDERS:
        return None
    csv_provider = _PROVIDER_ALIAS.get(key)
    if csv_provider is None:
        logger.warning("Unknown provider for cost calc: %s", provider_key)
        return None
    row = table.get((csv_provider, model_id, pricing_mode))
    if row is None and pricing_mode != "standard":
        # fallback to standard if requested mode is missing
        row = table.get((csv_provider, model_id, "standard"))
        if row is not None:
            logger.warning(
                "Pricing mode '%s' not found for %s/%s; falling back to standard",
                pricing_mode,
                csv_provider,
                model_id,
            )
    if row is None:
        logger.warning(
            "No pricing entry for %s/%s (mode=%s); cost will be recorded as 0",
            csv_provider,
            model_id,
            pricing_mode,
        )
    return row


def extract_usage(
    usage_metadata: dict[str, Any] | None,
    response_metadata: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Extract normalized token counts from LangChain metadata.

    LangChain の usage_metadata / response_metadata から正規化されたトークン数を抽出する.

    LangChain 1.0+ の usage_metadata:
        - input_tokens: 合計入力 (cache含む)
        - output_tokens: 合計出力 (reasoning/thinking含む場合あり)
        - total_tokens: 合計
        - input_token_details: {cache_read, cache_creation, ...}
        - output_token_details: {reasoning, ...}

    Anthropic extended thinking は output_token_details.reasoning 相当,
    OpenAI reasoning も同様に output_token_details.reasoning として入る.
    Google の cached_content_token_count は input_token_details.cache_read として入る.

    Args:
        usage_metadata (dict | None): AIMessage.usage_metadata
        response_metadata (dict | None): AIMessage.response_metadata (補完用)

    Returns:
        dict[str, int]: {input, cached_input, output, thinking} / 正規化トークン数
    """
    out = {"input": 0, "cached_input": 0, "output": 0, "thinking": 0}
    if not usage_metadata:
        # best-effort: response_metadata 側の token_usage (OpenAI 旧形式) を拾う
        if response_metadata:
            token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
            out["input"] = int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0)
            out["output"] = int(token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0)
        return out

    total_input = int(usage_metadata.get("input_tokens") or 0)
    total_output = int(usage_metadata.get("output_tokens") or 0)

    input_details = usage_metadata.get("input_token_details") or {}
    output_details = usage_metadata.get("output_token_details") or {}

    cache_read = int(input_details.get("cache_read") or 0)
    # Anthropic cache_creation は新規キャッシュ書き込み分 (課金上は通常の入力より高価だが
    # models.csv には cached_input (=cache_read) 価格のみ整備されているため, ここでは
    # cache_creation は通常入力扱いで集計する (= total_input から cache_read のみ差引).
    cached_input = cache_read
    uncached_input = max(0, total_input - cached_input)

    thinking = int(output_details.get("reasoning") or 0)
    uncached_output = max(0, total_output - thinking)

    out["input"] = uncached_input
    out["cached_input"] = cached_input
    out["output"] = uncached_output
    out["thinking"] = thinking
    return out


def calculate_cost(
    usage: dict[str, int],
    pricing: PricingRow | None,
) -> tuple[float, bool]:
    """Calculate USD cost for a single call from normalized usage.

    正規化されたトークン使用量から1回分のUSDコストを計算する.

    thinking / reasoning トークンは出力価格で課金 (Anthropic / OpenAI 共通の挙動).

    Args:
        usage (dict): {input, cached_input, output, thinking}
        pricing (PricingRow | None): Pricing row / 料金行. None は無料 (ollama等) または未登録.

    Returns:
        tuple[float, bool]: (cost_usd, unknown_pricing_flag)
    """
    if pricing is None:
        return 0.0, True
    unknown = False
    cost = 0.0
    if pricing.input_price_usd is not None:
        cost += usage["input"] * pricing.input_price_usd / _MILLION
    elif usage["input"] > 0:
        unknown = True
    if pricing.cached_input_price_usd is not None:
        cost += usage["cached_input"] * pricing.cached_input_price_usd / _MILLION
    elif usage["cached_input"] > 0 and pricing.input_price_usd is not None:
        # cached 価格未登録なら通常入力価格で代用
        cost += usage["cached_input"] * pricing.input_price_usd / _MILLION
    if pricing.output_price_usd is not None:
        cost += (usage["output"] + usage["thinking"]) * pricing.output_price_usd / _MILLION
    elif usage["output"] + usage["thinking"] > 0:
        unknown = True
    return cost, unknown


def build_record(  # noqa: PLR0913
    provider_key: str,
    model_id: str,
    pricing_mode: str,
    usage_metadata: dict[str, Any] | None,
    response_metadata: dict[str, Any] | None,
    table: dict[tuple[str, str, str], PricingRow],
) -> CostRecord:
    """Convenience: extract usage, resolve pricing, compute cost.

    usage抽出・価格解決・コスト計算を一括で行う.

    Args:
        provider_key (str): Config provider key / configのproviderキー
        model_id (str): Model identifier / モデルID
        pricing_mode (str): Pricing mode / 料金モード
        usage_metadata (dict | None): AIMessage.usage_metadata
        response_metadata (dict | None): AIMessage.response_metadata
        table (dict): Pricing table / 料金テーブル

    Returns:
        CostRecord: Per-call cost record / 呼び出し単位のコストレコード
    """
    pricing = resolve_pricing_row(table, provider_key, model_id, pricing_mode)
    usage = extract_usage(usage_metadata, response_metadata)
    cost, unknown = calculate_cost(usage, pricing)
    return CostRecord(
        provider=_PROVIDER_ALIAS.get(provider_key.lower().strip(), provider_key),
        model_id=model_id,
        pricing_mode=pricing.pricing_mode if pricing else pricing_mode,
        input_tokens=usage["input"],
        cached_input_tokens=usage["cached_input"],
        output_tokens=usage["output"],
        thinking_tokens=usage["thinking"],
        cost_usd=cost,
        unknown_pricing=unknown or (pricing is None and provider_key.lower().strip() not in _FREE_PROVIDERS),
    )
