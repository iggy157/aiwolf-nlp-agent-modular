"""Module that defines the base class for agents.

エージェントの基底クラスを定義するモジュール.
"""

from __future__ import annotations

import asyncio
import os
import random
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage

from aiwolf_nlp_common.packet import Info, Packet, Request, Role, Setting, Status, Talk

from utils.agent_logger import AgentLogger
from utils.stoppable_thread import StoppableThread

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
T = TypeVar("T")

_TALK_REQUESTS = {Request.TALK, Request.WHISPER}
_ACTION_REQUESTS = {Request.VOTE, Request.DIVINE, Request.GUARD, Request.ATTACK}
_SHARED_REQUESTS = {Request.INITIALIZE, Request.DAILY_INITIALIZE, Request.DAILY_FINISH}

_BLOCKS_DIR = Path(__file__).parent.joinpath("./../../prompts/aiwolf/blocks").resolve()
_JINJA_ENV = Environment(
    loader=FileSystemLoader(str(_BLOCKS_DIR)),
    autoescape=select_autoescape(enabled_extensions=(), default=False),
    trim_blocks=False,
    lstrip_blocks=False,
    keep_trailing_newline=False,
)


class Agent:
    """Base class for agents.

    エージェントの基底クラス.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,
    ) -> None:
        """Initialize the agent.

        エージェントの初期化を行う.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role / 役職
        """
        self.config = config
        self.agent_name = name
        self.agent_logger = AgentLogger(config, name, game_id)
        self.request: Request | None = None
        self.info: Info | None = None
        self.setting: Setting | None = None
        self.talk_history: list[Talk] = []
        self.whisper_history: list[Talk] = []
        self.role = role
        # グループチャット方式
        self.in_talk_phase = False
        self.in_whisper_phase = False

        self.sent_talk_count: int = 0
        self.sent_whisper_count: int = 0
        self.llm_model: BaseChatModel | None = None
        self.llm_message_history: list[BaseMessage] = []
        self.llm_model_talk: BaseChatModel | None = None
        self.llm_model_action: BaseChatModel | None = None
        self.llm_message_history_talk: list[BaseMessage] = []
        self.llm_message_history_action: list[BaseMessage] = []
        # single-turnモードで各日のdaily_initialize/daily_finishスナップショットを蓄積する.
        self.day_events: list[dict[str, Any]] = []

        load_dotenv(Path(__file__).parent.joinpath("./../../config/.env"))

    def _is_separate_langchain(self) -> bool:
        """Return whether LangChain instances are separated by request type.

        リクエスト種別ごとにLangChainを分離するかどうかを返す.

        Returns:
            bool: True if separated / 分離している場合はTrue
        """
        llm_config = self.config.get("llm", {})
        return bool(llm_config.get("separate_langchain", False))

    def _is_single_turn(self) -> bool:
        """Return whether the agent is running in single-turn mode.

        single-turnモードで動作しているかを返す.

        Returns:
            bool: True if single-turn / single-turnの場合はTrue
        """
        return str(self.config.get("mode", "multi_turn")) == "single_turn"

    @staticmethod
    def timeout(func: Callable[P, T]) -> Callable[P, T]:
        """Decorator to set action timeout.

        アクションタイムアウトを設定するデコレータ.

        Args:
            func (Callable[P, T]): Function to be decorated / デコレート対象の関数

        Returns:
            Callable[P, T]: Function with timeout functionality / タイムアウト機能を追加した関数
        """

        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            res: T | Exception = Exception("No result")

            def execute_with_timeout() -> None:
                nonlocal res
                try:
                    res = func(*args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    res = e

            thread = StoppableThread(target=execute_with_timeout)
            thread.start()
            self = args[0] if args else None
            if not isinstance(self, Agent):
                raise TypeError(self, " is not an Agent instance")
            timeout_value = (self.setting.timeout.action if hasattr(self, "setting") and self.setting else 0) // 1000
            if timeout_value > 0:
                thread.join(timeout=timeout_value)
                if thread.is_alive():
                    self.agent_logger.logger.warning(
                        "アクションがタイムアウトしました: %s",
                        self.request,
                    )
                    if bool(self.config["agent"]["kill_on_timeout"]):
                        thread.stop()
                        self.agent_logger.logger.warning(
                            "アクションを強制終了しました: %s",
                            self.request,
                        )
            else:
                thread.join()
            if isinstance(res, Exception):  # type: ignore[arg-type]
                raise res
            return res

        return _wrapper

    def set_packet(self, packet: Packet) -> None:
        """Set packet information.

        パケット情報をセットする.

        Args:
            packet (Packet): Received packet / 受信したパケット
        """
        self.request = packet.request
        if packet.info:
            self.info = packet.info
        if packet.setting:
            self.setting = packet.setting
        if packet.talk_history:
            self.talk_history.extend(packet.talk_history)
        if packet.whisper_history:
            self.whisper_history.extend(packet.whisper_history)

        # グループチャット方式
        if packet.new_talk:
            self.talk_history.append(packet.new_talk)
            self.on_talk_received(packet.new_talk)
        if packet.new_whisper:
            self.whisper_history.append(packet.new_whisper)
            self.on_whisper_received(packet.new_whisper)

        if self.request == Request.INITIALIZE:
            self.talk_history: list[Talk] = []
            self.whisper_history: list[Talk] = []
            self.llm_message_history: list[BaseMessage] = []
            self.llm_message_history_talk: list[BaseMessage] = []
            self.llm_message_history_action: list[BaseMessage] = []
            self.day_events = []
        if self.request in (Request.DAILY_INITIALIZE, Request.DAILY_FINISH) and packet.info is not None:
            self.day_events.append(
                {
                    "day": packet.info.day,
                    "phase": self.request.name.lower(),
                    "medium_result": packet.info.medium_result,
                    "divine_result": packet.info.divine_result,
                    "executed_agent": packet.info.executed_agent,
                    "attacked_agent": packet.info.attacked_agent,
                    "vote_list": packet.info.vote_list,
                    "attack_vote_list": packet.info.attack_vote_list,
                },
            )
        self.agent_logger.logger.debug(packet)

    def get_alive_agents(self) -> list[str]:
        """Get the list of alive agents.

        生存しているエージェントのリストを取得する.

        Returns:
            list[str]: List of alive agent names / 生存エージェント名のリスト
        """
        if not self.info:
            return []
        return [k for k, v in self.info.status_map.items() if v == Status.ALIVE]

    def on_talk_received(self, talk: Talk) -> None:
        """Called when a new talk is received (freeform mode).

        新しいトークを受信した時に呼ばれる (グループチャット方式用).

        Args:
            talk (Talk): Received talk / 受信したトーク
        """

    def on_whisper_received(self, whisper: Talk) -> None:
        """Called when a new whisper is received (freeform mode).

        新しい囁きを受信した時に呼ばれる (グループチャット方式用).

        Args:
            whisper (Talk): Received whisper / 受信した囁き
        """

    async def handle_talk_phase(self, send: Callable[[str], None]) -> None:
        """Handle talk phase in freeform mode.

        グループチャット方式でのトークフェーズ処理.

        Args:
            send (Callable[[str], None]): Send function / 送信関数
        """
        while self.in_talk_phase:
            if self.info and self.info.remain_count is not None and self.info.remain_count <= 0:
                break

            text = self.talk()
            if not self.in_talk_phase:
                break
            send(text)
            await asyncio.sleep(5)

    async def handle_whisper_phase(self, send: Callable[[str], None]) -> None:
        """Handle whisper phase in freeform mode.

        グループチャット方式での囁きフェーズ処理.

        Args:
            send (Callable[[str], None]): Send function / 送信関数
        """
        while self.in_whisper_phase:
            if self.info and self.info.remain_count is not None and self.info.remain_count <= 0:
                break

            text = self.whisper()
            if not self.in_whisper_phase:
                break
            send(text)
            await asyncio.sleep(5)

    def _resolve_targets(
        self,
        request: Request,
    ) -> list[tuple[BaseChatModel, list[BaseMessage], str]]:
        """Return list of (model, history, label) pairs to send the prompt to.

        プロンプトの送信先 (モデル, 履歴, ラベル) の組を返す.

        Args:
            request (Request): Request type / リクエストタイプ

        Returns:
            list[tuple[BaseChatModel, list[BaseMessage], str]]: Send targets / 送信先のリスト
        """
        if not self._is_separate_langchain():
            if self.llm_model is None:
                return []
            return [(self.llm_model, self.llm_message_history, "default")]

        targets: list[tuple[BaseChatModel, list[BaseMessage], str]] = []
        if request in _SHARED_REQUESTS:
            if self.llm_model_talk is not None:
                targets.append((self.llm_model_talk, self.llm_message_history_talk, "talk"))
            if self.llm_model_action is not None:
                targets.append((self.llm_model_action, self.llm_message_history_action, "action"))
        elif request in _TALK_REQUESTS:
            if self.llm_model_talk is not None:
                targets.append((self.llm_model_talk, self.llm_message_history_talk, "talk"))
        elif request in _ACTION_REQUESTS:
            if self.llm_model_action is not None:
                targets.append((self.llm_model_action, self.llm_message_history_action, "action"))
        return targets

    def _send_message_to_llm(self, request: Request | None) -> str | None:
        """Send message to LLM and get response.

        LLMにメッセージを送信して応答を取得する.

        Args:
            request (Request | None): The request type to process / 処理するリクエストタイプ

        Returns:
            str | None: LLM response or None if error occurred / LLMの応答またはエラー時はNone
        """
        if request is None:
            return None
        is_single_turn = self._is_single_turn()
        # single-turn では共通リクエストはLLMに送らず, day_events等としてコンテキスト保持のみ行う.
        if is_single_turn and request in _SHARED_REQUESTS:
            return None
        request_key = request.lower()
        if request_key not in self.config["prompt"]:
            return None
        prompt = self.config["prompt"][request_key]
        if float(self.config["llm"]["sleep_time"]) > 0:
            sleep(float(self.config["llm"]["sleep_time"]))
        key = {
            "info": self.info,
            "setting": self.setting,
            "talk_history": self.talk_history,
            "whisper_history": self.whisper_history,
            "role": self.role,
            "sent_talk_count": self.sent_talk_count,
            "sent_whisper_count": self.sent_whisper_count,
            "day_events": self.day_events,
            "mode": self.config.get("mode", "multi_turn"),
            "request_key": request_key,
        }
        template = _JINJA_ENV.from_string(prompt)
        prompt = template.render(**key).strip()
        targets = self._resolve_targets(request)
        if not targets:
            self.agent_logger.logger.error("LLM is not initialized")
            return None
        last_response: str | None = None
        for model, history, label in targets:
            try:
                if is_single_turn:
                    response = (model | StrOutputParser()).invoke([HumanMessage(content=prompt)])
                else:
                    history.append(HumanMessage(content=prompt))
                    response = (model | StrOutputParser()).invoke(history)
                    history.append(AIMessage(content=response))
                self.agent_logger.logger.info(["LLM", label, prompt, response])
                last_response = response
            except Exception:
                self.agent_logger.logger.exception("Failed to send message to LLM (%s)", label)
                continue
        return last_response

    @timeout
    def name(self) -> str:
        """Return response to name request.

        名前リクエストに対する応答を返す.

        Returns:
            str: Agent name / エージェント名
        """
        return self.agent_name

    def _create_llm_model(self, model_type: str) -> BaseChatModel:
        """Create an LLM model instance for the given provider type.

        指定されたプロバイダタイプのLLMモデルインスタンスを生成する.

        Args:
            model_type (str): Provider type / プロバイダタイプ

        Returns:
            BaseChatModel: Created LLM model / 生成したLLMモデル
        """
        match model_type:
            case "openai":
                return ChatOpenAI(
                    model=str(self.config["openai"]["model"]),
                    temperature=float(self.config["openai"]["temperature"]),
                    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
                )
            case "google":
                return ChatGoogleGenerativeAI(
                    model=str(self.config["google"]["model"]),
                    temperature=float(self.config["google"]["temperature"]),
                    api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),
                )
            case "vertexai":
                vertexai_config = self.config["vertexai"]
                return ChatGoogleGenerativeAI(
                    model=str(vertexai_config["model"]),
                    temperature=float(vertexai_config["temperature"]),
                    vertexai=True,
                )
            case "ollama":
                return ChatOllama(
                    model=str(self.config["ollama"]["model"]),
                    temperature=float(self.config["ollama"]["temperature"]),
                    base_url=str(self.config["ollama"]["base_url"]),
                )
            case "claude":
                claude_config = self.config["claude"]
                return ChatAnthropic(
                    model_name=str(claude_config["model"]),
                    temperature=float(claude_config["temperature"]),
                    timeout=None,
                    stop=None,
                    api_key=SecretStr(os.environ["CLAUDE_API_KEY"]),
                )
            case _:
                raise ValueError(model_type, "Unknown LLM type")

    def initialize(self) -> None:
        """Perform initialization for game start request.

        ゲーム開始リクエストに対する初期化処理を行う.
        """
        if self.info is None:
            return

        if self._is_separate_langchain():
            talk_type = str(self.config["llm"]["talk"]["type"])
            action_type = str(self.config["llm"]["action"]["type"])
            self.llm_model_talk = self._create_llm_model(talk_type)
            self.llm_model_action = self._create_llm_model(action_type)
        else:
            model_type = str(self.config["llm"]["type"])
            self.llm_model = self._create_llm_model(model_type)
        self._send_message_to_llm(self.request)

    def daily_initialize(self) -> None:
        """Perform processing for daily initialization request.

        昼開始リクエストに対する処理を行う.
        """
        self._send_message_to_llm(self.request)

    def whisper(self) -> str:
        """Return response to whisper request.

        囁きリクエストに対する応答を返す.

        Returns:
            str: Whisper message / 囁きメッセージ
        """
        response = self._send_message_to_llm(self.request)
        self.sent_whisper_count = len(self.whisper_history)
        return response or ""

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.

        Returns:
            str: Talk message / 発言メッセージ
        """
        response = self._send_message_to_llm(Request.TALK)
        self.sent_talk_count = len(self.talk_history)
        return response or ""

    def daily_finish(self) -> None:
        """Perform processing for daily finish request.

        昼終了リクエストに対する処理を行う.
        """
        self._send_message_to_llm(self.request)

    def divine(self) -> str:
        """Return response to divine request.

        占いリクエストに対する応答を返す.

        Returns:
            str: Agent name to divine / 占い対象のエージェント名
        """
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def guard(self) -> str:
        """Return response to guard request.

        護衛リクエストに対する応答を返す.

        Returns:
            str: Agent name to guard / 護衛対象のエージェント名
        """
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def attack(self) -> str:
        """Return response to attack request.

        襲撃リクエストに対する応答を返す.

        Returns:
            str: Agent name to attack / 襲撃対象のエージェント名
        """
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def finish(self) -> None:
        """Perform processing for game finish request.

        ゲーム終了リクエストに対する処理を行う.
        """

    @timeout
    def action(self) -> str | None:  # noqa: C901, PLR0911
        """Execute action according to request type.

        リクエストの種類に応じたアクションを実行する.

        Returns:
            str | None: Action result string or None / アクションの結果文字列またはNone
        """
        match self.request:
            case Request.NAME:
                return self.name()
            case Request.TALK:
                return self.talk()
            case Request.WHISPER:
                return self.whisper()
            case Request.VOTE:
                return self.vote()
            case Request.DIVINE:
                return self.divine()
            case Request.GUARD:
                return self.guard()
            case Request.ATTACK:
                return self.attack()
            case Request.INITIALIZE:
                self.initialize()
            case Request.DAILY_INITIALIZE:
                self.daily_initialize()
            case Request.DAILY_FINISH:
                self.daily_finish()
            case Request.FINISH:
                self.finish()
            case _:
                pass
        return None
