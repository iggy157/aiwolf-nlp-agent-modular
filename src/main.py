"""Script to launch agents according to configuration.

設定に応じたエージェントを起動するスクリプト.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
from pathlib import Path
from typing import Any

import yaml

from starter import connect

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
console_handler.setFormatter(formatter)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load main config and merge the mode-specific child config.

    メインconfigを読み込み, モードに応じた子configをマージして返す.

    The main config must define `mode` (multi_turn / single_turn) and `configs`
    which maps each mode to a child config path (relative to the main config).
    The child config is merged on top of the main config: on key conflict,
    the child config wins.

    Legacy flat configs (without `mode` / `configs`) are still accepted and
    returned as-is.

    Args:
        config_path (Path): Path to the main configuration file / メイン設定ファイルのパス

    Returns:
        dict[str, Any]: Merged configuration / マージ済みの設定
    """
    with Path.open(config_path) as f:
        main_config = yaml.safe_load(f)

    mode = main_config.get("mode")
    configs = main_config.get("configs")
    if mode is None or configs is None:
        return main_config

    child_rel = configs.get(mode)
    if not child_rel:
        msg = f"モード '{mode}' に対応する子configファイルが configs で定義されていません"
        raise ValueError(msg)

    child_path = (config_path.parent / child_rel).resolve()
    with Path.open(child_path) as f:
        child_config = yaml.safe_load(f) or {}

    merged: dict[str, Any] = {**main_config, **child_config}
    merged["mode"] = mode
    return merged


def execute(config_path: Path) -> None:
    """Execute based on the configuration file.

    設定ファイルをもとに実行する.

    Args:
        config_path (Path): Path to the configuration file / 設定ファイルのパス
    """
    config = load_config(config_path)
    logger.info("設定ファイルを読み込みました (mode=%s)", config.get("mode", "legacy"))

    agent_num = int(config["agent"]["num"])
    threads: list[multiprocessing.Process] = []
    for i in range(agent_num):
        thread = multiprocessing.Process(
            target=connect,
            args=(config, i + 1),
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        nargs="+",
        default=["./config/config.main.jp.yml"],
        help="メイン設定ファイルのパス (複数指定可). 既定: ./config/config.main.jp.yml (英語プロンプトは ./config/config.main.en.yml)",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for config_path in args.config:
        glob_path = Path(config_path)
        paths.extend([path for path in Path.glob(glob_path.parent, glob_path.name) if path.is_file()])

    multiprocessing.set_start_method("spawn")
    threads: list[multiprocessing.Process] = []
    for path in paths:
        thread = multiprocessing.Process(
            target=execute,
            args=(Path(path),),
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
