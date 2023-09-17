"""SDXLを利用してText2Imageを行うスクリプト."""
import logging
import sys
from argparse import ArgumentParser
from datetime import datetime
from enum import Enum
from logging import Formatter, StreamHandler
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import torch
from diffusers import DiffusionPipeline
from pydantic import BaseModel

_logger = logging.getLogger(__name__)


class _DeviceOption(Enum):
    """デバイスの選択肢."""

    CPU = "cpu"
    CUDA = "cuda"


class _RunConfig(BaseModel):
    """スクリプト実行のためのオプション."""

    prompt: str  # 画像生成に利用するプロンプト

    network_name: str  # 画像生成に利用するモデル名
    guidance_scale: float
    num_inference_steps: int  # 生成ステップ数
    seed: int  # 画像生成に利用するシード
    device: _DeviceOption  # 利用するデバイス

    verbose: int  # ログレベル


def _get_model_id(name: str) -> str:
    """いくつかのモデルについては簡易名称を用意する.それ以外は、入力値を利用する."""
    short_model_names = {
        # "short name": "long name"
        "SDXL1.0-base": "stabilityai/stable-diffusion-xl-base-1.0",
    }

    model_id = short_model_names.get(name, name)

    return model_id


def _main() -> None:
    """スクリプトのエントリポイント."""
    # 実行時引数の読み込み
    config = _parse_args()

    # ログ設定
    loglevel = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }.get(config.verbose, logging.DEBUG)
    script_filepath = Path(__file__)
    log_filepath = (
        Path("data/interim")
        / script_filepath.parent.name
        / f"{script_filepath.stem}.log"
    )
    log_filepath.parent.mkdir(exist_ok=True)
    _setup_logger(log_filepath, loglevel=loglevel)
    _logger.info(config)

    # 画像の出力フォルダ
    output_dir = (
        Path("data/processed")
        / script_filepath.stem
        / datetime.now().strftime(r"%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True)

    # モデル名の取得
    model_id = _get_model_id(config.network_name)

    # パイプラインの作成
    pipe_options: dict[str, Any] = dict()
    if config.device == _DeviceOption.CUDA:
        pipe_options["variant"] = "fp16"
        pipe_options["torch_dtype"] = torch.float16
    pipe = DiffusionPipeline.from_pretrained(
        model_id, use_safetensors=True, **pipe_options
    )
    pipe = pipe.to(config.device.value)

    # パイプラインの実行
    generator = torch.Generator(config.device.value).manual_seed(config.seed)
    image = pipe(
        config.prompt,
        guidance_scale=config.guidance_scale,
        generator=generator,
        # num_inference_steps=config.num_inference_steps,
    ).images[0]

    # 生成した画像の保存
    filename = datetime.now().strftime("%Y%m%d-%H%M%S") + ".png"
    image.save(output_dir / filename)


def _parse_args() -> _RunConfig:
    """スクリプト実行のための引数を読み込む."""
    parser = ArgumentParser(description="Text2Imageを実行する.")

    parser.add_argument(
        "prompt",
        help="画像生成に利用するプロンプト.",
    )

    parser.add_argument(
        "-n", "--network-name", default="SDXL1.0-base", help="画像生成に利用するモデル名."
    )
    parser.add_argument("-g", "--guidance-scale", default=7.5, help="guidance scale.")
    parser.add_argument("--seed", default=42, help="画像生成に利用するシード値.")
    parser.add_argument("-s", "--num-inference-steps", default=50, help="画像生成のステップ数.")
    parser.add_argument(
        "--device",
        default=_DeviceOption.CPU.value,
        choices=[v.value for v in _DeviceOption],
        help="利用するデバイス.",
    )

    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="詳細メッセージのレベルを設定."
    )

    args = parser.parse_args()
    config = _RunConfig(**vars(args))

    return config


def _setup_logger(filepath: Path | None, loglevel: int) -> None:
    """ロガー設定を行う.

    Parameters
    ----------
    filepath : Path | None
        ログ出力するファイルパス. Noneの場合はファイル出力しない.

    loglevel : int
        出力するログレベル.

    Notes
    -----
    ファイル出力とコンソール出力を行うように設定する。
    """
    lib_logger = logging.getLogger("src.exp20230515")

    _logger.setLevel(loglevel)
    lib_logger.setLevel(loglevel)

    # consoleログ
    console_handler = StreamHandler()
    console_handler.setLevel(loglevel)
    console_handler.setFormatter(
        Formatter("[%(levelname)7s] %(asctime)s (%(name)s) %(message)s")
    )
    _logger.addHandler(console_handler)
    lib_logger.addHandler(console_handler)

    # ファイル出力するログ
    # 基本的に大量に利用することを想定していないので、ログファイルは多くは残さない。
    if filepath is not None:
        file_handler = RotatingFileHandler(
            filepath,
            encoding="utf-8",
            mode="a",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=1,
        )
        file_handler.setLevel(loglevel)
        file_handler.setFormatter(
            Formatter("[%(levelname)7s] %(asctime)s (%(name)s) %(message)s")
        )
        _logger.addHandler(file_handler)
        lib_logger.addHandler(file_handler)


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        _logger.exception(e)
        sys.exit(1)
