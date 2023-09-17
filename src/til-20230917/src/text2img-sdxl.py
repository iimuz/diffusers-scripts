"""SDXLを利用してText2Imageを行うスクリプト."""
import logging
import sys
from argparse import ArgumentParser
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from diffusers import DiffusionPipeline
from pydantic import BaseModel

import internal.data_directory as data_directory
import internal.log_settings as log_settings

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
    loglevel = log_settings.get_loglevel_from_verbosity(config.verbose)
    script_filepath = Path(__file__)
    log_filepath = data_directory.get_interim_dir() / f"{script_filepath.stem}.log"
    log_filepath.parent.mkdir(exist_ok=True)
    log_settings.setup_lib_logger(log_filepath, loglevel=loglevel)
    log_settings.setup_logger(_logger, log_filepath, loglevel=loglevel)
    _logger.info(config)

    # 画像の出力フォルダ
    output_dir = (
        data_directory.get_processed_dir()
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
        num_inference_steps=config.num_inference_steps,
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


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        _logger.exception(e)
        sys.exit(1)
