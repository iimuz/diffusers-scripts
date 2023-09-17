---
title: Stable Diffusionの実行スクリプト
date: 2023-09-17
lastmod: 2023-09-17
---

## 概要

## 実行方法

### Text to Image

```sh
# `data/processed/{date-folder}`に画像が生成される
# CPU実行の場合
$ python src/text2img.py "a photo of an astronaut riding a horse on mars."

# CUDA実行の場合
$ python src/text2img.py --device=cuda "a photo of an astronaut riding a horse on mars."
```

## 環境構築

```sh
# torch cu117版を指定してますが、環境に合わせて適切なバージョンを指定してください。
# 動作確認したバージョンを固定で導入する場合
$ pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu117
# or
# 最新のバージョンを確認して導入する場合
$ pip install -e . --extra-index-url https://download.pytorch.org/whl/cu117
```

### 開発環境構築

```sh
# 動作確認したバージョンを固定で導入する場合
$ pip install -r requirements-dev.txt --index-url https://download.pytorch.org/whl/cu117
# or
# 最新のバージョンを確認して導入する場合
$ pip install -e . -c requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
```

### 実行環境の更新

既存の venv 環境を削除後に下記のコマンドで環境を構築する。

```sh
$ pip install -e . --extra-index-url https://download.pytorch.org/whl/cu117
$ pip freeze > requirements.txt
# requirements.txtに対して下記の変更を実施
#
# - pytorchのcudaバージョン指定を削除
# - `-e`で指定されている行を削除

# 開発環境の構築
# `-c`オプションでrequirements.txtの内容は一致させる
$ pip install -e .[dev] -c requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
$ pip freeze > requirements-dev.txt  # requirements.txtと同様に処理
```
