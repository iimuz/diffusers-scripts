---
title: Stable Diffusionの実行スクリプト
date: 2023-09-17
lastmod: 2023-09-19
---

## 概要

## 実行方法

各スクリプトの実行方法は、スクリプトファイルの docstring に記載しています。
以下は、実行を想定しているスクリプトのファイル名と、スクリプトの簡易説明のみ記載します。
スクリプトのオプションは`python hoge.py -h`のようにしてオプションを出力して確認してください。

- test2img.py: Stable Diffusion v1.x, v2.x を利用した Text2Image の実行用スクリプトです。
- test2img-sdxl.py: Stable Diffusion XL を利用した Text2Image の実行用スクリプトです。

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

## Tips

### モデルの保存場所

hugging face のモデルを利用する場合は、`$HOME/.cache/huggingface/hub`にモデルが保存される。
