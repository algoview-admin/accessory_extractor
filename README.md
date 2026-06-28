# Accessory Extractor

アクセサリー商品画像から背景・スタンドを除去し、透明PNG（切り抜き画像）を生成するブラウザツールです。

イヤリング・ネックレス・リング・バレッタ・ブレスレット・ブローチなど、あらゆるアクセサリーに対応しています。

## できること

| 方法 | 概要 | 向いているケース |
|------|------|----------------|
| **A: 手動クロップ + rembg** | ドラッグで範囲を指定して背景除去 | シンプルな白背景商品写真 |
| **B: SAM2（矩形＋前景点）** | 矩形と前景点を指定して高精度セグメンテーション | 複雑な背景・マネキン付き画像 |
| **C: Gemini 自動検出** | AIがアクセサリー位置を自動検出してクロップ→背景除去 | 手間を省きたい場合 |

いずれの方法も出力は透明背景の PNG です。

## 必要要件

- Python 3.11 以上
- 方法C を使う場合：Google AI API キー（[Google AI Studio](https://aistudio.google.com/) で取得）
- 方法B を使う場合：SAM2（別途インストール、後述）

## セットアップ

### 1. パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. SAM2 のインストール（方法B を使う場合のみ）

```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

### 3. API キーの設定（方法C を使う場合のみ）

プロジェクトルートに `.env` ファイルを作成し、以下を記述します：

```
GOOGLE_API_KEY=your_api_key_here
```

モデルを変更したい場合は `GEMINI_MODEL` も指定できます（省略時は `gemini-2.5-flash-preview-05-20`）：

```
GEMINI_MODEL=gemini-2.0-flash
```

## 起動方法

```bash
python server.py
```

起動後、ブラウザで `http://localhost:8080` にアクセスしてください。

## ファイル構成

```
accessory_extractor/
├── server.py                # FastAPI サーバー
├── extract_acc_rembg.py     # 方法A: rembg による背景除去
├── extract_acc_sam2.py      # 方法B: SAM2 によるセグメンテーション
├── extract_acc_gemini.py    # 方法C: Gemini による自動検出
├── static/
│   └── index.html           # ブラウザUI
├── input_data/              # 入力画像置き場（任意）
├── output_data/             # 出力PNG の保存先（.gitignore 対象）
└── requirements.txt
```

## CLIとしての使用

各スクリプトはコマンドラインからも直接実行できます。

```bash
# 方法A: 画像全体を処理
python extract_acc_rembg.py input_data/er149886/er149886_1.jpg

# 方法A: 範囲を指定してクロップ後に処理
python extract_acc_rembg.py input_data/er149886/er149886_1.jpg 120 80 400 520

# 方法B: 矩形を指定して SAM2 で処理
python extract_acc_sam2.py input_data/er149886/er149886_1.jpg 120 80 400 520

# 方法C: Gemini で自動検出
python extract_acc_gemini.py input_data/er149886/er149886_1.jpg
```

出力は `output_data/<商品ID>/<商品ID>_extracted.png` に保存されます。
