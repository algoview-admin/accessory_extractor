"""
Geminiでアクセサリーの位置を検出し、元画像をクロップ → rembgで背景除去
Usage:   python extract_acc_gemini.py <image_filepath> [item_type]
Example: python extract_acc_gemini.py ./input_data/er149886/er149886_2.jpg イヤリング

.env に GOOGLE_API_KEY と GEMINI_MODEL を記述してください。
"""

import sys
import os
import io
import re
import json
from pathlib import Path
from PIL import Image

BASE_DIR  = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent


def load_env() -> dict:
    env = {}
    for env_path in [BASE_DIR / ".env", REPO_ROOT / ".env"]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
    for k in list(env):
        if k in os.environ:
            env[k] = os.environ[k]
    return env


def load_api_key() -> str:
    key = load_env().get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise EnvironmentError(".env に GOOGLE_API_KEY=xxx を記述してください。")
    return key


def load_model() -> str:
    return load_env().get("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")


def build_bbox_prompt(item_type: str) -> str:
    return (
        f"この商品画像に写っている{item_type}の"
        "バウンディングボックスをJSONで返してください。\n"
        "展示スタンド・台座・台紙・背景は含めず、アクセサリー本体のみを対象にしてください。\n"
        "複数写っている場合は最も大きい（または左側の）1個を選んでください。\n"
        f'形式: {{"x1": <左端px>, "y1": <上端px>, "x2": <右端px>, "y2": <下端px>}}\n'
        "JSONのみ返してください。説明・テキスト不要。"
    )


def get_item_bbox(pil_img: Image.Image, item_type: str = "アクセサリー") -> tuple[int, int, int, int]:
    """Geminiにアクセサリーのバウンディングボックスをテキストで返してもらう"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=load_api_key())
    model  = load_model()

    W, H = pil_img.size
    png_buf = io.BytesIO()
    pil_img.convert("RGB").save(png_buf, format="PNG")

    prompt = f"画像サイズは {W}x{H} ピクセルです。\n" + build_bbox_prompt(item_type)

    print(f"Gemini APIで{item_type}の位置を検出中 ({model})...")
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=png_buf.getvalue(), mime_type="image/png"),
            types.Part.from_text(text=prompt),
        ],
    )

    text = response.text.strip()
    print(f"Geminiの応答: {text}")

    match = re.search(r'\{[^}]+\}', text)
    if not match:
        raise ValueError(f"バウンディングボックスを取得できませんでした: {text}")

    bbox = json.loads(match.group())
    x1 = max(0, int(bbox["x1"]))
    y1 = max(0, int(bbox["y1"]))
    x2 = min(W, int(bbox["x2"]))
    y2 = min(H, int(bbox["y2"]))
    return x1, y1, x2, y2


def extract(image_path: Path, output_path: Path, item_type: str = "アクセサリー") -> None:
    from extract_acc_rembg import rembg_tight

    print(f"入力: {image_path}")
    pil_img = Image.open(image_path)
    W, H = pil_img.size

    x1, y1, x2, y2 = get_item_bbox(pil_img, item_type)
    print(f"検出bbox: ({x1},{y1})-({x2},{y2})")

    pad = 20
    cropped = pil_img.convert("RGB").crop((
        max(0, x1 - pad), max(0, y1 - pad),
        min(W, x2 + pad), min(H, y2 + pad),
    ))

    print("rembgで背景を除去中...")
    result = rembg_tight(cropped)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, format="PNG")
    print(f"保存完了: {output_path}  ({result.width}x{result.height}px)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_acc_gemini.py <image_path> [item_type]")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    item_type  = sys.argv[2] if len(sys.argv) >= 3 else "アクセサリー"
    product_id = image_path.stem.split("_")[0]
    output_path = BASE_DIR / "output_data" / product_id / f"{product_id}_extracted.png"

    print(f"出力: {output_path}")
    extract(image_path, output_path, item_type)


if __name__ == "__main__":
    main()
