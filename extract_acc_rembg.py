"""
rembgによるアクセサリー背景除去
Usage: python extract_acc_rembg.py <image_filepath> [x1 y1 x2 y2]
       ボックス指定なしの場合は画像全体を処理

.envファイル不要（外部API不使用）
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
from rembg import remove as rembg_remove

BASE_DIR = Path(__file__).parent


def remove_dark_pixels(pil_rgba: Image.Image, threshold: int) -> Image.Image:
    """rembg後に残った暗いピクセル（黒スタンド・マネキン）を透明化する。
    threshold: RGB各チャンネルがこの値以下のピクセルを除去（目安: 40〜80）
    """
    arr = np.array(pil_rgba)
    r, g, b, a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]
    is_dark = (r <= threshold) & (g <= threshold) & (b <= threshold) & (a > 0)
    arr[is_dark, 3] = 0
    return Image.fromarray(arr, "RGBA")


def rembg_tight(pil_img: Image.Image, dark_threshold: int | None = None) -> Image.Image:
    """背景除去 → 暗ピクセル除去（任意）→ アルファ境界でタイトクロップ"""
    result = rembg_remove(pil_img.convert("RGB"))
    if dark_threshold is not None:
        result = remove_dark_pixels(result, dark_threshold)
    arr = np.array(result)
    rows = np.where(arr[:, :, 3] > 10)[0]
    cols = np.where(arr[:, :, 3] > 10)[1]
    if len(rows) > 0:
        p = 8
        result = result.crop((
            max(0, cols.min() - p), max(0, rows.min() - p),
            min(result.width,  cols.max() + p + 1),
            min(result.height, rows.max() + p + 1),
        ))
    return result


def tight_crop_mask(pil_rgba: Image.Image) -> Image.Image:
    """RGBAマスク画像をアルファ境界でタイトクロップ（SAM2出力用）"""
    arr = np.array(pil_rgba)
    rows = np.where(arr[:, :, 3] > 10)[0]
    cols = np.where(arr[:, :, 3] > 10)[1]
    if len(rows) > 0:
        p = 8
        pil_rgba = pil_rgba.crop((
            max(0, cols.min() - p), max(0, rows.min() - p),
            min(pil_rgba.width,  cols.max() + p + 1),
            min(pil_rgba.height, rows.max() + p + 1),
        ))
    return pil_rgba


def extract(image_path: Path, output_path: Path, box: tuple[int,int,int,int] | None = None) -> None:
    pil_img = Image.open(image_path)
    if box is not None:
        x1, y1, x2, y2 = box
        W, H = pil_img.size
        pil_img = pil_img.crop((max(0, x1), max(0, y1), min(W, x2), min(H, y2)))
        print(f"クロップ: ({x1},{y1})-({x2},{y2})")
    print("rembgで背景を除去中...")
    result = rembg_tight(pil_img)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, format="PNG")
    print(f"保存完了: {output_path}  ({result.width}x{result.height}px)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_acc_rembg.py <image_path> [x1 y1 x2 y2]")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    product_id = image_path.stem.split("_")[0]
    output_path = BASE_DIR / "output_data" / product_id / f"{product_id}_extracted.png"

    box = None
    if len(sys.argv) == 6:
        box = tuple(int(v) for v in sys.argv[2:6])

    print(f"入力: {image_path}")
    print(f"出力: {output_path}")
    extract(image_path, output_path, box)


if __name__ == "__main__":
    main()
