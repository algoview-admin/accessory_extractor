"""
SAM2によるアクセサリー背景除去（矩形ボックスプロンプト）
Usage: python extract_acc_sam2.py <image_filepath> x1 y1 x2 y2
Example: python extract_acc_sam2.py ./input_data/er149886/er149886_2.jpg 120 80 400 520
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image

from extract_acc_rembg import tight_crop_mask

BASE_DIR = Path(__file__).parent

_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"SAM2モデルをロード中 (device={device})...")
        _predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2.1-hiera-small", device=device
        )
        print("SAM2ロード完了")
    return _predictor


def extract_with_box(
    pil_img: Image.Image,
    box: list[float],
    points: list[list[float]] | None = None,
) -> tuple[Image.Image, Image.Image]:
    """
    SAM2の矩形プロンプト（＋任意の前景点）で前景をセグメントして返す。

    Args:
        pil_img: 入力画像（RGB）
        box: [x1, y1, x2, y2] 原寸ピクセル座標
        points: [[x, y], ...] 前景点（原寸ピクセル座標）

    Returns:
        (result, preview): result=透明RGBA, preview=マスクオーバーレイ
    """
    import torch

    arr = np.array(pil_img.convert("RGB"))
    H, W = arr.shape[:2]

    predictor = get_predictor()
    predictor.set_image(arr)

    point_coords = np.array(points, dtype=np.float32) if points else None
    point_labels = np.ones(len(points), dtype=np.int32) if points else None

    with torch.inference_mode():
        masks, _, _ = predictor.predict(
            box=np.array(box, dtype=np.float32),
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )

    mask = masks[0].astype(bool)

    # 透明RGBA
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[:, :, :3] = arr
    rgba[:, :, 3] = (mask * 255).astype(np.uint8)
    result = tight_crop_mask(Image.fromarray(rgba, "RGBA"))

    # マスクオーバーレイプレビュー
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    overlay[mask, 0] = 80
    overlay[mask, 1] = 180
    overlay[mask, 3] = 120
    preview = Image.alpha_composite(
        pil_img.convert("RGBA"),
        Image.fromarray(overlay, "RGBA"),
    )

    return result, preview


def main():
    if len(sys.argv) != 6:
        print("Usage: python extract_acc_sam2.py <image_path> x1 y1 x2 y2")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    box = [float(v) for v in sys.argv[2:6]]
    product_id = image_path.stem.split("_")[0]
    output_path = BASE_DIR / "output_data" / product_id / f"{product_id}_extracted.png"

    print(f"入力: {image_path}")
    print(f"ボックス: {box}")
    pil_img = Image.open(image_path)
    result, _ = extract_with_box(pil_img, box)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, format="PNG")
    print(f"保存完了: {output_path}  ({result.width}x{result.height}px)")


if __name__ == "__main__":
    main()
