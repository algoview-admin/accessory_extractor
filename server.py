"""
アクセサリー抽出サーバー
Usage: python server.py
"""

import io
import json
import base64
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from extract_acc_gemini import get_item_bbox, BASE_DIR
from extract_acc_rembg import rembg_tight
from extract_acc_sam2 import extract_with_box as sam2_extract

app = FastAPI()

STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def save_output(result: Image.Image, product_id: str) -> str | None:
    if not product_id:
        return None
    out = BASE_DIR / "output_data" / product_id / f"{product_id}_extracted.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(out, format="PNG")
    return str(out.relative_to(BASE_DIR))


@app.get("/", response_class=HTMLResponse)
async def root():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


def _parse_dark_threshold(val: str) -> int | None:
    try:
        t = int(val)
        return t if 0 < t < 256 else None
    except (ValueError, TypeError):
        return None


@app.post("/extract/gemini")
async def extract_gemini(
    file: UploadFile = File(...),
    product_id: str = Form(""),
    dark_threshold: str = Form(""),
):
    try:
        pil_img = Image.open(io.BytesIO(await file.read()))
        W, H = pil_img.size

        x1, y1, x2, y2 = get_item_bbox(pil_img)

        preview = pil_img.convert("RGBA").copy()
        ImageDraw.Draw(preview).rectangle([x1, y1, x2, y2], outline=(255, 80, 0), width=4)

        pad = 20
        cropped = pil_img.convert("RGB").crop((
            max(0, x1 - pad), max(0, y1 - pad),
            min(W, x2 + pad), min(H, y2 + pad),
        ))
        result = rembg_tight(cropped, _parse_dark_threshold(dark_threshold))
        saved = save_output(result, product_id.strip())

        return JSONResponse({"preview": to_b64(preview), "output": to_b64(result), "saved": saved})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/extract/manual")
async def extract_manual(
    file: UploadFile = File(...),
    product_id: str = Form(""),
    box: str = Form("[]"),
    dark_threshold: str = Form(""),
):
    try:
        pil_img = Image.open(io.BytesIO(await file.read()))
        coords = json.loads(box)
        if len(coords) == 4:
            x1, y1, x2, y2 = (int(c) for c in coords)
            W, H = pil_img.size
            pil_img = pil_img.crop((max(0, x1), max(0, y1), min(W, x2), min(H, y2)))
        result = rembg_tight(pil_img, _parse_dark_threshold(dark_threshold))
        saved = save_output(result, product_id.strip())

        return JSONResponse({"output": to_b64(result), "saved": saved})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/extract/sam2")
async def extract_sam2(
    file: UploadFile = File(...),
    box: str = Form("[]"),
    points: str = Form("[]"),
    product_id: str = Form(""),
):
    try:
        coords = json.loads(box)
        if len(coords) != 4:
            return JSONResponse({"error": "アクセサリーを囲む矩形をドラッグで描いてください"}, status_code=400)

        pts = json.loads(points) or None
        pil_img = Image.open(io.BytesIO(await file.read()))
        result, preview = sam2_extract(pil_img, coords, pts)
        saved = save_output(result, product_id.strip())

        return JSONResponse({"output": to_b64(result), "preview": to_b64(preview), "saved": saved})
    except ImportError:
        return JSONResponse({"error": "SAM2がインストールされていません。pip install sam2 を実行してください"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
