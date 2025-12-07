# -*- coding: utf-8 -*-
# SymbolGeneration/Agent/agents/generator_agent.py
import base64
import time
from pathlib import Path
from typing import List, Optional

import requests
from openai import OpenAI
from ..config import MODELS, OPENAI_API_KEY, IMAGE_SIZE, CREATIVE_SAMPLES
from ..utils import log
from .prompt_planner import compile_prompt
from PIL import Image

client = OpenAI(api_key=OPENAI_API_KEY)
SUPPORTED_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}


def _download_with_retry(url: str, out_path: Path, tries: int = 3, timeout: int = 20) -> bool:
    for _ in range(tries):
        try:
            r = requests.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            out_path.write_bytes(r.content)
            return True
        except Exception:
            pass
    return False


def run_generator(outline_path: Optional[str],
                  style_json: str,
                  user_text: str = "",
                  structure_spec=None,
                  base_image: Optional[str] = None,   # ← 新增，可选
                  mask_image: Optional[str] = None    # ← 新增，可选
                  ) -> List[str]:
    """
    生成器（兼容原有调用）。
    - 若传入 base_image+mask_image，则优先尝试 images.edits（蒙版编辑）；
      否则回退 images.generate（纯文本）。
    - 输出：本地 PNG 路径列表。
    """
    size = IMAGE_SIZE if IMAGE_SIZE in SUPPORTED_SIZES else "1024x1024"
    n_samples = max(1, int(CREATIVE_SAMPLES))

    OUT_DIR = (Path(__file__).resolve().parents[1] / "outputs")
    IMG_DIR = OUT_DIR / "images"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d-%H%M%S")
    saved: List[str] = []

    for i in range(n_samples):
        variation = f"Encourage variation #{i+1}: explore composition/texture diversity while preserving recognizability."
        prompt = compile_prompt(
            user_text=user_text,
            style_json=style_json,
            structure_spec=structure_spec,
            variation_note=variation
        )

        # 记录提示词
        (OUT_DIR / f"IconGenerator_prompt_{ts}_{i + 1}.txt").write_text(prompt, encoding="utf-8")
        out_path = IMG_DIR / f"candidate_{ts}_{i + 1}.png"

        resp = None
        # —— 判断是否支持编辑接口
        supports_edits = hasattr(client.images, "edits") or hasattr(client.images, "edit")
        # 1) 如可编辑且传入了底图+蒙版，先试编辑；失败则回退纯生成
        if base_image and mask_image and supports_edits:
            try:
                # 兼容两种命名：edits / edit
                edits_call = getattr(client.images, "edits", None) or getattr(client.images, "edit", None)
                resp = edits_call(
                    model=MODELS["IMAGE_MODEL"],
                    image=open(base_image, "rb"),
                    mask=open(mask_image, "rb"),
                    prompt=prompt,
                    size=size,
                    n=1
                )
            except Exception as e:
                print(f"⚠️ images.edits 调用失败，将回退 generate：{e}")

        # 2) 首次或回退：纯生成
        if resp is None:
            try:
                resp = client.images.generate(
                    model=MODELS["IMAGE_MODEL"],
                    prompt=prompt,
                    size=size,
                    n=1
                )
            except Exception as e:
                print(f"⚠️ images.generate 失败：{e}")
                continue

        # 3) 保存输出（优先 b64，其次 URL）
        datum = getattr(resp, "data", [None])[0]
        b64 = getattr(datum, "b64_json", None)
        url = getattr(datum, "url", None)

        if isinstance(b64, str) and b64:
            out_path.write_bytes(base64.b64decode(b64))
            saved.append(str(out_path))
            print(f"🖼️ 已保存本地图片: {out_path}")
        elif isinstance(url, str) and url:
            if _download_with_retry(url, out_path):
                saved.append(str(out_path))
                print(f"🖼️ 已保存本地图片(回退URL): {out_path}")
            else:
                print("⚠️ URL 下载失败")
        else:
            print("⚠️ 无可用图像数据")

        time.sleep(0.15)

    if not saved:
        raise RuntimeError("Image API returned no usable images (b64/url).")
    log("IconGenerator", f"{len(saved)} local images\n" + "\n".join(saved))
    return saved
