# -*- coding: utf-8 -*-
# SymbolGeneration/Agent/agents/vectorizer_agent.py
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

# ===== 新增：优先尝试 Python 绑定的 vtracer =====
try:
    import vtracer as _vtracer
    _HAS_VTRACER_PY = True
except Exception:
    _HAS_VTRACER_PY = False
# =================================================


def _run_cli(cmd: List[str]) -> bool:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False


def _estimate_bg_mask_by_border(img_bgr: np.ndarray, tol: int = 28) -> np.ndarray:
    """
    用四条边像素估计背景颜色，计算颜色距离 < tol 的像素视为背景。
    返回 0/255 掩码（255=背景）。
    """
    h, w = img_bgr.shape[:2]
    border = np.concatenate([
        img_bgr[0, :, :], img_bgr[-1, :, :],
        img_bgr[:, 0, :], img_bgr[:, -1, :]
    ], axis=0).reshape(-1, 3).astype(np.float32)
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(img_bgr.astype(np.float32) - bg[None, None, :], axis=2)
    mask = (dist < tol).astype(np.uint8) * 255
    return mask


def _prep_no_bg_png(src_png: Path, tol: int = 28) -> Path:
    """
    生成一个临时 RGBA PNG：将估计的背景像素 alpha 置 0。
    """
    img = cv2.imread(str(src_png), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(src_png)
    bgmask = _estimate_bg_mask_by_border(img, tol=tol)  # 255=BG
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[bgmask == 255, 3] = 0
    tmp = src_png.with_name(src_png.stem + "_nobg.png")
    cv2.imwrite(str(tmp), rgba)
    return tmp


def _to_pgm_for_potrace(png_path: Path, pgm_path: Path, threshold: int) -> None:
    # 对透明背景进行合成：透明处置白，以免被当作主体
    img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {png_path}")
    if img.shape[-1] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        bg = np.full_like(bgr, 255)
        bgr = np.where(alpha[..., None] > 0, bgr, bg)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(str(pgm_path), bw)


def _hex_color_ok(c: str, default: str) -> str:
    if isinstance(c, str) and len(c) == 7 and c.startswith("#"):
        return c
    return default


# ===== 新增：Python 绑定 vtracer 的封装 =====
def _try_vtracer_py(png_path: Path, out_svg: Path) -> bool:
    """
    使用 Python 版 vtracer 进行多色分层矢量化。
    参照 vtracer_py README 的 convert_image_to_svg_py 参数。
    """
    if not _HAS_VTRACER_PY:
        return False
    try:
        _vtracer.convert_image_to_svg_py(
            str(png_path),
            str(out_svg),
            colormode='color',        # 或 'binary'（单色更快）
            hierarchical='stacked',   # 'stacked' 或 'cutout'
            mode='spline',            # 'spline' / 'polygon' / 'none'
            filter_speckle=4,
            color_precision=6,
            layer_difference=16,
            corner_threshold=60,
            length_threshold=4.0,
            max_iterations=10,
            splice_threshold=45,
            path_precision=6
        )
        return out_svg.exists() and out_svg.stat().st_size > 0
    except Exception:
        return False
# ==================================================


def png_to_svg(
    input_png: str,
    out_svg: Optional[str] = None,
    method: str = "auto",                # "auto"|"vtracer"|"potrace"|"opencv"
    threshold: int = 180,                # binarization threshold for potrace/opencv
    simplify_eps: float = 1.2,           # polygon simplification epsilon (px)
    fill_color: str = "#0B3D91",
    stroke_color: str = "#0B3D91",
    stroke_width: float = 1.0,
    remove_background: bool = True,      # 先剔除大背景
    bg_tolerance: int = 28               # 背景相似度阈值
) -> str:
    """
    PNG → SVG。优先级：
    1) vtracer (Python 绑定) → 2) vtracer CLI → 3) potrace → 4) OpenCV 兜底。
    """
    inp = Path(input_png)
    if not inp.exists():
        raise FileNotFoundError(f"[Vectorizer] PNG not found: {inp}")

    out = Path(out_svg) if out_svg else inp.with_suffix(".svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    fill_color = _hex_color_ok(fill_color, "#0B3D91")
    stroke_color = _hex_color_ok(stroke_color, "#0B3D91")

    src_for_trace = inp
    if remove_background:
        try:
            src_for_trace = _prep_no_bg_png(inp, tol=bg_tolerance)
        except Exception:
            src_for_trace = inp  # 失败则直接用原图

    # 1) vtracer (Python 绑定) —— 你已经通过 pip 安装了这个
    if method in ("auto", "vtracer"):
        if _try_vtracer_py(src_for_trace, out):
            return str(out)
        # 1b) vtracer CLI（系统 PATH 有可执行文件时再试）
        vtracer_cli = shutil.which("vtracer")
        if vtracer_cli:
            ok = _run_cli([
                vtracer_cli,
                "--mode", "spline",
                "--color_precision", "2",
                "--filter_speckle",
                "--hierarchical",
                "-o", str(out),
                str(src_for_trace)
            ])
            if ok and out.exists():
                return str(out)
            if method == "vtracer":
                raise RuntimeError("vtracer failed.")

    # 2) potrace（如果安装了 CLI）
    if method in ("auto", "potrace"):
        potrace = shutil.which("potrace")
        if potrace:
            pgm = inp.with_suffix(".pgm")
            _to_pgm_for_potrace(src_for_trace, pgm, threshold=threshold)
            ok = _run_cli([
                potrace,
                str(pgm),
                "-s",
                "-o", str(out),
                "--flat",
                "--longcoding"
            ])
            try:
                pgm.unlink(missing_ok=True)
            except Exception:
                pass
            if ok and out.exists():
                return str(out)
            if method == "potrace":
                raise RuntimeError("potrace failed.")

    # 3) OpenCV 兜底（单色路径）
    img = cv2.imread(str(src_for_trace), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Cannot read PNG for OpenCV fallback.")
    if img.shape[-1] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        # 透明视为背景 → 直接置白
        bgr = np.where(alpha[..., None] > 0, bgr, np.full_like(bgr, 255))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    def _path_d(cnt: np.ndarray) -> str:
        cnt = cnt.reshape(-1, 2)
        approx = cv2.approxPolyDP(cnt, epsilon=simplify_eps, closed=True).reshape(-1, 2)
        if approx.shape[0] == 0:
            return ""
        cmds = [f"M{approx[0,0]} {approx[0,1]}"]
        for i in range(1, approx.shape[0]):
            x, y = approx[i]
            cmds.append(f"L{x} {y}")
        cmds.append("Z")
        return " ".join(cmds)

    h, w = bw.shape[:2]
    paths = []
    for cnt in contours:
        d = _path_d(cnt)
        if not d:
            continue
        paths.append(
            f'<path d="{d}" fill="{fill_color}" stroke="{stroke_color}" stroke-width="{stroke_width}"/>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">' + "".join(paths) + "</svg>"
    )
    out.write_text(svg, encoding="utf-8")
    return str(out)
