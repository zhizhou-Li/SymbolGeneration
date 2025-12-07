# -*- coding: utf-8 -*-
"""
SymbolGeneration/Agent/run_experiments.py

一键批量运行实验：
- Baseline: 单步 run_generator（不做多智能体协作/多轮评审）
- Multi-Agent: run_micromap_experiment（完整框架）

用法：
    cd SymbolGeneration
    # 先配置好 OPENAI_API_KEY
    python -m Agent.run_experiments
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List

from SymbolGeneration.Agent.agents.generator_agent import run_generator
from SymbolGeneration.Agent.orchestrator import run_micromap_experiment

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = OUT_DIR / "experiment_results.json"

# ====== 实验样例（先给你示范几条，按论文需要自行扩展到 20–40 条） ======
EXPERIMENTS: List[Dict[str, Any]] = [
    {
        "id": "LANZHOUZHONGSHAN",
        "text": "生成具有艺术化风格的兰州中山桥图标，要求结构可辨、黑白二值化、线条均匀、留白均衡"
    },
    {
        "id": "BAITAMOUTION",
        "text": "生成具有艺术化风格的兰州的白塔山图标，要求结构可辨、黑白二值化、线条均匀、留白均衡"
    },
    {
        "id": "HUANGHEMOTHER",
        "text": "生成具有艺术化风格的兰州的黄河母亲雕塑图标，要求结构可辨、黑白二值化、线条均匀、留白均衡"
    },
    {
        "id": "SHANGZI",
        "text": "生成具有艺术化风格的商丘市的商字图标，要求结构可辨、黑白二值化、线条均匀、留白均衡"
    },
    # TODO: 这里继续补充你论文中设计的其他类别
]

def run_baseline(user_text: str) -> str:
    """
    单步基线方法：
    - 不做结构推断 / 评审迭代
    - 用极简 style_json + user_text 调用 run_generator
    - 返回选择的基线 PNG 路径
    """
    candidates = run_generator(
        outline_path=None,
        style_json="{}",          # 交给 prompt_planner 做最基础规划
        user_text=user_text,
        structure_spec=None,
        base_image=None,
        mask_image=None,
    )
    if not candidates:
        raise RuntimeError("Baseline generation failed: no images returned")
    # 如需更严谨，可以在此加 Reviewer 挑最优，这里先取第一张保证流程简单可复现
    return candidates[0]

def main():
    all_results: List[Dict[str, Any]] = []

    for item in EXPERIMENTS:
        exp_id = item["id"]
        text = item["text"]

        print("\n" + "=" * 80)
        print(f"🧪 实验样例: {exp_id}")
        print(f"说明: {text}")

        # 1) Baseline
        try:
            baseline_png = run_baseline(text)
            print(f"✅ Baseline 完成: {baseline_png}")
        except Exception as e:
            print(f"⚠️ Baseline 失败: {e}")
            baseline_png = None

        # 2) Multi-Agent 完整框架
        try:
            full_res = run_micromap_experiment(
                image_path=None,
                user_text=text,
                user_structure_spec=None,
                max_rounds=3,
                force_entity_type=None,
            )
            print(f"✅ Multi-Agent 完成: best_png={full_res.get('best_png')}")
        except Exception as e:
            print(f"⚠️ Multi-Agent 流程失败: {e}")
            full_res = None

        all_results.append({
            "id": exp_id,
            "text": text,
            "baseline_png": baseline_png,
            "multi_agent": full_res,
        })

    # 写入 JSON，供后续 CLIP / 统计分析使用
    RESULT_PATH.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print("\n✅ 所有实验完成，结果已保存到:", RESULT_PATH)

if __name__ == "__main__":
    main()
