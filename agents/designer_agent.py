# -*- coding: utf-8 -*-
# SymbolGeneration/Agent/agents/designer_agent.py
from openai import OpenAI
from ..config import MODELS, OPENAI_API_KEY
from ..utils import log, save_json, extract_json
import json

client = OpenAI(api_key=OPENAI_API_KEY)

STYLE_SCHEMA_HINT = """
Output ONLY a JSON object. Include fields like:
{
  "style_name": "string",
  "stroke": {"width": number, "pattern": "solid|dash", "corner": "round|miter"},
  "fill": {"type": "none|flat|gradient", "opacity": 0-1},
  "palette": ["#RRGGBB", "..."],
  "simplification": {"tolerance_px": number, "max_points": number},
  "iconography": {"emphasis": ["outline|verticality|truss_structure|polygonal_top_chord"], "negative_space": true/false},
  "export": {"size": 512, "background": "transparent|white"}
}
"""

def _sanitize_style_json(style_str: str, structure_spec) -> str:
    try:
        sj = extract_json(style_str) or {}
        icon = sj.get("iconography") or {}
        emph = list(dict.fromkeys(icon.get("emphasis") or []))

        sys = ""
        if isinstance(structure_spec, dict):
            sys = (structure_spec.get("structural_system") or structure_spec.get("superstructure") or "").lower()
            if structure_spec.get("top_chord_profile") in ("polygonal","camelback"):
                if "polygonal_top_chord" not in emph:
                    emph.append("polygonal_top_chord")

        # 如果桁架体系：剔除 arches，添加 truss_structure
        if sys == "truss":
            emph = [e for e in emph if e.lower() != "arches"]
            if "truss_structure" not in (x.lower() for x in emph):
                emph.append("truss_structure")

        icon["emphasis"] = emph
        sj["iconography"] = icon

        # 强制二色调
        pal = sj.get("palette") or []
        pal = pal[:4] if len(pal) >= 4 else ["#2E4A62", "#AAB7C4","#D9D9D9","#E3E7EC"]
        sj["palette"] = pal

        # 填充透明度限制
        fill = sj.get("fill") or {}
        try:
            op = float(fill.get("opacity", 1))
        except Exception:
            op = 1.0
        fill["opacity"] = max(0.0, min(1.0, op))
        sj["fill"] = fill

        return json.dumps(sj, ensure_ascii=False)
    except Exception:
        return style_str

def run_designer(landmark_json: str, schema: str, structure_spec=None) -> str:
    spec_text = json.dumps(structure_spec, ensure_ascii=False) if structure_spec else "{}"
    resp = client.chat.completions.create(
        model=MODELS["LLM_MODEL"],
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system",
             "content": "You are a cartographic symbol designer that outputs ONLY JSON style sheets."},
            {"role": "user",
             "content": (
                 f"Landmark (optional):\n{landmark_json}\n\n"
                 f"Interpreter schema (optional):\n{schema}\n\n"
                 f"Structural constraints (STRICT):\n{spec_text}\n"
                 f"If structural_system='truss', DO NOT emphasize 'arches'; "
                 f"emphasize 'truss_structure' and optionally 'polygonal_top_chord' instead.\n\n"
                 f"{STYLE_SCHEMA_HINT}"
             )}
        ]
    )
    content = resp.choices[0].message.content
    log("SymbolDesigner_raw", content)
    content = _sanitize_style_json(content, structure_spec)
    save_json("SymbolDesigner_json", extract_json(content) or {})
    return content

def refine_designer(prev_style_json: str, review_data: dict, structure_spec=None) -> str:
    spec_text = json.dumps(structure_spec, ensure_ascii=False) if structure_spec else "{}"
    resp = client.chat.completions.create(
        model=MODELS["LLM_MODEL"],
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system",
             "content": "You are a cartographic symbol designer. Refine the JSON style to fix issues while preserving recognizability. Output ONLY JSON."},
            {"role": "user", "content": f"Previous style JSON:\n{prev_style_json}"},
            {"role": "assistant", "content": f"Reviewer feedback JSON:\n{review_data}\n"
                                             f"Structural constraints (STRICT):\n{spec_text}\n"
                                             f"If structural_system='truss', remove 'arches' from emphasis and add 'truss_structure'."},
            {"role": "user", "content": STYLE_SCHEMA_HINT}
        ]
    )
    content = resp.choices[0].message.content
    log("SymbolDesigner_refined_raw", content)
    content = _sanitize_style_json(content, structure_spec)
    save_json("SymbolDesigner_refined_json", extract_json(content) or {})
    return content
