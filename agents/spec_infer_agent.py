# agents/spec_infer_agent.py
from __future__ import annotations
from typing import Any, Dict, Optional, List
from openai import OpenAI
from ..config import OPENAI_API_KEY, MODELS
from ..utils import save_json, log, extract_json

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_MSG = (
    "You are a universal spec planner for image generation. "
    "Turn user intent and optional detector context into a COMPACT JSON spec. "
    "Use short values, avoid prose. If uncertain, use 'unknown' or omit. "
    "For bridges as icons, prefer view='side_elevation' or 'isometric'. "
    "Distinguish structural system vs visual motif:\n"
    "- structural_system: truss|arch|suspension|cable_stayed|beam|frame|unknown\n"
    "- top_chord_profile: flat|polygonal|camelback|curved|unknown\n"
    "- arch_rib_presence: true|false\n"
    "Schema:\n"
    "{"
    ' "entity": {"name": str, "aliases": [str], "location": str},'
    ' "task_type": "engineering|art|concept|product|logo|other",'
    ' "view": "side_elevation|isometric|front|perspective|top|emblematic|unknown",'
    ' "style": {"palette":[str], "texture": str, "linework": str, "medium": str},'
    ' "constraints":{"must":[str], "must_not":[str]},'
    ' "structure": {"structural_system":"truss|arch|suspension|cable_stayed|beam|frame|unknown","top_chord_profile":"flat|polygonal|camelback|curved|unknown","arch_rib_presence":false,"material_hint":"steel|stone|concrete|wood|mixed|unknown","spans":null},'
    ' "priority":{"recognizability":"high|medium|low"}'
    "}"
)

def infer_structure_spec(user_text: str, detector_spec: Optional[str | Dict[str, Any]] = None) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": [
            {"type": "text", "text": f"User intent:\n{user_text}"},
            {"type": "text", "text": f"Optional detector context:\n{detector_spec}"} if detector_spec else {"type":"text","text":"(no detector context)"}
        ]}
    ]
    resp = client.chat.completions.create(
        model=MODELS["LLM_MODEL"],
        response_format={"type": "json_object"},
        temperature=0.0,
        messages=messages,
    )
    raw = resp.choices[0].message.content
    log("SpecInfer_raw", raw)
    spec = extract_json(raw) or {}
    save_json("SpecInfer", spec)
    return spec
