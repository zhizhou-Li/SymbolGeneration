# -*- coding: utf-8 -*-
# SymbolGeneration/Agent/agents/reviewer_agent.py
from __future__ import annotations
import base64, mimetypes
from pathlib import Path
from typing import Any, Dict, Optional
from openai import OpenAI

from ..config import MODELS, OPENAI_API_KEY
from ..utils import log, save_json, extract_json
from .spec_utils import json_to_constraints

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_MSG = (
    "You are a rigorous cartographic reviewer for micro-map icons. "
    "Return ONLY a JSON object with fields: "
    "{clarity_score:0-100, aesthetic_score:0-100, recognizability_score:0-100, "
    "structure_penalty:0-100, violations:[], suggestions:[]} "
    "If MUST structure (e.g., structural_system='truss') is violated (e.g., arch ribs, suspension towers, cables), "
    "set recognizability_score <= 10 and increase structure_penalty by >= 60."
)

def _to_image_content(source: str) -> Dict[str, Any]:
    if not source:
        return {"type": "text", "text": "(no image provided)"}
    if source.startswith("http") or source.startswith("data:"):
        return {"type": "image_url", "image_url": {"url": source}}
    p = Path(source)
    if not p.exists():
        return {"type": "text", "text": f"(image not found) {source}"}
    b = p.read_bytes()
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    b64 = base64.b64encode(b).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}

def run_reviewer(symbol_input: str, structure_spec: Optional[Dict[str, Any] | str] = None) -> Dict[str, Any]:
    spec_dict = structure_spec if isinstance(structure_spec, dict) else extract_json(structure_spec or "") or {}
    must, must_not = json_to_constraints(spec_dict)

    checklist = []
    if must:
        checklist.append("STRUCTURE MUST:\n" + "\n".join(f"- {m}" for m in must))
    if must_not:
        checklist.append("STRUCTURE MUST-NOT:\n" + "\n".join(f"- {x}" for x in must_not))
    checklist_text = "\n".join(checklist) if checklist else "STRUCTURE MUST: (none)\nSTRUCTURE MUST-NOT: (none)"

    content_img = _to_image_content(symbol_input)

    resp = client.chat.completions.create(
        model=MODELS["LLM_MODEL"],
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": [
                {"type": "text", "text": f"Review this icon.\n{checklist_text}"},
                content_img
            ]}
        ]
    )
    raw = resp.choices[0].message.content
    log("MapReviewer_raw", raw)
    data = extract_json(raw) or {
        "clarity_score": 0, "aesthetic_score": 0, "recognizability_score": 0,
        "structure_penalty": 100, "violations": ["parse_error"], "suggestions": []
    }
    save_json("MapReviewer", data)
    return data
