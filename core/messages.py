# Agent/core/messages.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import time, uuid

# 统一的“任务/消息”头
@dataclass
class Msg:
    topic: str                    # 主题，如 "detector.request"
    job_id: str                   # 工单 id（贯穿全流程）
    corr_id: str = field(default_factory=lambda: str(uuid.uuid4()))   # 消息相关 id
    sender: str = "unknown"
    payload: Dict[str, Any] = field(default_factory=dict)
    refs: Dict[str, Any] = field(default_factory=dict)   # 路径/URL/中间件 key
    confidence: float = 1.0
    ts: float = field(default_factory=time.time)
    error: Optional[str] = None

# 常用主题枚举（建议集中管理，避免硬编码）
TOPICS = {
    "INTENT_REQ": "intent.request",
    "INTENT_RES": "intent.result",

    "DETECT_REQ": "detector.request",
    "DETECT_RES": "detector.result",

    "GROUND_REQ": "grounder.request",
    "GROUND_RES": "grounder.result",

    "SPEC_REQ":   "specinfer.request",
    "SPEC_RES":   "specinfer.result",

    "MERGE_REQ":  "merge.request",
    "MERGE_RES":  "merge.result",

    "DESIGN_REQ": "designer.request",
    "DESIGN_RES": "designer.result",
    "REFINE_REQ": "designer.refine",

    "PROMPT_REQ": "prompt.request",
    "PROMPT_RES": "prompt.result",

    "GEN_REQ":    "generator.request",
    "GEN_RES":    "generator.result",

    "REVIEW_REQ": "reviewer.request",
    "REVIEW_RES": "reviewer.result",

    "ARBITER_REQ":"arbiter.request",
    "ARBITER_RES":"arbiter.result",

    "VECTOR_REQ": "vectorizer.request",
    "VECTOR_RES": "vectorizer.result",

    "DONE":       "pipeline.done",
    "ERROR":      "pipeline.error",

    "REVIEW_STRUCT_REQ": "reviewer.structure.request",
    "REVIEW_AESTH_REQ":  "reviewer.aesthetic.request",
    "REVIEW_RES":        "reviewer.result",
}
