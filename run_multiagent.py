# -*- coding: utf-8 -*-
import asyncio, uuid
from Agent.core.blackboard import Blackboard
from Agent.core.messages import Msg, TOPICS

# —— 中枢 Agent ——
from Agent.core.planner_agent import PlannerAgent
from Agent.core.arbiter_agent import ArbiterAgent
from Agent.core.memory_agent import MemoryAgent

# —— 业务 Worker ——
from Agent.wrappers.grounder_worker import GrounderWorker
from Agent.wrappers.specinfer_worker import SpecInferWorker
from Agent.wrappers.merge_worker import MergeWorker
from Agent.wrappers.designer_worker import DesignerWorker
from Agent.wrappers.generator_worker import GeneratorWorker
from Agent.wrappers.reviewer_workers import StructureReviewer, AestheticReviewer
from Agent.wrappers.vectorizer_worker import VectorizerWorker
from Agent.wrappers.detector_worker import DetectorWorker  # 仅在传入 image 时才会用

async def _watch_errors(bb: Blackboard):
    async for e in bb.subscribe(TOPICS["ERROR"]):
        print("\n[PIPELINE.ERROR]", e.payload.get("err"))
        tr = e.payload.get("trace", "")
        if tr:
            print(tr[:2000])

async def _run_job(user_text: str, image_path: str | None = None, rounds: int = 3):
    bb = Blackboard()

    # 启动所有 Agent（并发）
    agents = [
        PlannerAgent(bb, max_rounds=rounds),
        ArbiterAgent(bb),
        MemoryAgent(bb),

        GrounderWorker(bb),
        SpecInferWorker(bb),
        MergeWorker(bb),
        DesignerWorker(bb),
        GeneratorWorker(bb),
        StructureReviewer(bb),
        AestheticReviewer(bb),
        VectorizerWorker(bb),
    ]
    # 仅当提供了图片时才启用 DetectorWorker
    if image_path:
        agents.append(DetectorWorker(bb))

    tasks = [asyncio.create_task(a.start()) for a in agents]
    tasks.append(asyncio.create_task(_watch_errors(bb)))

    # 发起一条工单
    job_id = str(uuid.uuid4())
    payload = {"user_text": user_text}
    if image_path:
        payload["image_path"] = image_path

    await bb.publish(Msg(topic=TOPICS["INTENT_REQ"], job_id=job_id, sender="CLI", payload=payload))

    # 等待 DONE（同一个 job_id）
    async for m in bb.subscribe(TOPICS["DONE"]):
        if m.job_id == job_id:
            print("\n✅ DONE")
            print("决策综评：", m.payload.get("review"))
            break

    # 优雅退出
    for t in tasks:
        t.cancel()

def run(user_text: str, image_path: str | None = None, rounds: int = 3):
    """对外同步入口，便于像 orchestrator 一样直接调用"""
    asyncio.run(_run_job(user_text, image_path, rounds))

if __name__ == "__main__":
    # ===== 像 orchestrator.py 一样在这里改默认参数 =====
    DEFAULT_TEXT = "生成具有艺术化风格的兰州中山桥图标，要求结构可辨、黑白二值化、线条均匀、留白均衡"
    DEFAULT_IMAGE = None
    # 例：照片→符号模式（需要就把上面置换掉）
    # DEFAULT_TEXT = "生成具有艺术化风格的兰州中山桥图标，要求结构可辨、二色调、留白均衡"
    # DEFAULT_IMAGE = r"Z:\python_projects\map_entropy\SymbolGeneration\images\Zhongshan_Bridge_in_Lanzhou.jpg"

    DEFAULT_ROUNDS = 1
    run(DEFAULT_TEXT, DEFAULT_IMAGE, DEFAULT_ROUNDS)
