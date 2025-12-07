# Agent/core/planner_agent.py
import asyncio
from .agent_base import Agent
from .messages import Msg, TOPICS

class PlannerAgent(Agent):
    def __init__(self, bb, max_rounds=3):
        super().__init__("Planner", bb, [TOPICS["INTENT_REQ"], TOPICS["ARBITER_RES"]])
        self.state = {}     # job_id -> {"round":1, "style_json":..., "spec":..., ...}
        self.max_rounds = max_rounds

    async def handle(self, msg: Msg):
        if msg.topic == TOPICS["INTENT_REQ"]:
            await self._kickoff(msg)
        elif msg.topic == TOPICS["ARBITER_RES"]:
            await self._decide_next(msg)

    async def _kickoff(self, msg: Msg):
        j = msg.job_id
        user_text = msg.payload["user_text"]
        image_path = msg.payload.get("image_path")

        # ① 并行：Grounder + SpecInfer（有图才发 Detector）
        print("[Planner] → publish GROUND_REQ / SPEC_REQ", flush=True)
        await self.bb.publish(Msg(topic=TOPICS["GROUND_REQ"], job_id=j, sender=self.name,
                                  payload={"user_text": user_text}))
        await self.bb.publish(Msg(topic=TOPICS["SPEC_REQ"], job_id=j, sender=self.name,
                                  payload={"user_text": user_text}))

        detect_msg = None
        if image_path:
            print("[Planner] → publish DETECT_REQ", flush=True)
            await self.bb.publish(Msg(topic=TOPICS["DETECT_REQ"], job_id=j, sender=self.name,
                                      payload={"image_path": image_path, "schema": "{\"kind\":\"landmark\"}"}))

        # ② 等结果（Detector 可选 + 有超时）
        ground = await self._await_one(j, TOPICS["GROUND_RES"], label="GROUND_RES")
        spec = await self._await_one(j, TOPICS["SPEC_RES"], label="SPEC_RES")

        detect = None
        if image_path:
            detect = await self._await_optional(j, TOPICS["DETECT_RES"], timeout=5.0, label="DETECT_RES(optional)")

        # ③ 合并规范
        print("[Planner] → publish MERGE_REQ", flush=True)
        await self.bb.publish(Msg(topic=TOPICS["MERGE_REQ"], job_id=j, sender=self.name,
                                  payload={"user_spec": spec.payload.get("spec"),
                                           "detector_spec": (detect.payload.get("detector") if detect else {}),
                                           "defaults": ground.payload.get("grounded")}))

        merged = await self._await_one(j, TOPICS["MERGE_RES"], label="MERGE_RES")
        self.state[j] = {"round": 1, "spec": merged.payload["merged"]}

        # ④ 设计样式
        print("[Planner] → publish DESIGN_REQ", flush=True)
        await self.bb.publish(Msg(topic=TOPICS["DESIGN_REQ"], job_id=j, sender=self.name,
                                  payload={"detector_spec": (detect.payload.get("detector") if detect else "{}"),
                                           "schema": "{}", "structure_spec": merged.payload["merged"]}))
        style = await self._await_one(j, TOPICS["DESIGN_RES"], label="DESIGN_RES")
        self.state[j]["style_json"] = style.payload["style_json"]

        # ⑤ 生成候选
        print("[Planner] → publish GEN_REQ", flush=True)
        await self.bb.publish(Msg(topic=TOPICS["GEN_REQ"], job_id=j, sender=self.name,
                                  payload={"style_json": style.payload["style_json"],
                                           "user_text": user_text,
                                           "structure_spec": merged.payload["merged"]}))
        gen = await self._await_one(j, TOPICS["GEN_RES"], label="GEN_RES")
        best_png = gen.payload["best_png"]
        self.state[j]["best_png"] = best_png

        # ⑥ 并行两位审稿人
        print("[Planner] → REVIEW_STRUCT_REQ & REVIEW_AESTH_REQ")
        await self.bb.publish(Msg(topic=TOPICS["REVIEW_STRUCT_REQ"], job_id=j, sender=self.name,
                                  payload={"image_path": best_png, "structure_spec": merged.payload["merged"]}))
        await self.bb.publish(Msg(topic=TOPICS["REVIEW_AESTH_REQ"], job_id=j, sender=self.name,
                                  payload={"image_path": best_png, "structure_spec": merged.payload["merged"]}))

    async def _decide_next(self, msg: Msg):
        j = msg.job_id
        st = self.state.get(j, {"round": 1})
        decision = (msg.payload or {}).get("decision")
        fused = (msg.payload or {}).get("review") or {}

        # ====== 收敛：触发矢量化，再 DONE ======
        if decision == "stop" or st["round"] >= self.max_rounds:
            best_png = st.get("best_png")  # 在 _kickoff() / 生成阶段要把 best_png 存入 state
            svg_path = None
            if best_png:
                print("[Planner] → VECTOR_REQ", flush=True)
                await self.bb.publish(Msg(
                    topic=TOPICS["VECTOR_REQ"], job_id=j, sender=self.name,
                    payload={"png_path": best_png, "method": "auto", "simplify_eps": 1.0}
                ))
                vec = await self._await_one(j, TOPICS["VECTOR_RES"], label="VECTOR_RES")
                svg_path = (vec.payload or {}).get("svg_path")

            await self.bb.publish(Msg(
                topic=TOPICS["DONE"], job_id=j, sender=self.name,
                payload={"review": fused, "svg_path": svg_path}
            ))
            self.state.pop(j, None)
            return

        # ====== 继续细化：Designer → Generator → 双审稿人 ======
        st["round"] += 1
        await self.bb.publish(Msg(
            topic=TOPICS["REFINE_REQ"], job_id=j, sender=self.name,
            payload={
                "prev_style_json": st["style_json"],
                "review_json": fused,
                "structure_spec": st["spec"]
            }
        ))
        style = await self._await_one(j, TOPICS["DESIGN_RES"], label="DESIGN_RES")
        st["style_json"] = style.payload["style_json"]

        # 新一轮生成
        await self.bb.publish(Msg(
            topic=TOPICS["GEN_REQ"], job_id=j, sender=self.name,
            payload={
                "style_json": st["style_json"],
                "user_text": "reuse",
                "structure_spec": st["spec"]
            }
        ))
        gen = await self._await_one(j, TOPICS["GEN_RES"], label="GEN_RES")
        best_png = gen.payload["best_png"]
        st["best_png"] = best_png  # ← 记住给收敛时矢量化用

        # 并行两位审稿人（分开队列，避免互相吞消息）
        print("[Planner] → REVIEW_STRUCT_REQ & REVIEW_AESTH_REQ", flush=True)
        await self.bb.publish(Msg(
            topic=TOPICS["REVIEW_STRUCT_REQ"], job_id=j, sender=self.name,
            payload={"image_path": best_png, "structure_spec": st["spec"]}
        ))
        await self.bb.publish(Msg(
            topic=TOPICS["REVIEW_AESTH_REQ"], job_id=j, sender=self.name,
            payload={"image_path": best_png, "structure_spec": st["spec"]}
        ))

    async def _await_one(self, job_id: str, topic: str, timeout: float = 30.0, label: str = ""):
        import time, asyncio
        start = time.time()
        q = self.bb.topic(topic)
        while time.time() - start < timeout:
            msg = await asyncio.wait_for(q.get(), timeout=timeout)
            if msg.job_id == job_id:
                if label: print(f"[Planner] ✓ {label}", flush=True)
                return msg
            # 不是我这单的消息：丢回队尾，避免“吃掉别人”的消息
            await q.put(msg)
        raise TimeoutError(f"wait {topic} for job {job_id} timeout")

    async def _await_optional(self, job_id: str, topic: str, timeout: float = 2.0, label: str = ""):
        try:
            return await self._await_one(job_id, topic, timeout=timeout, label=label)
        except Exception:
            print(f"[Planner] • {label} not available, continue", flush=True)
            return None
