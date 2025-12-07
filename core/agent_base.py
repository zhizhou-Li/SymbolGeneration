# Agent/core/agent_base.py
import asyncio, traceback
from typing import List, Callable
from .messages import Msg
from .blackboard import Blackboard

class Agent:
    def __init__(self, name: str, bb: Blackboard, subscriptions: List[str]):
        self.name = name
        self.bb = bb
        self.subscriptions = subscriptions

    async def start(self):
        tasks = [asyncio.create_task(self._consume(topic)) for topic in self.subscriptions]
        await asyncio.gather(*tasks)

    async def _consume(self, topic: str):
        async for msg in self.bb.subscribe(topic):
            try:
                await self.handle(msg)
            except Exception as e:
                tb = traceback.format_exc()
                await self.bb.publish(Msg(topic="pipeline.error", job_id=msg.job_id,
                                          sender=self.name, payload={"err": str(e), "trace": tb}))

    async def handle(self, msg: Msg):
        """子类实现：处理消息并 publish 新消息"""
        raise NotImplementedError
