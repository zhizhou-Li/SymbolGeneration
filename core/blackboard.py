# Agent/core/blackboard.py
import asyncio
from typing import Dict, List, AsyncIterator
from .messages import Msg

class Blackboard:
    def __init__(self):
        self._topics: Dict[str, asyncio.Queue] = {}
        self._memory: Dict[str, dict] = {}   # 长期记忆键值对：如 entity_key -> {"best_style":..., "svg":...}

    def topic(self, name: str) -> asyncio.Queue:
        if name not in self._topics:
            self._topics[name] = asyncio.Queue()
        return self._topics[name]

    async def publish(self, msg: Msg):
        await self.topic(msg.topic).put(msg)

    async def subscribe(self, topic: str) -> AsyncIterator[Msg]:
        q = self.topic(topic)
        while True:
            yield await q.get()

    # —— 记忆读写（给 memory_agent 用）——
    def mem_get(self, key: str, default=None):
        return self._memory.get(key, default)

    def mem_set(self, key: str, value: dict):
        self._memory[key] = value
