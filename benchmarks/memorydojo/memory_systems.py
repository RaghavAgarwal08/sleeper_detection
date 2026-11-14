from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemorySystemType(str, Enum):
    LETTA_CORE = "letta"
    MEM0_FACTS = "mem0"
    AMEM_NOTES = "amem"
    GRAPHITI_GRAPH = "graphiti"
    MEMORA_EPISODIC = "memora"


@dataclass
class MemoryItem:
    item_id: str
    content: str
    session_id: int
    is_poison: bool = False
    is_twin: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class MemorySystemSimulator:
    def __init__(self, system_type: MemorySystemType):
        self.system_type = system_type
        self.store: list[MemoryItem] = []

    def write(self, session_idx: int, observations: list[str], is_poison: bool = False, is_twin: bool = False) -> list[MemoryItem]:
        new_items = []
        for i, obs in enumerate(observations):
            item = MemoryItem(
                item_id=f"mem_{session_idx}_{i}",
                content=obs,
                session_id=session_idx,
                is_poison=is_poison,
                is_twin=is_twin,
            )
            new_items.append(item)
            self.store.append(item)
        return new_items

    def retrieve(self, query: str, k: int = 5) -> list[MemoryItem]:
        if not self.store:
            return []
        if self.system_type == MemorySystemType.LETTA_CORE:
            return self.store[:k]
        elif self.system_type == MemorySystemType.MEM0_FACTS:
            return [m for m in self.store if any(w in m.content.lower() for w in query.lower().split())][:k]
        else:
            return self.store[-k:]
