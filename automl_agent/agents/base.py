from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List


@dataclass
class AgentEvent:
    agent: str
    message: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BaseAgent:
    """Minimal agent base with structured event logging."""

    name = "Base Agent"

    def __init__(self) -> None:
        self.events: List[AgentEvent] = []

    def log(self, message: str) -> None:
        self.events.append(AgentEvent(agent=self.name, message=message))

