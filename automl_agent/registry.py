from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


_MAX_REGISTRY_ENTRIES = 500


class ModelRegistry:
    """Append-only JSON registry for locally packaged model versions."""

    def __init__(self, registry_path: Path) -> None:
        self.registry_path = registry_path

    def register(self, entry: Dict[str, Any], max_entries: int = _MAX_REGISTRY_ENTRIES) -> List[Dict[str, Any]]:
        records = self.list()
        records.append(entry)
        if max_entries > 0 and len(records) > max_entries:
            records = records[-max_entries:]
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
        return records

    def list(self) -> List[Dict[str, Any]]:
        if not self.registry_path.exists():
            return []
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

