from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List

class JsonlStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text('', encoding='utf-8')

    def append(self, item: Dict[str, Any]):
        line = json.dumps(item, ensure_ascii=False)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(line + '\n')

    def tail(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            with self.path.open('r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            return []
        tail_lines = lines[-limit:]
        return [json.loads(x) for x in tail_lines if x.strip()]