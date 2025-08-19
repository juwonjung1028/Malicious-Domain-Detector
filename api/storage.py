from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, Any, List


class JsonlStore:
    """
    간단한 JSONL 로깅 스토어.
    - append(): id(uuid) 자동 부여
    - tail(): 마지막 N개 로드
    - delete_by_id(): 단일 항목 삭제
    - clear(): 전체 삭제
    """
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def append(self, item: Dict[str, Any]):
        # 신규 항목에는 id 자동 부여
        item.setdefault("id", uuid.uuid4().hex)
        line = json.dumps(item, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def tail(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            with self.path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return []
        tail_lines = lines[-limit:]
        out: List[Dict[str, Any]] = []
        for x in tail_lines:
            s = x.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                out.append(obj)
            except Exception:
                # 손상된 라인 무시
                continue
        return out

    def clear(self) -> None:
        self.path.write_text("", encoding="utf-8")

    def delete_by_id(self, _id: str) -> bool:
        try:
            with self.path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return False

        found = False
        kept: List[str] = []
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                kept.append(ln)
                continue
            if str(obj.get("id")) == str(_id):
                found = True
                continue
            kept.append(ln)

        if found:
            with self.path.open("w", encoding="utf-8") as f:
                f.write("".join(kept))
        return found
