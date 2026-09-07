from typing import List, Dict, Any
from datetime import datetime, timezone
import uuid


class InMemoryHistoryStore:
    def __init__(self):
        self._items: List[Dict[str, Any]] = []

    def add(
        self,
        filename: str,
        predicted_class: str,
        confidence: float,
        probabilities: Dict[str, float],
    ) -> Dict[str, Any]:
        item = {
            "id": str(uuid.uuid4()),
            "filename": filename,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._items.insert(0, item)
        return item

    def list(self, limit: int = 20):
        return self._items[:limit]

    def clear(self):
        self._items = []


history_store = InMemoryHistoryStore()