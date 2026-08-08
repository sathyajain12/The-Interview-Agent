"""In-process session store.

Persistent accounts and long-term history are explicitly out of scope, so
sessions live in memory with a TTL and a bounded size. A thread lock keeps
concurrent turns on the same session serialised - uvicorn runs the sync
endpoint in a thread pool, so two requests for one sessionId genuinely can
overlap.
"""

from __future__ import annotations

import threading
import time
from typing import Iterator

from .config import settings
from .interviewer import Interview


class SessionStore:
    def __init__(self, ttl: int, max_sessions: int) -> None:
        self._ttl = ttl
        self._max = max_sessions
        self._items: dict[str, Interview] = {}
        self._lock = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def lock_for(self, session_id: str) -> threading.Lock:
        with self._lock:
            return self._locks.setdefault(session_id, threading.Lock())

    def get(self, session_id: str) -> Interview | None:
        with self._lock:
            self._evict_locked()
            return self._items.get(session_id)

    def put(self, session_id: str, interview: Interview) -> None:
        with self._lock:
            self._items[session_id] = interview
            self._evict_locked()

    def drop(self, session_id: str) -> bool:
        with self._lock:
            self._locks.pop(session_id, None)
            return self._items.pop(session_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __iter__(self) -> Iterator[tuple[str, Interview]]:
        with self._lock:
            return iter(list(self._items.items()))

    def _evict_locked(self) -> None:
        now = time.time()
        stale = [k for k, v in self._items.items() if now - v.updated_at > self._ttl]
        for key in stale:
            self._items.pop(key, None)
            self._locks.pop(key, None)

        overflow = len(self._items) - self._max
        if overflow > 0:
            oldest = sorted(self._items.items(), key=lambda kv: kv[1].updated_at)[:overflow]
            for key, _ in oldest:
                self._items.pop(key, None)
                self._locks.pop(key, None)


store = SessionStore(settings.session_ttl_seconds, settings.max_sessions)
