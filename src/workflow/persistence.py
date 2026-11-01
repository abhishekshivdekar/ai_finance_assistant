"""Where conversations are kept: a SQLite file (survives a restart) or, when that is not possible, memory.

Design patterns:
- FACTORY with a safe FALLBACK: `open_store()` returns a ConversationStore that is ALWAYS usable. Example: the file is
  corrupt and cannot be renamed because another program holds it (Windows) -> the store is built on LangGraph's
  in-memory MemorySaver, the app starts, and `store.status.message` tells the user conversations are not saved this run.
- SINGLETON per process (the facade caches the store) with one connection shared by all Streamlit threads.

Thread safety: the connection is opened with check_same_thread=False because every Streamlit rerun may run on another
thread. SqliteSaver serialises its own statements with an internal lock, and the activity table below uses the SAME
lock (`saver.cursor()`), so no two statements overlap. Turns of one conversation are ordered by the facade's per-thread lock.

Privacy: nothing here logs a thread id or a message; only counts and statuses are logged. The `thread_activity` table
(thread_id, last_seen) is ours; ages are never read from the library's own tables.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from src.utils.config import PROJECT_ROOT, get_config
from src.utils.logger import get_logger

log = get_logger(__name__)
SECONDS_PER_DAY = 86_400
SIDE_FILES = ("", "-wal", "-shm")          # the database and its write-ahead log files


@dataclass(frozen=True)
class PersistenceStatus:
    """Whether conversations are saved to disk in this run, and the sentence the sidebar shows (from config)."""

    saving: bool
    message: str


class ConversationStore:
    """A checkpointer plus the bookkeeping for retention and deletion (when was each thread last used).

    Example: after a turn the facade calls touch("web-ab12..."); at start-up cleanup(30, 200) deletes threads unused for
    30 days and, beyond 200 threads, the least recently used ones.
    """

    def __init__(self, saver: Any, status: PersistenceStatus, conn: sqlite3.Connection | None = None) -> None:
        self.saver, self.status, self._conn = saver, status, conn
        self._memory_activity: dict[str, float] = {}

    def touch(self, thread_id: str, now: float | None = None) -> None:
        """Remember that this thread was used just now."""
        stamp = time.time() if now is None else now
        if self._conn is None:
            self._memory_activity[thread_id] = stamp
            return
        with self.saver.cursor() as cur:
            cur.execute("INSERT OR REPLACE INTO thread_activity (thread_id, last_seen) VALUES (?, ?)", (thread_id, stamp))

    def _activity(self) -> list[tuple[str, float]]:
        """(thread_id, last_seen) of every known thread, least recently used first."""
        if self._conn is None:
            return sorted(self._memory_activity.items(), key=lambda item: item[1])
        with self.saver.cursor(transaction=False) as cur:
            cur.execute("SELECT thread_id, last_seen FROM thread_activity ORDER BY last_seen")
            return [(str(row[0]), float(row[1])) for row in cur.fetchall()]

    def delete(self, thread_id: str) -> None:
        """Remove the thread's checkpoints and writes and its activity row (the rows are really deleted)."""
        self.saver.delete_thread(thread_id)
        if self._conn is None:
            self._memory_activity.pop(thread_id, None)
            return
        with self.saver.cursor() as cur:
            cur.execute("DELETE FROM thread_activity WHERE thread_id = ?", (thread_id,))

    def cleanup(self, max_age_days: int, max_threads: int, now: float | None = None) -> int:
        """Delete threads unused for `max_age_days`, then the least recently used ones beyond `max_threads`. Returns how many."""
        cutoff = (time.time() if now is None else now) - max_age_days * SECONDS_PER_DAY
        kept = self._activity()
        doomed = [thread for thread, seen in kept if seen < cutoff]
        remaining = [thread for thread, seen in kept if seen >= cutoff]
        doomed += remaining[: max(0, len(remaining) - max_threads)]
        for thread in doomed:
            self.delete(thread)
        if doomed:
            log.info("Retention removed %d conversation(s)", len(doomed))
        return len(doomed)

    def thread_count(self) -> int:
        """How many threads are known (for tests and the log)."""
        return len(self._activity())

    def close(self) -> None:
        """Close the file (called once when the process exits). Safe to call twice."""
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:                               # already closed or broken: nothing left to release
                log.debug("Closing the conversation file failed", exc_info=True)


def memory_store(message: str) -> ConversationStore:
    """A store on the in-memory MemorySaver (tests, saving switched off, or the file could not be used)."""
    return ConversationStore(MemorySaver(), PersistenceStatus(False, message))


def _open_sqlite(path: Path, settings: dict[str, Any], message: str) -> ConversationStore:
    """Open (or create) the file, create the tables and check the file; any failure closes the connection and raises."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=float(settings["busy_timeout_seconds"]))
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        conn.execute("CREATE TABLE IF NOT EXISTS thread_activity (thread_id TEXT PRIMARY KEY, last_seen REAL NOT NULL)")
        conn.commit()
        if conn.execute("PRAGMA quick_check(1)").fetchone()[0] != "ok":
            raise sqlite3.DatabaseError("quick_check failed")
    except BaseException:
        conn.close()
        raise
    return ConversationStore(saver, PersistenceStatus(True, message), conn)


def _set_aside(path: Path) -> None:
    """Rename the damaged file (and its -wal / -shm files) to *.corrupt-<time>; raises OSError when a file is in use."""
    stamp = int(time.time())
    for suffix in SIDE_FILES:
        source = path.with_name(path.name + suffix)
        if source.exists():
            source.replace(path.with_name(f"{path.name}.corrupt-{stamp}{suffix}"))


def open_store(settings: dict[str, Any] | None = None) -> ConversationStore:
    """The conversation store for this run. Never raises: every failure ends in a memory store with an explanatory status.

    - saving switched off -> memory;  file missing -> created;  file locked, unwritable or the folder missing -> memory;
    - file corrupt -> renamed aside and a new file started; if the rename fails (Windows: the file is in use) -> memory.
    """
    settings = get_config()["persistence"] if settings is None else settings
    messages = settings["messages"]
    if not settings["enabled"]:
        return memory_store(messages["off"])
    path = Path(settings["path"])
    path = path if path.is_absolute() else PROJECT_ROOT / path
    try:
        return _open_sqlite(path, settings, messages["saving"])
    except sqlite3.OperationalError:
        log.warning("The conversation file is locked or cannot be opened; conversations stay in memory this run")
        return memory_store(messages["unavailable"])
    except sqlite3.DatabaseError:
        log.warning("The conversation file is damaged; setting it aside")
    except OSError:
        log.warning("The conversation folder cannot be used; conversations stay in memory this run")
        return memory_store(messages["unavailable"])
    try:
        _set_aside(path)
    except OSError:
        log.warning("The damaged conversation file could not be renamed; conversations stay in memory this run")
        return memory_store(messages["corrupt_locked"])
    try:
        return _open_sqlite(path, settings, messages["recovered"])
    except (sqlite3.Error, OSError):
        log.warning("A new conversation file could not be created; conversations stay in memory this run")
        return memory_store(messages["unavailable"])
