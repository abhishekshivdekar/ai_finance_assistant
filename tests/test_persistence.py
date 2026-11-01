"""Conversations that survive a restart (UI step 5a). Every test uses a temp-file SQLite or memory: no network, no LLM."""
from __future__ import annotations

import copy
import logging
import sqlite3
import time
from pathlib import Path

import pytest
import yaml
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from src.utils.config import ConfigError, get_config
from src.web_app import backend
from src.workflow import facade, nodes, persistence
from src.workflow.graph import build_graph
from src.workflow.persistence import ConversationStore, PersistenceStatus, memory_store, open_store
from src.workflow.routing import Decision
from src.workflow.turn import new_turn_input

SETTINGS = get_config()["persistence"]
MESSAGES = SETTINGS["messages"]
DAY = 86_400
SRC = {"title": "Capital Gains Basics", "url": "https://www.incometax.gov.in", "chunk_id": "a/b#0", "score": 0.7, "snippet": "A capital gain is a profit."}


def settings_for(tmp_path, **changes):
    return {**copy.deepcopy(SETTINGS), "path": str(tmp_path / "data" / "conversations.sqlite"), **changes}


def say(key):
    return " ".join(MESSAGES[key].split())


@pytest.fixture
def stores():
    """Opened stores are closed at the end of the test (Windows cannot delete an open file)."""
    opened = []

    def make(settings):
        store = open_store(settings)
        opened.append(store)
        return store

    yield make
    for store in opened:
        store.close()


# ============================== opening the store ==============================

def test_a_missing_file_and_folder_are_created_and_conversations_are_saved(tmp_path, stores):
    settings = settings_for(tmp_path)
    store = stores(settings)
    assert isinstance(store.saver, SqliteSaver) and store.status == PersistenceStatus(True, MESSAGES["saving"])
    assert Path(settings["path"]).exists() and store.thread_count() == 0


def test_a_relative_path_is_resolved_from_the_project_folder(tmp_path, stores, monkeypatch):
    monkeypatch.setattr(persistence, "PROJECT_ROOT", tmp_path)
    store = stores(settings_for(tmp_path, path="data/x.sqlite"))
    assert (tmp_path / "data" / "x.sqlite").exists() and store.status.saving


def test_the_default_settings_come_from_the_config(tmp_path, stores, monkeypatch):
    monkeypatch.setattr(persistence, "PROJECT_ROOT", tmp_path)
    store = stores(None)
    assert store.status.saving and (tmp_path / "data" / "conversations.sqlite").exists()


def test_saving_switched_off_uses_memory(tmp_path):
    store = open_store(settings_for(tmp_path, enabled=False))
    assert isinstance(store.saver, MemorySaver) and store.status == PersistenceStatus(False, MESSAGES["disabled"])
    assert not (tmp_path / "data").exists()


def test_a_corrupt_file_is_set_aside_and_a_new_one_started(tmp_path, stores):
    settings = settings_for(tmp_path)
    path = Path(settings["path"])
    path.parent.mkdir()
    path.write_bytes(b"this is not a database" * 100)
    Path(str(path) + "-wal").write_bytes(b"left over")
    store = stores(settings)
    assert store.status == PersistenceStatus(True, MESSAGES["recovered"]) and isinstance(store.saver, SqliteSaver)
    aside = sorted(p.name for p in path.parent.iterdir() if ".corrupt-" in p.name)
    assert aside and aside[0].startswith("conversations.sqlite.corrupt-")
    store.touch("t1")
    assert store.thread_count() == 1


def test_a_corrupt_file_that_cannot_be_renamed_falls_back_to_memory(tmp_path, monkeypatch):
    settings = settings_for(tmp_path)
    path = Path(settings["path"])
    path.parent.mkdir()
    path.write_bytes(b"garbage" * 200)

    def locked(self, target):
        raise PermissionError("the file is in use")

    monkeypatch.setattr(Path, "replace", locked)
    store = open_store(settings)
    assert isinstance(store.saver, MemorySaver) and store.status == PersistenceStatus(False, MESSAGES["corrupt_locked"])
    assert path.read_bytes().startswith(b"garbage")                      # the damaged file was left alone


def test_a_file_that_is_still_bad_after_the_rename_falls_back_to_memory(tmp_path, monkeypatch):
    class Cursor:
        def fetchone(self):
            return ["*** in database main ***"]

    class BadCheck:
        """A connection whose integrity check always fails."""

        def __init__(self, conn):
            self._conn = conn

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def execute(self, sql, *args):
            return Cursor() if "quick_check" in sql else self._conn.execute(sql, *args)

    real_connect = sqlite3.connect
    monkeypatch.setattr(persistence.sqlite3, "connect", lambda *a, **k: BadCheck(real_connect(*a, **k)))
    store = open_store(settings_for(tmp_path))
    assert isinstance(store.saver, MemorySaver) and store.status.message == MESSAGES["unavailable"]


def test_a_locked_file_falls_back_to_memory_without_renaming_it(tmp_path):
    settings = settings_for(tmp_path, busy_timeout_seconds=0.05)
    path = Path(settings["path"])
    path.parent.mkdir()
    holder = sqlite3.connect(str(path), isolation_level=None)
    holder.execute("PRAGMA journal_mode=DELETE")
    holder.execute("CREATE TABLE t (x)")
    holder.execute("BEGIN EXCLUSIVE")
    try:
        store = open_store(settings)
    finally:
        holder.close()
    assert isinstance(store.saver, MemorySaver) and store.status.message == MESSAGES["unavailable"]
    assert not [p for p in path.parent.iterdir() if ".corrupt-" in p.name]


def test_a_folder_that_cannot_be_created_falls_back_to_memory(tmp_path):
    blocker = tmp_path / "data"
    blocker.write_text("a file where the folder should be")
    store = open_store(settings_for(tmp_path))
    assert isinstance(store.saver, MemorySaver) and store.status.message == MESSAGES["unavailable"]


# ============================== bookkeeping: touch, delete, retention ==============================

def run_turn(graph, thread, query="What is a bond?", **kwargs):
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 25}
    for _ in graph.stream(new_turn_input(query, **kwargs), config, stream_mode="updates", durability="exit"):
        pass
    return graph.get_state(config).values


class Agent:
    def __init__(self, name, sources=()):
        self.name, self.sources = name, list(sources)

    def run(self, state):
        return {"agent": self.name, "turn_id": state["turn_id"], "status": "ok", "answer": f"{self.name} says hello",
                "sources": self.sources, "error": None, "data": None}


def graph_on(store):
    agents = {name: Agent(name, [SRC] if name == "finance_qa" else []) for name in get_config()["workflow"]["agent_intents"]}
    return build_graph(lambda state: Decision(["finance_qa"], 0.9, None, None), agents=agents, checkpointer=store.saver)


def row_counts(path, thread):
    con = sqlite3.connect(path)
    try:
        return [con.execute(f"SELECT COUNT(*) FROM {table} WHERE thread_id = ?", (thread,)).fetchone()[0] for table in ("checkpoints", "writes", "thread_activity")]
    finally:
        con.close()


def test_delete_removes_the_rows_of_that_thread_only(tmp_path, stores):
    settings = settings_for(tmp_path)
    store = stores(settings)
    graph = graph_on(store)
    for thread in ("mine", "yours"):
        run_turn(graph, thread)
        store.touch(thread)
    assert row_counts(settings["path"], "mine")[0] >= 1 and row_counts(settings["path"], "mine")[2] == 1
    store.delete("mine")
    assert row_counts(settings["path"], "mine") == [0, 0, 0] and row_counts(settings["path"], "yours")[0] >= 1
    assert graph.get_state({"configurable": {"thread_id": "mine"}}).values == {}


def test_retention_deletes_old_threads_then_the_least_recently_used_beyond_the_limit(tmp_path, stores):
    store = stores(settings_for(tmp_path))
    now = 1_000 * DAY
    for name, age in (("ancient", 40), ("old", 31), ("a", 5), ("b", 4), ("c", 3), ("d", 2)):
        store.touch(name, now - age * DAY)
    assert store.cleanup(max_age_days=30, max_threads=3, now=now) == 3          # ancient + old by age, then "a" by count
    assert [thread for thread, _ in store._activity()] == ["b", "c", "d"]
    assert store.cleanup(max_age_days=30, max_threads=3, now=now) == 0


def test_a_touch_updates_the_last_use_and_retention_works_on_a_memory_store_too():
    store = memory_store("memory")
    store.touch("t1", 10.0)
    store.touch("t1", 20.0)
    store.touch("t2", 30.0)
    assert store._activity() == [("t1", 20.0), ("t2", 30.0)]
    assert store.cleanup(max_age_days=1, max_threads=1, now=30.0) == 1 and store.thread_count() == 1
    store.delete("t2")
    store.delete("missing")
    assert store.thread_count() == 0 and store.status == PersistenceStatus(False, "memory")


def test_close_is_safe_twice_and_survives_a_connection_that_refuses_to_close():
    memory_store("m").close()

    class Broken:
        def close(self):
            raise sqlite3.ProgrammingError("already closed")

    store = ConversationStore(MemorySaver(), PersistenceStatus(False, "m"), Broken())      # type: ignore[arg-type]
    store.close()


# ============================== a restart continues the thread ==============================

def test_a_second_graph_on_the_same_file_continues_the_conversation(tmp_path, stores):
    settings = settings_for(tmp_path)
    first = stores(settings)
    run_turn(graph_on(first), "t1", "What is a bond?")
    run_turn(graph_on(first), "t1", "Market analysis", intent_hint="market")
    first.close()                                                                       # the app stops
    second = stores(settings)                                                           # the app starts again on the same file
    graph = graph_on(second)
    before = graph.get_state({"configurable": {"thread_id": "t1"}}).values
    assert [m.content for m in before["messages"] if m.type == "human"] == ["What is a bond?", "Market analysis"]
    after = run_turn(graph, "t1", "And a stock?")
    assert [m.content for m in after["messages"] if m.type == "human"] == ["What is a bond?", "Market analysis", "And a stock?"]
    assert len(after["messages"]) == 6 and after["final_answer"]


# ============================== the facade on top ==============================

@pytest.fixture
def app(tmp_path, monkeypatch):
    """The facade on a temp-file store with fake agents; app.restart() simulates stopping and starting the app."""
    settings = settings_for(tmp_path)
    holder = {}

    class Env:
        def restart(self):
            for cached in (facade.get_graph, facade.get_store):
                getattr(cached, "cache_clear", lambda: None)()
            if "store" in holder:
                holder["store"].close()
            holder["store"] = open_store(settings)

    env = Env()
    monkeypatch.setattr(facade, "open_store", lambda *a, **k: holder["store"])
    monkeypatch.setattr(facade, "Router", lambda: lambda state: Decision(["market" if state["intent_hint"] == "market" else "finance_qa"], 0.9, None, None))
    monkeypatch.setattr(
        facade, "build_graph",
        lambda classify, checkpointer: build_graph(
            classify, checkpointer=checkpointer,
            agents={name: Agent(name, [SRC] if name == "finance_qa" else []) for name in get_config()["workflow"]["agent_intents"]},
        ),
    )
    env.restart()
    yield env
    for cached in (facade.get_graph, facade.get_store):
        getattr(cached, "cache_clear", lambda: None)()          # a test may have replaced the cached function by a plain one
    holder["store"].close()


def test_history_after_a_restart_has_the_turns_their_sources_and_their_tab_tags(app):
    first = facade.run_query("What is a bond?", "web-1")
    assert first.stopped is None
    facade.run_query("Market analysis for AAA.", "web-1", intent_hint="market", market_params={"tickers": ["AAA"]})
    app.restart()
    items = facade.history("web-1")
    assert [(i.role, i.tab) for i in items] == [("user", None), ("assistant", None), ("user", "market"), ("assistant", "market")]
    assert items[0].text == "What is a bond?" and items[1].text == first.answer and items[1].sources == [SRC] and items[3].sources == []
    assert facade.run_query("And a stock?", "web-1").stopped is None and len(facade.history("web-1")) == 6


def test_history_of_an_unknown_thread_is_empty_and_a_failure_is_an_empty_list(app, monkeypatch):
    assert facade.history("never-used") == []
    monkeypatch.setattr(facade, "get_graph", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    assert facade.history("web-1") == []


def test_delete_thread_removes_the_conversation_and_reports_failure(app, monkeypatch):
    facade.run_query("What is a bond?", "web-1")
    facade.run_query("What is a bond?", "web-2")
    assert facade.delete_thread("web-1") is True
    assert facade.history("web-1") == [] and len(facade.history("web-2")) == 2
    monkeypatch.setattr(facade, "get_store", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    assert facade.delete_thread("web-2") is False


def test_a_turn_is_recorded_for_retention_and_a_recording_failure_never_breaks_it(app, monkeypatch):
    facade.run_query("What is a bond?", "web-1")
    assert [thread for thread, _ in facade.get_store()._activity()] == ["web-1"]
    monkeypatch.setattr(facade.get_store(), "touch", lambda thread: (_ for _ in ()).throw(sqlite3.OperationalError("locked")))
    assert facade.run_query("Again", "web-1").stopped is None


def test_get_store_cleans_up_old_threads_once_and_closes_at_exit(tmp_path, monkeypatch):
    store = memory_store("m")
    store.touch("stale", time.time() - 99 * DAY)
    registered = []
    facade.get_store.cache_clear()
    monkeypatch.setattr(facade, "open_store", lambda: store)
    monkeypatch.setattr(facade.atexit, "register", lambda fn: registered.append(fn))
    try:
        assert facade.get_store() is store and facade.get_store() is store
        assert store.thread_count() == 0 and registered == [store.close]
        assert facade.persistence_status() == PersistenceStatus(False, "m")
    finally:
        facade.get_store.cache_clear()


def test_a_failing_cleanup_is_logged_and_the_store_is_still_returned(monkeypatch):
    store = memory_store("m")
    monkeypatch.setattr(store, "cleanup", lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("locked")))
    facade.get_store.cache_clear()
    monkeypatch.setattr(facade, "open_store", lambda: store)
    monkeypatch.setattr(facade.atexit, "register", lambda fn: None)
    try:
        assert facade.get_store() is store
    finally:
        facade.get_store.cache_clear()


def test_the_graph_uses_the_stores_checkpointer(monkeypatch):
    store = memory_store("m")
    seen = {}
    facade.get_graph.cache_clear()
    facade.get_store.cache_clear()
    monkeypatch.setattr(facade, "get_store", lambda: store)
    monkeypatch.setattr(facade, "Router", lambda: "router")
    monkeypatch.setattr(facade, "build_graph", lambda classify, checkpointer: seen.update(classify=classify, saver=checkpointer) or "graph")
    try:
        assert facade.get_graph() == "graph" and seen == {"classify": "router", "saver": store.saver}
    finally:
        facade.get_graph.cache_clear()


def test_the_status_falls_back_to_the_unavailable_sentence_when_the_store_cannot_be_read(monkeypatch):
    monkeypatch.setattr(facade, "get_store", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    assert facade.persistence_status() == PersistenceStatus(False, say("unavailable"))


def test_each_turn_asks_for_one_checkpoint_per_turn(monkeypatch):
    seen = []

    class Spy:
        def stream(self, turn, config, stream_mode, durability=None):
            seen.append(durability)
            return iter(())

        def get_state(self, config):
            return type("Snap", (), {"values": {"final_answer": "ok", "turn_id": ""}})()

    monkeypatch.setattr(facade, "get_graph", lambda: Spy())
    monkeypatch.setattr(facade, "_remember", lambda thread: None)
    facade.run_query("hello", "t1")
    assert seen == ["exit"]


def test_thread_locks_disappear_when_unused_but_are_shared_while_in_use():
    import gc

    with facade._thread_lock("same") as _:
        assert facade._thread_lock("same") is facade._thread_lock("same")
        assert "same" in facade._thread_locks
    gc.collect()
    assert "same" not in facade._thread_locks
    for number in range(50):
        with facade._thread_lock(f"t{number}"):
            pass
    gc.collect()
    assert len(facade._thread_locks) == 0


def test_the_lock_makes_a_second_turn_on_the_same_thread_wait():
    lock = facade._thread_lock("busy")
    with lock:
        assert lock._lock.locked()
    assert not lock._lock.locked()


def test_neither_the_thread_id_nor_the_message_text_is_logged(app, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    facade.run_query("SECRET QUESTION", "SECRET-THREAD")
    facade.history("SECRET-THREAD")
    facade.delete_thread("SECRET-THREAD")
    monkeypatch.setattr(facade, "get_graph", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    facade.history("SECRET-THREAD")
    store = memory_store("m")
    store.touch("SECRET-THREAD", 1.0)
    store.cleanup(1, 1, now=10 * DAY)
    assert "SECRET" not in caplog.text and caplog.text


# ============================== what is stored on the messages ==============================

def test_the_human_message_carries_the_tab_only_for_a_hinted_turn():
    assert new_turn_input("hi")["messages"][0].additional_kwargs == {}
    assert new_turn_input("hi", intent_hint="goal_planning")["messages"][0].additional_kwargs == {"tab": "goal_planning"}


def test_the_assistant_message_carries_its_sources():
    update = nodes.reply("An answer.", [SRC])
    assert update["messages"][0].additional_kwargs == {"sources": [SRC]} and update["sources"] == [SRC]
    assert nodes.reply("No sources.")["messages"][0].additional_kwargs == {"sources": []}


# ============================== the backend seam ==============================

def test_the_backend_functions_call_the_facade(monkeypatch):
    monkeypatch.setattr(backend, "history", lambda thread: ["h", thread])
    monkeypatch.setattr(backend, "delete_thread", lambda thread: thread == "yes")
    monkeypatch.setattr(backend, "persistence_status", lambda: PersistenceStatus(True, "on"))
    assert backend.read_history("t") == ["h", "t"] and backend.forget("yes") is True and backend.forget("no") is False
    assert backend.persistence() == PersistenceStatus(True, "on")


# ============================== the config ==============================

def _config(tmp_path, change):
    cfg = copy.deepcopy(dict(get_config()))
    get_config.cache_clear()
    change(cfg)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def test_a_valid_persistence_block_loads(tmp_path):
    assert get_config(_config(tmp_path, lambda c: None))["persistence"]["max_threads"] == SETTINGS["max_threads"]
    get_config.cache_clear()


@pytest.mark.parametrize("change, text", [
    (lambda c: c.update(persistence="x"), "must be a mapping"),
    (lambda c: c["persistence"].update(enabled="yes"), "enabled"),
    (lambda c: c["persistence"].update(path=" "), "path"),
    (lambda c: c["persistence"].update(path=5), "path"),
    (lambda c: c["persistence"].update(busy_timeout_seconds=0), "busy_timeout_seconds"),
    (lambda c: c["persistence"].update(busy_timeout_seconds=True), "busy_timeout_seconds"),
    (lambda c: c["persistence"].update(max_age_days=0), "max_age_days"),
    (lambda c: c["persistence"].update(max_threads=1.5), "max_threads"),
    (lambda c: c["persistence"]["messages"].pop("saving"), "messages.saving"),
    (lambda c: c["persistence"]["messages"].update(corrupt_locked=" "), "messages.corrupt_locked"),
    (lambda c: c["persistence"].update(messages=None), "messages.saving"),
])
def test_a_bad_persistence_block_is_rejected(tmp_path, change, text):
    path = _config(tmp_path, change)
    with pytest.raises(ConfigError, match=text):
        get_config(path)
    get_config.cache_clear()
