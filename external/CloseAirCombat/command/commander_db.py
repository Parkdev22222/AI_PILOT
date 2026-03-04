import json
import os
import sqlite3
from typing import Any, Dict, Iterable, List, Tuple


class CommanderCombatDB:
    """SQLite backend for multi-region commander/engagement telemetry."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        db_dir = os.path.dirname(os.path.abspath(db_path))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS commander_engagement_telemetry (
                    run_id TEXT NOT NULL,
                    global_step INTEGER NOT NULL,
                    env_id TEXT NOT NULL,
                    region TEXT NOT NULL,
                    team TEXT NOT NULL,
                    unit_id TEXT NOT NULL,
                    x_km REAL NOT NULL,
                    y_km REAL NOT NULL,
                    z_km REAL NOT NULL,
                    is_shotdown INTEGER NOT NULL,
                    missiles_remaining INTEGER NOT NULL,
                    PRIMARY KEY (run_id, global_step, env_id, team, unit_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS commander_events (
                    run_id TEXT NOT NULL,
                    global_step INTEGER NOT NULL,
                    env_id TEXT,
                    region TEXT,
                    event_type TEXT NOT NULL,
                    event_payload TEXT NOT NULL
                )
                """
            )

    def log_engagement_snapshot(
        self,
        run_id: str,
        global_step: int,
        env_id: str,
        region: str,
        snapshot: Dict[str, Any],
    ):
        rows = list(self._snapshot_rows(run_id, global_step, env_id, region, snapshot))
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO commander_engagement_telemetry
                (run_id, global_step, env_id, region, team, unit_id, x_km, y_km, z_km, is_shotdown, missiles_remaining)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def log_event(
        self,
        run_id: str,
        global_step: int,
        event_type: str,
        event_payload: Dict[str, Any],
        env_id: str = "",
        region: str = "",
    ):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO commander_events (run_id, global_step, env_id, region, event_type, event_payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    int(global_step),
                    env_id,
                    region,
                    event_type,
                    json.dumps(event_payload, ensure_ascii=False),
                ),
            )

    def query(self, sql_query: str) -> str:
        stripped = sql_query.strip().lower()
        if not stripped.startswith("select"):
            raise ValueError("Only SELECT queries are allowed.")
        with self._connect() as conn:
            cursor = conn.execute(sql_query)
            columns = [desc[0] for desc in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return json.dumps(rows, ensure_ascii=False)

    def _snapshot_rows(
        self,
        run_id: str,
        global_step: int,
        env_id: str,
        region: str,
        snapshot: Dict[str, Any],
    ) -> Iterable[Tuple[Any, ...]]:
        for team_key in ("allies", "enemies"):
            for unit in snapshot.get(team_key, []):
                pos = unit.get("position", [0.0, 0.0, 0.0])
                yield (
                    run_id,
                    int(global_step),
                    env_id,
                    region,
                    team_key,
                    str(unit.get("uid", "")),
                    float(pos[0]) / 1000.0,
                    float(pos[1]) / 1000.0,
                    float(pos[2]) / 1000.0,
                    int(bool(unit.get("is_shotdown", False))),
                    int(unit.get("missiles_remaining", 0)),
                )
