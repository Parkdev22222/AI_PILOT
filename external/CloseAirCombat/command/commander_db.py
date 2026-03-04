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
                    lon REAL NOT NULL,
                    lat REAL NOT NULL,
                    alt_km REAL NOT NULL,
                    is_shotdown INTEGER NOT NULL,
                    missiles_remaining INTEGER NOT NULL,
                    PRIMARY KEY (run_id, global_step, env_id, team, unit_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS commander_tracks (
                    run_id TEXT NOT NULL,
                    global_step INTEGER NOT NULL,
                    track_id TEXT NOT NULL,
                    team TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    lon REAL NOT NULL,
                    lat REAL NOT NULL,
                    status TEXT NOT NULL,
                    env_id TEXT,
                    region TEXT,
                    PRIMARY KEY (run_id, global_step, track_id)
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
                (run_id, global_step, env_id, region, team, unit_id, lon, lat, alt_km, is_shotdown, missiles_remaining)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def log_tracks(self, run_id: str, global_step: int, rows: List[Dict[str, Any]]):
        if not rows:
            return
        payload = [
            (
                run_id,
                int(global_step),
                str(r["track_id"]),
                str(r["team"]),
                str(r["group_id"]),
                float(r["lon"]),
                float(r["lat"]),
                str(r.get("status", "TRANSIT")),
                str(r.get("env_id", "")),
                str(r.get("region", "")),
            )
            for r in rows
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO commander_tracks
                (run_id, global_step, track_id, team, group_id, lon, lat, status, env_id, region)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
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

    def get_live_state(self, run_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            latest_step_row = conn.execute(
                "SELECT COALESCE(MAX(global_step), 0) FROM commander_tracks WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            latest_step = int(latest_step_row[0]) if latest_step_row else 0

            track_rows = conn.execute(
                """
                SELECT track_id, team, group_id, lon, lat, status, env_id, region
                FROM commander_tracks
                WHERE run_id = ? AND global_step = ?
                ORDER BY team, track_id
                """,
                (run_id, latest_step),
            ).fetchall()

            engagement_rows = conn.execute(
                """
                SELECT env_id, region,
                       AVG(lon) AS center_lon,
                       AVG(lat) AS center_lat
                FROM commander_engagement_telemetry
                WHERE run_id = ? AND global_step = ?
                GROUP BY env_id, region
                """,
                (run_id, latest_step),
            ).fetchall()

        return {
            "run_id": run_id,
            "global_step": latest_step,
            "tracks": [
                {
                    "track_id": r[0],
                    "team": r[1],
                    "group_id": r[2],
                    "lon": r[3],
                    "lat": r[4],
                    "status": r[5],
                    "env_id": r[6],
                    "region": r[7],
                }
                for r in track_rows
            ],
            "engagements": [
                {
                    "env_id": r[0],
                    "region": r[1],
                    "center_lon": r[2],
                    "center_lat": r[3],
                }
                for r in engagement_rows
            ],
        }

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
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2]) / 1000.0,
                    int(bool(unit.get("is_shotdown", False))),
                    int(unit.get("missiles_remaining", 0)),
                )
