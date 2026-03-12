"""
combat_db.py
============
SQLite 기반 실시간 전투 시뮬레이션 데이터베이스 관리자.

Tables:
  - simulation_info   : 시뮬레이션 메타 정보
  - aircraft_state    : 매 스텝 항공기 상태
  - formation_info    : 편대 정보
  - events            : 이벤트 로그 (이벤트 발생 시 LLM 판단 대상)
  - llm_decisions     : LLM 결정 로그
"""

import json
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional


class CombatDB:
    """Thread-safe SQLite database manager for combat simulation."""

    def __init__(self, db_path: str = "combat_simulation.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            c = conn.cursor()

            c.executescript("""
                CREATE TABLE IF NOT EXISTS simulation_info (
                    sim_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time      TEXT    NOT NULL,
                    num_friendly    INTEGER NOT NULL,
                    num_enemy       INTEGER NOT NULL,
                    center_lon      REAL    NOT NULL,
                    center_lat      REAL    NOT NULL,
                    status          TEXT    DEFAULT 'active'
                );

                CREATE TABLE IF NOT EXISTS formation_info (
                    formation_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    sim_id          INTEGER NOT NULL,
                    team            TEXT    NOT NULL,   -- 'friendly' | 'enemy'
                    base_name       TEXT    NOT NULL,
                    base_lon        REAL    NOT NULL,
                    base_lat        REAL    NOT NULL,
                    aircraft_uids   TEXT    NOT NULL,   -- JSON list
                    paired_enemy_id INTEGER,            -- only for friendly
                    FOREIGN KEY (sim_id) REFERENCES simulation_info(sim_id)
                );

                CREATE TABLE IF NOT EXISTS aircraft_state (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    sim_id          INTEGER NOT NULL,
                    step            INTEGER NOT NULL,
                    timestamp       REAL    NOT NULL,
                    aircraft_uid    TEXT    NOT NULL,
                    team            TEXT    NOT NULL,
                    formation_id    INTEGER NOT NULL,
                    base_name       TEXT    NOT NULL,
                    is_alive        INTEGER NOT NULL,   -- 0 | 1
                    lon             REAL,
                    lat             REAL,
                    alt             REAL,
                    heading_deg     REAL,
                    speed_mps       REAL,
                    health          REAL,
                    missiles_left   INTEGER,
                    flight_phase    TEXT,               -- 'approach'|'combat'|'rtb'|'reload'|'support'
                    death_cause     TEXT DEFAULT 'alive', -- 'alive'|'shot_down'|'crashed'
                    FOREIGN KEY (sim_id) REFERENCES simulation_info(sim_id)
                );

                CREATE INDEX IF NOT EXISTS idx_state_step
                    ON aircraft_state(sim_id, step);

                CREATE TABLE IF NOT EXISTS events (
                    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    sim_id          INTEGER NOT NULL,
                    step            INTEGER NOT NULL,
                    timestamp       REAL    NOT NULL,
                    event_type      TEXT    NOT NULL,   -- 'major_loss' | 'ammo_depleted'
                    details_json    TEXT    NOT NULL,
                    llm_decision    TEXT,               -- filled after LLM response
                    resolved        INTEGER DEFAULT 0,
                    FOREIGN KEY (sim_id) REFERENCES simulation_info(sim_id)
                );

                CREATE TABLE IF NOT EXISTS llm_decisions (
                    decision_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                    sim_id          INTEGER NOT NULL,
                    step            INTEGER NOT NULL,
                    timestamp       REAL    NOT NULL,
                    decision_type   TEXT    NOT NULL,
                    input_prompt    TEXT    NOT NULL,
                    output_decision TEXT    NOT NULL,
                    reasoning       TEXT,
                    FOREIGN KEY (sim_id) REFERENCES simulation_info(sim_id)
                );
            """)
            conn.commit()
            # Migration: add death_cause column to existing DBs
            try:
                conn.execute(
                    "ALTER TABLE aircraft_state ADD COLUMN death_cause TEXT DEFAULT 'alive'"
                )
                conn.commit()
            except Exception:
                pass  # Column already exists
            conn.close()

    # ------------------------------------------------------------------
    # simulation_info
    # ------------------------------------------------------------------

    def create_simulation(
        self,
        num_friendly: int,
        num_enemy: int,
        center_lon: float,
        center_lat: float,
    ) -> int:
        """새 시뮬레이션 레코드 생성. sim_id 반환."""
        with self._lock:
            conn = self._get_conn()
            c = conn.cursor()
            c.execute(
                "INSERT INTO simulation_info "
                "(start_time, num_friendly, num_enemy, center_lon, center_lat, status) "
                "VALUES (?, ?, ?, ?, ?, 'active')",
                (datetime.utcnow().isoformat(), num_friendly, num_enemy,
                 center_lon, center_lat),
            )
            sim_id = c.lastrowid
            conn.commit()
            conn.close()
        return sim_id

    def update_simulation_status(self, sim_id: int, status: str):
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE simulation_info SET status=? WHERE sim_id=?",
                (status, sim_id),
            )
            conn.commit()
            conn.close()

    # ------------------------------------------------------------------
    # formation_info
    # ------------------------------------------------------------------

    def add_formation(
        self,
        sim_id: int,
        team: str,
        base_name: str,
        base_lon: float,
        base_lat: float,
        aircraft_uids: List[str],
        paired_enemy_id: Optional[int] = None,
    ) -> int:
        """편대 정보 저장. formation_id 반환."""
        with self._lock:
            conn = self._get_conn()
            c = conn.cursor()
            c.execute(
                "INSERT INTO formation_info "
                "(sim_id, team, base_name, base_lon, base_lat, aircraft_uids, paired_enemy_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sim_id, team, base_name, base_lon, base_lat,
                 json.dumps(aircraft_uids), paired_enemy_id),
            )
            formation_id = c.lastrowid
            conn.commit()
            conn.close()
        return formation_id

    def get_formations(self, sim_id: int, team: Optional[str] = None) -> List[Dict]:
        with self._lock:
            conn = self._get_conn()
            if team:
                rows = conn.execute(
                    "SELECT * FROM formation_info WHERE sim_id=? AND team=?",
                    (sim_id, team),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM formation_info WHERE sim_id=?", (sim_id,)
                ).fetchall()
            result = [dict(r) for r in rows]
            conn.close()
        for r in result:
            r["aircraft_uids"] = json.loads(r["aircraft_uids"])
        return result

    # ------------------------------------------------------------------
    # aircraft_state
    # ------------------------------------------------------------------

    def save_aircraft_state(
        self,
        sim_id: int,
        step: int,
        timestamp: float,
        aircraft_uid: str,
        team: str,
        formation_id: int,
        base_name: str,
        is_alive: bool,
        lon: float,
        lat: float,
        alt: float,
        heading_deg: float,
        speed_mps: float,
        health: float,
        missiles_left: int,
        flight_phase: str,
        death_cause: str = "alive",
    ):
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO aircraft_state "
                "(sim_id, step, timestamp, aircraft_uid, team, formation_id, base_name, "
                "is_alive, lon, lat, alt, heading_deg, speed_mps, health, missiles_left, flight_phase, death_cause) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (sim_id, step, timestamp, aircraft_uid, team, formation_id,
                 base_name, int(is_alive), lon, lat, alt, heading_deg,
                 speed_mps, health, missiles_left, flight_phase, death_cause),
            )
            conn.commit()
            conn.close()

    def get_latest_aircraft_states(self, sim_id: int) -> List[Dict]:
        """현재 시뮬레이션의 최신 스텝 각 항공기 상태 조회."""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute("""
                SELECT a.*
                FROM aircraft_state a
                INNER JOIN (
                    SELECT aircraft_uid, MAX(step) AS max_step
                    FROM aircraft_state
                    WHERE sim_id = ?
                    GROUP BY aircraft_uid
                ) b ON a.aircraft_uid = b.aircraft_uid AND a.step = b.max_step
                WHERE a.sim_id = ?
            """, (sim_id, sim_id)).fetchall()
            result = [dict(r) for r in rows]
            conn.close()
        return result

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------

    def log_event(
        self,
        sim_id: int,
        step: int,
        timestamp: float,
        event_type: str,
        details: Dict[str, Any],
    ) -> int:
        """이벤트 기록. event_id 반환."""
        with self._lock:
            conn = self._get_conn()
            c = conn.cursor()
            c.execute(
                "INSERT INTO events "
                "(sim_id, step, timestamp, event_type, details_json, resolved) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                (sim_id, step, timestamp, event_type, json.dumps(details)),
            )
            event_id = c.lastrowid
            conn.commit()
            conn.close()
        return event_id

    def update_event_decision(self, event_id: int, llm_decision: Dict[str, Any]):
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE events SET llm_decision=?, resolved=1 WHERE event_id=?",
                (json.dumps(llm_decision), event_id),
            )
            conn.commit()
            conn.close()

    def get_unresolved_events(self, sim_id: int) -> List[Dict]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM events WHERE sim_id=? AND resolved=0",
                (sim_id,),
            ).fetchall()
            result = [dict(r) for r in rows]
            conn.close()
        for r in result:
            r["details_json"] = json.loads(r["details_json"])
        return result

    def get_event_by_id(self, event_id: int) -> Optional[Dict]:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM events WHERE event_id=?", (event_id,)
            ).fetchone()
            conn.close()
        if row is None:
            return None
        r = dict(row)
        r["details_json"] = json.loads(r["details_json"])
        return r

    # ------------------------------------------------------------------
    # llm_decisions
    # ------------------------------------------------------------------

    def log_llm_decision(
        self,
        sim_id: int,
        step: int,
        timestamp: float,
        decision_type: str,
        input_prompt: str,
        output_decision: str,
        reasoning: str = "",
    ) -> int:
        with self._lock:
            conn = self._get_conn()
            c = conn.cursor()
            c.execute(
                "INSERT INTO llm_decisions "
                "(sim_id, step, timestamp, decision_type, input_prompt, output_decision, reasoning) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sim_id, step, timestamp, decision_type,
                 input_prompt, output_decision, reasoning),
            )
            decision_id = c.lastrowid
            conn.commit()
            conn.close()
        return decision_id

    def get_recent_decisions(self, sim_id: int, limit: int = 10) -> List[Dict]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM llm_decisions WHERE sim_id=? ORDER BY step DESC LIMIT ?",
                (sim_id, limit),
            ).fetchall()
            result = [dict(r) for r in rows]
            conn.close()
        return result
