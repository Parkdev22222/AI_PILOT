import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from envs.JSBSim.envs import MultipleCombatEnv
from .commander_db import CommanderCombatDB


@dataclass
class BaseState:
    name: str
    position_km: Tuple[float, float]
    ready_fighters: int


@dataclass
class AirTrack:
    track_id: str
    side: str
    position_km: np.ndarray
    heading_rad: float
    speed_kmph: float = 2000.0
    is_shotdown: bool = False
    missiles_remaining: int = 4

    def step(self, dt_hours: float):
        if self.is_shotdown:
            return
        dx = self.speed_kmph * math.cos(self.heading_rad) * dt_hours
        dy = self.speed_kmph * math.sin(self.heading_rad) * dt_hours
        self.position_km = self.position_km + np.array([dx, dy], dtype=np.float64)


@dataclass
class Engagement:
    env_id: str
    region: str
    env: MultipleCombatEnv
    allies: List[AirTrack] = field(default_factory=list)
    enemies: List[AirTrack] = field(default_factory=list)


class Exaone4CommanderAgent:
    """smolagents-based commander wrapper. Falls back to heuristic if unavailable."""

    def __init__(self, db: CommanderCombatDB, model_id: str = "exaone4"):
        self.db = db
        self.model_id = model_id
        self.agent = self._build_agent()

    def _build_agent(self):
        try:
            from smolagents import CodeAgent, LiteLLMModel, tool
        except Exception:
            return None

        @tool
        def query_battle_db(sql_query: str) -> str:
            """Run SELECT query for current Korea air battle DB."""
            return self.db.query(sql_query)

        model = LiteLLMModel(model_id=self.model_id)
        return CodeAgent(tools=[query_battle_db], model=model)

    def decide_scramble(self, prompt: str) -> str:
        if self.agent is None:
            return "HEURISTIC: scramble nearest base with available fighters"
        try:
            return str(self.agent.run(prompt))
        except Exception:
            return "HEURISTIC: scramble nearest base with available fighters"

    def decide_rtb(self, prompt: str) -> str:
        if self.agent is None:
            return "RTB_IF_SHOTDOWN_OR_EMPTY"
        try:
            return str(self.agent.run(prompt))
        except Exception:
            return "RTB_IF_SHOTDOWN_OR_EMPTY"


class KoreaAirCommanderSystem:
    """Top-level commander simulation with multi-region MultiCombat engagements."""

    def __init__(
        self,
        db_path: str,
        scenario_name: str = "2v2/NoWeapon/HierarchySelfplay",
        model_id: str = "exaone4",
    ):
        self.db = CommanderCombatDB(db_path)
        self.run_id = f"korea_commander_{int(time.time())}"
        self.scenario_name = scenario_name
        self.commander = Exaone4CommanderAgent(self.db, model_id=model_id)

        self.bases: Dict[str, BaseState] = {
            "Seosan": BaseState("Seosan", (126.5, 36.7), ready_fighters=8),
            "Daegu": BaseState("Daegu", (128.6, 35.9), ready_fighters=8),
            "Gangneung": BaseState("Gangneung", (128.9, 37.8), ready_fighters=8),
        }

        self.enemy_tracks: Dict[str, AirTrack] = {}
        self.friendly_tracks: Dict[str, AirTrack] = {}
        self.engagements: Dict[str, Engagement] = {}
        self.global_step = 0

    def add_enemy_wave(self, track_id: str, start_km: Tuple[float, float], heading_rad: float):
        self.enemy_tracks[track_id] = AirTrack(
            track_id=track_id,
            side="enemy",
            position_km=np.array(start_km, dtype=np.float64),
            heading_rad=heading_rad,
        )

    def step(self, dt_seconds: float = 10.0):
        self.global_step += 1
        dt_hours = dt_seconds / 3600.0

        for track in self.enemy_tracks.values():
            track.step(dt_hours)
        for track in self.friendly_tracks.values():
            track.step(dt_hours)

        self._launch_interceptors()
        self._activate_engagements_if_needed()
        self._step_active_engagements()

    def _launch_interceptors(self):
        for enemy in self.enemy_tracks.values():
            if enemy.is_shotdown:
                continue
            nearest_base = self._nearest_available_base(enemy.position_km)
            if nearest_base is None:
                continue
            if self._has_assigned_friendly(enemy.track_id):
                continue

            prompt = (
                "한반도 전역 상황을 바탕으로 적기 남하 경로를 고려해 출격 기지를 선택하라. "
                f"적기={enemy.track_id}, 위치={enemy.position_km.tolist()}, 후보기지={list(self.bases.keys())}"
            )
            commander_answer = self.commander.decide_scramble(prompt)
            self.db.log_event(
                run_id=self.run_id,
                global_step=self.global_step,
                event_type="SCRAMBLE_DECISION",
                event_payload={"enemy_id": enemy.track_id, "decision": commander_answer},
            )

            base = nearest_base
            base.ready_fighters -= 1
            heading = math.atan2(enemy.position_km[1] - base.position_km[1], enemy.position_km[0] - base.position_km[0])
            fid = f"F_{base.name}_{enemy.track_id}"
            self.friendly_tracks[fid] = AirTrack(
                track_id=fid,
                side="ally",
                position_km=np.array(base.position_km, dtype=np.float64),
                heading_rad=heading,
            )

    def _activate_engagements_if_needed(self):
        for fid, friendly in list(self.friendly_tracks.items()):
            if friendly.is_shotdown:
                continue
            enemy = self._paired_enemy(fid)
            if enemy is None or enemy.is_shotdown:
                continue
            dist_km = float(np.linalg.norm(friendly.position_km - enemy.position_km))
            if dist_km > 40.0:
                continue

            env_id = f"env_{fid}_{enemy.track_id}"
            if env_id in self.engagements:
                continue

            region = self._region_name((friendly.position_km + enemy.position_km) / 2.0)
            os.environ.setdefault("CLOSEAIRCOMBAT_TELEMETRY_DB", self.db.db_path)
            env = MultipleCombatEnv(self.scenario_name)
            env.reset()
            self.engagements[env_id] = Engagement(env_id=env_id, region=region, env=env, allies=[friendly], enemies=[enemy])
            self.db.log_event(
                run_id=self.run_id,
                global_step=self.global_step,
                env_id=env_id,
                region=region,
                event_type="ENGAGEMENT_START",
                event_payload={"distance_km": dist_km},
            )

    def _step_active_engagements(self):
        for env_id, engagement in list(self.engagements.items()):
            env = engagement.env
            action = np.zeros((env.num_agents, 4), dtype=np.int64)
            obs, share_obs, rewards, dones, info = env.step(action)
            battle_snapshot = info.get("battle_snapshot", {"allies": [], "enemies": []})
            self.db.log_engagement_snapshot(
                run_id=self.run_id,
                global_step=self.global_step,
                env_id=env_id,
                region=engagement.region,
                snapshot=battle_snapshot,
            )

            self._handle_events_with_llm(env_id, engagement.region, battle_snapshot)

            if np.all(np.array(dones).squeeze(-1)):
                self.db.log_event(
                    run_id=self.run_id,
                    global_step=self.global_step,
                    env_id=env_id,
                    region=engagement.region,
                    event_type="ENGAGEMENT_END",
                    event_payload={"reason": "env_done"},
                )
                env.close()
                del self.engagements[env_id]

    def _handle_events_with_llm(self, env_id: str, region: str, snapshot: Dict):
        for ally in snapshot.get("allies", []):
            shotdown = bool(ally.get("is_shotdown", False))
            empty = int(ally.get("missiles_remaining", 0)) <= 0
            if not (shotdown or empty):
                continue
            prompt = (
                f"교전지역={region}, 아군기={ally.get('uid')}, 격추={shotdown}, 무장고갈={empty}. "
                "복귀(RTB) 여부를 판단하라."
            )
            decision = self.commander.decide_rtb(prompt)
            self.db.log_event(
                run_id=self.run_id,
                global_step=self.global_step,
                env_id=env_id,
                region=region,
                event_type="RTB_DECISION",
                event_payload={"ally": ally.get("uid"), "decision": decision},
            )

    def _nearest_available_base(self, enemy_pos: np.ndarray) -> Optional[BaseState]:
        candidates = [b for b in self.bases.values() if b.ready_fighters > 0]
        if not candidates:
            return None
        return min(candidates, key=lambda b: np.linalg.norm(enemy_pos - np.array(b.position_km, dtype=np.float64)))

    def _has_assigned_friendly(self, enemy_track_id: str) -> bool:
        return any(tid.endswith(f"_{enemy_track_id}") for tid in self.friendly_tracks.keys())

    def _paired_enemy(self, friendly_id: str) -> Optional[AirTrack]:
        enemy_id = friendly_id.split("_")[-1]
        return self.enemy_tracks.get(enemy_id)

    @staticmethod
    def _region_name(midpoint_km: np.ndarray) -> str:
        x, y = float(midpoint_km[0]), float(midpoint_km[1])
        if y >= 37.5:
            return "중부권"
        if y <= 35.8:
            return "남부권"
        return "수도권-중남부 경계"
