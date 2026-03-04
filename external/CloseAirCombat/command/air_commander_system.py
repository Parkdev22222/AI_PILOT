import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from algorithms.ppo.ppo_actor import PPOActor
from envs.JSBSim.envs import MultipleCombatEnv
from .commander_db import CommanderCombatDB


class _PolicyArgs:
    """Minimal actor args compatible with renders/render_2v2.py."""

    def __init__(self, device: torch.device) -> None:
        self.gain = 0.01
        self.hidden_size = "128 128"
        self.act_hidden_size = "128 128"
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_prior = True


@dataclass
class BaseState:
    name: str
    lon_lat: Tuple[float, float]
    ready_fighters: int


@dataclass
class AirTrack:
    track_id: str
    side: str
    lon_lat: np.ndarray
    heading_rad: float
    speed_kmph: float = 2000.0
    is_shotdown: bool = False
    missiles_remaining: int = 4

    def step(self, dt_hours: float):
        if self.is_shotdown:
            return
        distance_km = self.speed_kmph * dt_hours
        east_km = distance_km * math.cos(self.heading_rad)
        north_km = distance_km * math.sin(self.heading_rad)

        lat = float(self.lon_lat[1])
        dlat = north_km / 111.0
        dlon = east_km / max(111.0 * math.cos(math.radians(lat)), 1e-6)
        self.lon_lat = self.lon_lat + np.array([dlon, dlat], dtype=np.float64)


@dataclass
class Engagement:
    env_id: str
    region: str
    env: MultipleCombatEnv
    allies: List[AirTrack] = field(default_factory=list)
    enemies: List[AirTrack] = field(default_factory=list)
    obs: Optional[np.ndarray] = None
    ego_rnn_states: Optional[np.ndarray] = None
    enm_rnn_states: Optional[np.ndarray] = None


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
        ego_policy_dir: str = "",
        enm_policy_dir: str = "",
        ego_policy_index: str = "latest",
        enm_policy_index: str = "latest",
        policy_device: str = "cpu",
    ):
        self.db = CommanderCombatDB(db_path)
        self.run_id = f"korea_commander_{int(time.time())}"
        self.scenario_name = scenario_name
        self.commander = Exaone4CommanderAgent(self.db, model_id=model_id)

        self.ego_policy_dir = ego_policy_dir
        self.enm_policy_dir = enm_policy_dir
        self.ego_policy_index = str(ego_policy_index)
        self.enm_policy_index = str(enm_policy_index)
        self.policy_device = torch.device(policy_device)
        self._policy_args = _PolicyArgs(self.policy_device)
        self._ego_policy: Optional[PPOActor] = None
        self._enm_policy: Optional[PPOActor] = None

        self.bases: Dict[str, BaseState] = {
            "Seosan": BaseState("Seosan", (126.5, 36.7), ready_fighters=8),
            "Daegu": BaseState("Daegu", (128.6, 35.9), ready_fighters=8),
            "Gangneung": BaseState("Gangneung", (128.9, 37.8), ready_fighters=8),
        }

        self.enemy_tracks: Dict[str, AirTrack] = {}
        self.friendly_tracks: Dict[str, AirTrack] = {}
        self.engagements: Dict[str, Engagement] = {}
        self.global_step = 0

    def add_enemy_wave(self, track_id: str, start_lon_lat: Tuple[float, float], heading_rad: float):
        self.enemy_tracks[track_id] = AirTrack(
            track_id=track_id,
            side="enemy",
            lon_lat=np.array(start_lon_lat, dtype=np.float64),
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
        self._log_live_tracks()

    def _load_policies_if_needed(self, env: MultipleCombatEnv):
        if self._ego_policy is not None and self._enm_policy is not None:
            return
        if not self.ego_policy_dir or not self.enm_policy_dir:
            return

        self._ego_policy = PPOActor(self._policy_args, env.observation_space, env.action_space, device=self.policy_device)
        self._enm_policy = PPOActor(self._policy_args, env.observation_space, env.action_space, device=self.policy_device)
        self._ego_policy.eval()
        self._enm_policy.eval()

        ego_path = os.path.join(self.ego_policy_dir, f"actor_{self.ego_policy_index}.pt")
        enm_path = os.path.join(self.enm_policy_dir, f"actor_{self.enm_policy_index}.pt")
        self._ego_policy.load_state_dict(torch.load(ego_path, map_location=self.policy_device))
        self._enm_policy.load_state_dict(torch.load(enm_path, map_location=self.policy_device))

    def _launch_interceptors(self):
        for enemy in self.enemy_tracks.values():
            if enemy.is_shotdown:
                continue
            nearest_base = self._nearest_available_base(enemy.lon_lat)
            if nearest_base is None:
                continue
            if self._has_assigned_friendly(enemy.track_id):
                continue

            prompt = (
                "한반도 전역 상황을 바탕으로 적기 남하 경로를 고려해 출격 기지를 선택하라. "
                f"적기={enemy.track_id}, 위치={enemy.lon_lat.tolist()}, 후보기지={list(self.bases.keys())}"
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
            vec = enemy.lon_lat - np.array(base.lon_lat, dtype=np.float64)
            heading = math.atan2(vec[1], vec[0])
            fid = f"F_{base.name}_{enemy.track_id}"
            self.friendly_tracks[fid] = AirTrack(
                track_id=fid,
                side="ally",
                lon_lat=np.array(base.lon_lat, dtype=np.float64),
                heading_rad=heading,
            )

    def _activate_engagements_if_needed(self):
        for fid, friendly in list(self.friendly_tracks.items()):
            if friendly.is_shotdown:
                continue
            enemy = self._paired_enemy(fid)
            if enemy is None or enemy.is_shotdown:
                continue
            dist_km = self._haversine_km(tuple(friendly.lon_lat), tuple(enemy.lon_lat))
            if dist_km > 40.0:
                continue

            env_id = f"env_{fid}_{enemy.track_id}"
            if env_id in self.engagements:
                continue

            midpoint = (friendly.lon_lat + enemy.lon_lat) / 2.0
            region = self._region_name(midpoint)
            os.environ.setdefault("CLOSEAIRCOMBAT_TELEMETRY_DB", self.db.db_path)
            env = MultipleCombatEnv(self.scenario_name)
            obs, _ = env.reset()
            self._load_policies_if_needed(env)

            n_side = env.num_agents // 2
            engagement = Engagement(
                env_id=env_id,
                region=region,
                env=env,
                allies=[friendly],
                enemies=[enemy],
                obs=obs,
                ego_rnn_states=np.zeros(
                    (n_side, self._policy_args.recurrent_hidden_layers, self._policy_args.recurrent_hidden_size),
                    dtype=np.float32,
                ),
                enm_rnn_states=np.zeros(
                    (n_side, self._policy_args.recurrent_hidden_layers, self._policy_args.recurrent_hidden_size),
                    dtype=np.float32,
                ),
            )
            self.engagements[env_id] = engagement
            self.db.log_event(
                run_id=self.run_id,
                global_step=self.global_step,
                env_id=env_id,
                region=region,
                event_type="ENGAGEMENT_START",
                event_payload={"distance_km": dist_km},
            )

    def _get_policy_actions(self, engagement: Engagement) -> np.ndarray:
        env = engagement.env
        if self._ego_policy is None or self._enm_policy is None or engagement.obs is None:
            return np.zeros((env.num_agents, 4), dtype=np.int64)

        n_side = env.num_agents // 2
        masks = np.ones((n_side, 1), dtype=np.float32)
        ego_obs = engagement.obs[:n_side, ...]
        enm_obs = engagement.obs[n_side:, ...]

        with torch.no_grad():
            ego_actions, _, ego_rnn = self._ego_policy(ego_obs, engagement.ego_rnn_states, masks, deterministic=True)
            enm_actions, _, enm_rnn = self._enm_policy(enm_obs, engagement.enm_rnn_states, masks, deterministic=True)

        engagement.ego_rnn_states = ego_rnn.detach().cpu().numpy()
        engagement.enm_rnn_states = enm_rnn.detach().cpu().numpy()
        actions = np.concatenate([ego_actions.detach().cpu().numpy(), enm_actions.detach().cpu().numpy()], axis=0)
        return actions.astype(np.int64)

    def _step_active_engagements(self):
        for env_id, engagement in list(self.engagements.items()):
            action = self._get_policy_actions(engagement)
            obs, _, _, dones, info = engagement.env.step(action)
            engagement.obs = obs

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
                engagement.env.close()
                del self.engagements[env_id]

    def _log_live_tracks(self):
        engaged_pairs = {}
        for env_id, engagement in self.engagements.items():
            for a in engagement.allies:
                engaged_pairs[a.track_id] = (env_id, engagement.region)
            for e in engagement.enemies:
                engaged_pairs[e.track_id] = (env_id, engagement.region)

        rows = []
        for tid, t in self.friendly_tracks.items():
            env_ref = engaged_pairs.get(tid)
            rows.append(
                {
                    "track_id": tid,
                    "team": "ally",
                    "group_id": self._group_id_from_track(tid),
                    "lon": float(t.lon_lat[0]),
                    "lat": float(t.lon_lat[1]),
                    "status": "ENGAGED" if env_ref else "TRANSIT",
                    "env_id": env_ref[0] if env_ref else "",
                    "region": env_ref[1] if env_ref else "",
                }
            )
        for tid, t in self.enemy_tracks.items():
            env_ref = engaged_pairs.get(tid)
            rows.append(
                {
                    "track_id": tid,
                    "team": "enemy",
                    "group_id": tid,
                    "lon": float(t.lon_lat[0]),
                    "lat": float(t.lon_lat[1]),
                    "status": "ENGAGED" if env_ref else "TRANSIT",
                    "env_id": env_ref[0] if env_ref else "",
                    "region": env_ref[1] if env_ref else "",
                }
            )

        self.db.log_tracks(self.run_id, self.global_step, rows)

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

    def _nearest_available_base(self, enemy_lon_lat: np.ndarray) -> Optional[BaseState]:
        candidates = [b for b in self.bases.values() if b.ready_fighters > 0]
        if not candidates:
            return None
        return min(candidates, key=lambda b: self._haversine_km(tuple(enemy_lon_lat), b.lon_lat))

    def _has_assigned_friendly(self, enemy_track_id: str) -> bool:
        return any(tid.endswith(f"_{enemy_track_id}") for tid in self.friendly_tracks.keys())

    def _paired_enemy(self, friendly_id: str) -> Optional[AirTrack]:
        enemy_id = self._group_id_from_track(friendly_id)
        return self.enemy_tracks.get(enemy_id)

    @staticmethod
    def _group_id_from_track(friendly_track_id: str) -> str:
        return friendly_track_id.split("_")[-1]

    @staticmethod
    def _haversine_km(a_lon_lat: Tuple[float, float], b_lon_lat: Tuple[float, float]) -> float:
        lon1, lat1 = map(math.radians, a_lon_lat)
        lon2, lat2 = map(math.radians, b_lon_lat)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 6371.0 * 2 * math.asin(math.sqrt(h))

    @staticmethod
    def _region_name(midpoint_lon_lat: np.ndarray) -> str:
        lat = float(midpoint_lon_lat[1])
        if lat >= 37.5:
            return "중부권"
        if lat <= 35.8:
            return "남부권"
        return "수도권-중남부 경계"
