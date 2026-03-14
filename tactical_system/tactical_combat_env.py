"""
tactical_combat_env.py
======================
MultipleCombatEnv_LLM 을 확장한 한반도 전술 전투 환경.

주요 기능
---------
- 외부에서 전달받은 초기 위치로 JSBSim 시뮬레이터 초기화
- 매 스텝 항공기 상태를 SQL DB 에 실시간 저장
- 접근 단계 (>20 km): 자동 조종 (heading controller)
- 교전 단계 (<20 km): RL 정책 (PPOActor / HierarchicalMultipleCombat)
- 이벤트 감지 → bool 플래그 설정
  * event_major_loss      : 아군 50% 이상 손실
  * event_ammo_depleted   : 아군 기체 무장 고갈
- RTB / 지원 편대 스폰 처리
"""

import logging
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── CloseAirCombat 경로 삽입 ─────────────────────────────────────────────────
_CAC_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "external", "CloseAirCombat"
)
if _CAC_ROOT not in sys.path:
    sys.path.insert(0, _CAC_ROOT)

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv_LLM
from envs.JSBSim.core.simulatior import AircraftSimulator, MissileSimulator
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import LLA2NEU, get_AO_TA_R
from algorithms.mappo.ppo_actor import PPOActor

from .combat_db import CombatDB

logger = logging.getLogger(__name__)

APPROACH_DISTANCE_M = 20_000     # 접근 → 교전 전환 거리 (20 km)
RELOAD_DISTANCE_M   = 2_000      # 기지 도착 판정 거리 (2 km)
SUPPORT_SPAWN_ALT_M = 6_000      # 지원 편대 스폰 고도 (m)
CRUISE_SPEED_FPS    = 800.0      # 순항 속도 (ft/s ≈ 244 m/s)
CRUISE_ALT_M        = 6_000.0    # 순항 고도 (m)
RELOAD_MISSILES     = 5          # 재장착 미사일 수

NUM_NEAREST_ENEMIES = 2          # 관측에 포함할 최근접 적 기체 수
# obs_length = 9(ego) + 6*(1파트너 + NUM_NEAREST_ENEMIES + 1미사일슬롯) = 33
_OBS_LENGTH = 9 + (1 + NUM_NEAREST_ENEMIES + 1) * 6


# ─────────────────────────────────────────────────────────────────────────────
# 동적 config 생성 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def make_dynamic_config(
    aircraft_configs: Dict,
    battle_field_center: Tuple[float, float, float] = (127.5, 38.5, 0.0),
    sim_freq: int = 60,
    agent_interaction_steps: int = 12,
    max_steps: int = 3000,
):
    """YAML parse 없이 동적으로 EnvConfig 오브젝트 생성."""
    attrs = dict(
        task="hierarchical_multiplecombat_shoot",
        sim_freq=sim_freq,
        agent_interaction_steps=agent_interaction_steps,
        max_steps=max_steps,
        altitude_limit=2500,
        acceleration_limit_x=10.0,
        acceleration_limit_y=10.0,
        acceleration_limit_z=10.0,
        battle_field_center=list(battle_field_center),
        aircraft_configs=aircraft_configs,
        max_attack_angle=45,
        max_attack_distance=14000,
        min_attack_interval=125,
        PostureReward_scale=15.0,
        PostureReward_potential=True,
        PostureReward_orientation_version="v2",
        PostureReward_range_version="v3",
        AltitudeReward_safe_altitude=4.0,
        AltitudeReward_danger_altitude=3.5,
        AltitudeReward_Kv=0.2,
    )
    return type("EnvConfig", (), attrs)


# ─────────────────────────────────────────────────────────────────────────────
# 오토파일럿 (접근 단계 heading controller)
# ─────────────────────────────────────────────────────────────────────────────

def _bearing_rad(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """두 지점 간 방위각 (rad, 북=0, 시계방향)."""
    d_lon = math.radians(lon2 - lon1)
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    x = math.sin(d_lon) * math.cos(lat2_r)
    y = (math.cos(lat1_r) * math.sin(lat2_r)
         - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(d_lon))
    return math.atan2(x, y) % (2 * math.pi)


def _haversine_m(lon1, lat1, lon2, lat2) -> float:
    """두 지점 간 대원 거리 (m)."""
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def compute_autopilot_action(
    sim: AircraftSimulator,
    target_lon: float,
    target_lat: float,
    target_alt_m: float = CRUISE_ALT_M,
) -> np.ndarray:
    """
    목표 지점 방향으로 기동하는 저수준 제어 액션 계산.

    Returns
    -------
    np.ndarray shape (4,) : [aileron, elevator, rudder, throttle]  정규화 완료
    """
    lon, lat, alt = sim.get_geodetic()
    roll, pitch, heading = sim.get_rpy()   # rad
    v_north, v_east, v_down = sim.get_velocity()

    desired_heading = _bearing_rad(lon, lat, target_lon, target_lat)

    # 헤딩 오차 (−π … π)
    hdg_err = desired_heading - heading
    while hdg_err > math.pi:
        hdg_err -= 2 * math.pi
    while hdg_err < -math.pi:
        hdg_err += 2 * math.pi

    # 고도 오차
    alt_err = target_alt_m - alt

    # ─── PD 제어기 ───
    # aileron: 뱅크-투-턴
    aileron = float(np.clip(1.2 * hdg_err, -1.0, 1.0))
    # elevator: 고도 보정 + 피치 댐핑
    elevator = float(np.clip(-0.0003 * alt_err - 1.5 * pitch, -1.0, 1.0))
    rudder = 0.0
    throttle = 0.85

    # action_var 의 정규화된 값 직접 반환 (HierarchicalTask 의 baseline lowlevel 우회)
    return np.array([aileron, elevator, rudder, throttle], dtype=np.float32)


def _discrete_to_normalized(action_continuous: np.ndarray) -> np.ndarray:
    """
    연속 제어값 [-1..1, -1..1, -1..1, 0..1] →
    MultiDiscrete [41, 41, 41, 30] 인덱스로 변환.
    (MultipleCombatTask.normalize_action 역변환)
    """
    nvec = np.array([41, 41, 41, 30])
    disc = np.zeros(4, dtype=np.int32)
    disc[0] = int(round((action_continuous[0] + 1.0) * (nvec[0] - 1) / 2.0))
    disc[1] = int(round((action_continuous[1] + 1.0) * (nvec[1] - 1) / 2.0))
    disc[2] = int(round((action_continuous[2] + 1.0) * (nvec[2] - 1) / 2.0))
    disc[3] = int(round((action_continuous[3] - 0.4) * (nvec[3] - 1) / 0.5))
    return np.clip(disc, 0, nvec - 1)


def _autopilot_hierarchical_action(
    sim: AircraftSimulator,
    target_lon: float,
    target_lat: float,
    target_alt_m: float = CRUISE_ALT_M,
) -> np.ndarray:
    """
    목표 지점을 향하는 고수준 계층적 행동 계산.

    Returns
    -------
    np.ndarray shape (4,) : [alt_idx, hdg_idx, vel_idx, shoot_idx]
    for MultiDiscrete([3, 5, 3, 2])
      alt_idx  : 0=상승(+0.1) / 1=유지(0) / 2=하강(-0.1)
      hdg_idx  : 0=좌-30° / 1=좌-15° / 2=직진 / 3=우+15° / 4=우+30°
      vel_idx  : 0=가속(+0.05) / 1=유지 / 2=감속(-0.05)
      shoot_idx: 0=미사격
    """
    lon, lat, alt = sim.get_geodetic()
    _, _, heading = sim.get_rpy()   # rad

    desired_heading = _bearing_rad(lon, lat, target_lon, target_lat)

    # 헤딩 오차 (−π … π)
    hdg_err = desired_heading - heading
    while hdg_err > math.pi:
        hdg_err -= 2 * math.pi
    while hdg_err < -math.pi:
        hdg_err += 2 * math.pi
    hdg_err_deg = math.degrees(hdg_err)

    # norm_delta_heading = [−π/6, −π/12, 0, π/12, π/6]
    if hdg_err_deg < -15:
        hdg_idx = 0
    elif hdg_err_deg < -5:
        hdg_idx = 1
    elif hdg_err_deg <= 5:
        hdg_idx = 2
    elif hdg_err_deg <= 15:
        hdg_idx = 3
    else:
        hdg_idx = 4

    # norm_delta_altitude = [0.1, 0, −0.1]
    alt_err = target_alt_m - alt
    if alt_err > 200:
        alt_idx = 0
    elif alt_err < -200:
        alt_idx = 2
    else:
        alt_idx = 1

    return np.array([alt_idx, hdg_idx, 1, 0], dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# 전술 전투 환경
# ─────────────────────────────────────────────────────────────────────────────

class TacticalCombatEnv(MultipleCombatEnv_LLM):
    """
    한반도 전술 전투 환경.

    Parameters
    ----------
    formation_pairs : list of dict
        [{"friendly_uids": ["A0100","A0200"],
          "enemy_uids":    ["B0100","B0200"],
          "friendly_base": {"name":…,"lon":…,"lat":…},
          "enemy_base":    {"name":…,"lon":…,"lat":…},
          "friendly_init_states": {uid: init_state_dict, …},
          "enemy_init_states":    {uid: init_state_dict, …}}, …]
    db : CombatDB
    sim_id : int
    ego_policy_path : str   PPOActor 체크포인트 경로
    enm_policy_path : str
    battle_field_center : (lon, lat, alt)
    """

    def __init__(
        self,
        formation_pairs: List[Dict],
        db: CombatDB,
        sim_id: int,
        ego_policy_path: str,
        enm_policy_path: str,
        battle_field_center: Tuple[float, float, float] = (127.5, 38.5, 0.0),
        device: str = "cpu",
        attack_target: Optional[Tuple[float, float]] = None,
    ):
        self.formation_pairs = formation_pairs
        self.db = db
        self.sim_id = sim_id
        self.device = torch.device(device)

        # ── 항공기 configs 동적 생성 ──────────────────────────────────────
        aircraft_configs = {}
        for pair in formation_pairs:
            for uid, state in pair["friendly_init_states"].items():
                aircraft_configs[uid] = {
                    "color": "Blue",
                    "model": "f16",
                    "init_state": state,
                    "missile": 2,
                }
            for uid, state in pair["enemy_init_states"].items():
                aircraft_configs[uid] = {
                    "color": "Red",
                    "model": "f16",
                    "init_state": state,
                    "missile": 2,
                }

        config = make_dynamic_config(
            aircraft_configs=aircraft_configs,
            battle_field_center=battle_field_center,
        )

        # ── BaseEnv.__init__ 우회: config 직접 주입 ───────────────────────
        self.config = config
        self.max_steps = config.max_steps
        self.sim_freq = config.sim_freq
        self.agent_interaction_steps = config.agent_interaction_steps
        clon, clat, calt = battle_field_center
        self.center_lon = clon
        self.center_lat = clat
        self.center_alt = calt
        self._create_records = False

        self.load()  # load_task() + load_simulator()

        # MultipleCombatEnv_LLM 추가 초기화
        self._build_partner_pairs()
        self._update_obs_space()
        self._patch_task_get_obs()  # task.get_obs → 2-nearest-enemy 버전으로 교체

        # ── RL 정책 로드 ──────────────────────────────────────────────────
        self.ego_policy = self._load_policy(ego_policy_path)
        self.enm_policy = self._load_policy(enm_policy_path)

        # ── 단계 관리 ─────────────────────────────────────────────────────
        # pair_idx → 'approach' | 'combat' | 'rtb_loss' | 'rtb_ammo' | 'support' | 'done'
        self.pair_phases: Dict[int, str] = {
            i: "approach" for i in range(len(formation_pairs))
        }

        # 재장착 대기 기체: {uid: {"home_lon":…,"home_lat":…,"combat_lon":…,"combat_lat":…}}
        self.reload_pending: Dict[str, Dict] = {}
        # 복귀 완료 후 재출격 대기: {uid: {"combat_lon":…,"combat_lat":…,"missiles":5}}
        self.reloaded_returning: Dict[str, Dict] = {}

        # ── 이벤트 플래그 ─────────────────────────────────────────────────
        self.event_major_loss: bool = False
        self.event_major_loss_id: Optional[int] = None
        self.event_ammo_depleted: bool = False
        self.event_ammo_depleted_ids: Dict[str, int] = {}  # uid → event_id
        # 편대 단위 무장 고갈: pair_idx → event_id
        self.event_formation_ammo_depleted: Dict[int, int] = {}
        # 아군 편대 전멸: pair_idx → event_id
        self.event_formation_destroyed: Dict[int, int] = {}
        # 적 편대 전멸(아군 승리): pair_idx → event_id
        self.event_enemy_formation_destroyed: Dict[int, int] = {}

        # 시나리오 공격 목표 지점 (lon, lat) — 적기 approach 방향 결정에 사용
        self.attack_target: Optional[Tuple[float, float]] = attack_target
        if attack_target is not None:
            logger.info(
                f"시나리오 공격 목표 설정: 경도 {attack_target[0]:.3f}°, "
                f"위도 {attack_target[1]:.3f}°"
            )

        # 스폰 일련번호 (중복 UID 방지)
        self._spawn_counter: int = 0

        # RL RNN 상태
        n_agents = len(self.ego_ids)
        self._ego_rnn = np.zeros((1, 1, 128), dtype=np.float32)
        self._enm_rnn = np.zeros((1, 1, 128), dtype=np.float32)
        self._masks = np.ones((max(n_agents, 1), 1), dtype=np.float32)

        # 편대 → DB formation_id 매핑
        self._friendly_formation_ids: Dict[int, int] = {}
        self._enemy_formation_ids: Dict[int, int] = {}

        # 지원 편대 스폰 추적
        self._support_spawned: Dict[int, bool] = {i: False for i in range(len(formation_pairs))}

        # 임무완료 편대 재배정 추적 (set of pair_idx)
        self._victory_reassigned: set = set()

    # ------------------------------------------------------------------
    # 정책 로드
    # ------------------------------------------------------------------

    def _load_policy(self, path: str) -> Optional[PPOActor]:
        if not path or not os.path.exists(path):
            logger.warning(f"정책 파일 없음: {path}. 오토파일럿으로 대체.")
            return None

        class _Args:
            gain = 0.01
            hidden_size = "128 128"
            act_hidden_size = "128 128"
            activation_id = 1
            use_feature_normalization = False
            use_recurrent_policy = True
            recurrent_hidden_size = 128
            recurrent_hidden_layers = 1
            tpdv = dict(dtype=torch.float32, device=torch.device("cpu"))

        policy = PPOActor(_Args(), self.observation_space, self.action_space,
                          device=self.device)
        policy.load_state_dict(torch.load(path, map_location=self.device))
        policy.eval()
        return policy

    # ------------------------------------------------------------------
    # 재설정
    # ------------------------------------------------------------------

    def reset(self):
        obs, share_obs = super().reset()
        # 이벤트 플래그 초기화
        self.event_major_loss = False
        self.event_major_loss_id = None
        self.event_ammo_depleted = False
        self.event_ammo_depleted_ids.clear()
        self.event_formation_ammo_depleted.clear()
        self.event_formation_destroyed.clear()
        self.event_enemy_formation_destroyed.clear()
        self._spawn_counter = 0
        self._victory_reassigned.clear()
        self.reload_pending.clear()
        self.reloaded_returning.clear()
        self.pair_phases = {i: "approach" for i in range(len(self.formation_pairs))}
        self._support_spawned = {i: False for i in range(len(self.formation_pairs))}
        n_agents = len(self.ego_ids)
        self._ego_rnn = np.zeros((1, 1, 128), dtype=np.float32)
        self._enm_rnn = np.zeros((1, 1, 128), dtype=np.float32)
        self._masks = np.ones((max(n_agents, 1), 1), dtype=np.float32)
        return obs, share_obs

    # ------------------------------------------------------------------
    # 메인 스텝 (접근 + 교전 + 이벤트 처리 통합)
    # ------------------------------------------------------------------

    def step_tactical(self) -> Dict:
        """
        전술 전투 환경의 단일 스텝.
        접근 단계: 오토파일럿 / 교전 단계: RL 정책.

        Returns
        -------
        info dict (dones, rewards, events, …)
        """
        # ── 임무완료 편대 재배정 (safe_return done=True 방지) ───────────
        # self.step() 내부 get_termination() 이 enemies 를 검사하기 전에
        # 새 적기를 enemies 에 추가해야 done=True 를 막을 수 있음
        self._reassign_victorious_formations()

        # ── 행동 계산 ──────────────────────────────────────────────────
        actions = self._compute_actions()

        # ── 환경 스텝 ──────────────────────────────────────────────────
        obs, share_obs, rewards, dones, info = self.step(actions)

        # ── 재장착 기체 처리 ────────────────────────────────────────────
        self._process_reload_aircraft()

        # ── DB 저장 ────────────────────────────────────────────────────
        self._save_states_to_db()

        # ── 단계 전환 확인 ──────────────────────────────────────────────
        self._update_pair_phases()

        # ── 이벤트 감지 ────────────────────────────────────────────────
        self._detect_events()

        info["pair_phases"] = dict(self.pair_phases)
        info["event_major_loss"] = self.event_major_loss
        info["event_major_loss_id"] = self.event_major_loss_id
        info["event_ammo_depleted"] = self.event_ammo_depleted
        info["event_ammo_depleted_ids"] = dict(self.event_ammo_depleted_ids)
        info["event_formation_ammo_depleted"] = dict(self.event_formation_ammo_depleted)
        info["event_formation_destroyed"] = dict(self.event_formation_destroyed)
        info["event_enemy_formation_destroyed"] = dict(self.event_enemy_formation_destroyed)

        return obs, share_obs, rewards, dones, info

    # ------------------------------------------------------------------
    # 행동 계산
    # ------------------------------------------------------------------

    def _compute_actions(self) -> np.ndarray:
        """
        각 항공기의 단계에 따라 오토파일럿 또는 RL 정책 행동 반환.
        shape: (num_agents, action_dim)
        """
        all_actions = {}

        # 아군
        for i, pair in enumerate(self.formation_pairs):
            phase = self.pair_phases[i]
            friendly_uids = [u for u in pair["friendly_uids"] if u in self.agents]
            enemy_uids    = [u for u in pair["enemy_uids"]    if u in self.agents]

            for uid in friendly_uids:
                sim = self.agents[uid]
                if not sim.is_alive:
                    all_actions[uid] = np.array([1, 2, 1, 0], dtype=np.int32)
                    continue

                if uid in self.reload_pending:
                    # LLM RTB 명령 — task.go_dest()로 기지 귀환
                    home = self.reload_pending[uid]
                    all_actions[uid] = self.task.go_dest(
                        self, uid,
                        dest_lon=home["home_lon"],
                        dest_lat=home["home_lat"],
                    )
                elif uid in self.reloaded_returning:
                    # 재장착 완료, 교전 지역 복귀 중 (오토파일럿 유지)
                    ret = self.reloaded_returning[uid]
                    all_actions[uid] = _autopilot_hierarchical_action(
                        sim, ret["combat_lon"], ret["combat_lat"])
                elif phase == "approach":
                    # 적 편대 중심으로 접근 (오토파일럿 유지)
                    tgt_lon, tgt_lat = self._formation_centroid(enemy_uids)
                    all_actions[uid] = _autopilot_hierarchical_action(
                        sim, tgt_lon, tgt_lat)
                elif phase in ("rtb_loss", "rtb_victory"):
                    # LLM RTB 명령 — task.go_dest()로 기지 귀환
                    base = pair["friendly_base"]
                    all_actions[uid] = self.task.go_dest(
                        self, uid,
                        dest_lon=base["lon"],
                        dest_lat=base["lat"],
                        target_alt_m=5000.0,
                    )
                elif phase == "support":
                    # 지원 요청 → 기존 잔존 기체는 계속 전투 (RL)
                    all_actions[uid] = self._rl_action_ego(uid)
                else:
                    # combat 단계: RL
                    all_actions[uid] = self._rl_action_ego(uid)

        # 적군 (항상 오토파일럿 접근 or RL 교전)
        for i, pair in enumerate(self.formation_pairs):
            phase = self.pair_phases[i]
            friendly_uids = [u for u in pair["friendly_uids"] if u in self.agents]
            enemy_uids    = [u for u in pair["enemy_uids"]    if u in self.agents]

            for uid in enemy_uids:
                sim = self.agents[uid]
                if not sim.is_alive:
                    all_actions[uid] = np.array([1, 2, 1, 0], dtype=np.int32)
                    continue
                if phase == "approach":
                    if self.attack_target is not None:
                        # 시나리오 설정 공격 목표로 go_dest 기동
                        tgt_lon, tgt_lat = self.attack_target
                        all_actions[uid] = self.task.go_dest(
                            self, uid,
                            dest_lon=tgt_lon,
                            dest_lat=tgt_lat,
                        )
                    else:
                        tgt_lon, tgt_lat = self._formation_centroid(friendly_uids)
                        all_actions[uid] = _autopilot_hierarchical_action(
                            sim, tgt_lon, tgt_lat)
                else:
                    all_actions[uid] = self._rl_action_enm(uid)

        # _pack 순서에 맞춰 배열 구성
        ordered = [all_actions[uid] for uid in self.ego_ids + self.enm_ids]
        return np.array(ordered, dtype=np.int32)

    # ------------------------------------------------------------------
    # 관측 공간 오버라이드 (파트너 1기 + 최근접 적 2기 + 미사일 슬롯)
    # ------------------------------------------------------------------

    def _patch_task_get_obs(self):
        """
        task.get_obs(env, agent_id)를 2-nearest-enemy 버전으로 교체.
        normalize_action 내부에서 task.get_obs가 호출되므로 반드시 패치 필요.
        """
        import types
        env_ref = self

        def patched_get_obs(task_self, env, agent_id):
            return env_ref._get_paired_obs(agent_id)

        self.task.get_obs = types.MethodType(patched_get_obs, self.task)

    def _update_obs_space(self):
        """obs_length = 9 + (1파트너 + NUM_NEAREST_ENEMIES + 1미사일) * 6 = 33 고정."""
        from gymnasium import spaces as gym_spaces
        self._obs_length = _OBS_LENGTH
        self.task.obs_length = self._obs_length
        self.task.observation_space = gym_spaces.Box(
            low=-10, high=10., shape=(self._obs_length,))
        self.task.share_observation_space = gym_spaces.Box(
            low=-10, high=10., shape=(self.num_agents * self._obs_length,))

    def _get_paired_obs(self, agent_id: str) -> np.ndarray:
        """
        HierarchicalMultipleCombatShootTask.get_obs 구조를 따르되
        적군은 ego 기준 가장 가까운 NUM_NEAREST_ENEMIES 기만 포함.

        Layout (33 dim):
          [0:9]   ego 상태
          [9:15]  파트너 상대 정보 (없으면 zeros)
          [15:21] 최근접 적1 상대 정보
          [21:27] 최근접 적2 상대 정보 (없으면 zeros)
          [27:33] 미사일 경고 정보 (없으면 zeros)
        """
        norm_obs = np.zeros(self._obs_length)
        state_var = self.task.state_var

        # ── (1) ego ────────────────────────────────────────────────────
        ego_state = np.array(
            self.agents[agent_id].get_property_values(state_var))
        ego_cur_ned = LLA2NEU(
            *ego_state[:3], self.center_lon, self.center_lat, self.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        norm_obs[0] = ego_state[2] / 5000
        norm_obs[1] = np.sin(ego_state[3])
        norm_obs[2] = np.cos(ego_state[3])
        norm_obs[3] = np.sin(ego_state[4])
        norm_obs[4] = np.cos(ego_state[4])
        norm_obs[5] = ego_state[9] / 340
        norm_obs[6] = ego_state[10] / 340
        norm_obs[7] = ego_state[11] / 340
        norm_obs[8] = ego_state[12] / 340

        # ── (2) 파트너 + 최근접 적 2기 ────────────────────────────────
        partner_id = self.partner_map.get(agent_id)
        partner_sims = (
            [self.agents[partner_id]]
            if partner_id and partner_id in self.agents else []
        )

        ego_sim = self.agents[agent_id]
        ego_lon, ego_lat, _ = ego_sim.get_geodetic()
        alive_enemies = [e for e in ego_sim.enemies if e.is_alive]
        alive_enemies.sort(
            key=lambda e: _haversine_m(ego_lon, ego_lat, *e.get_geodetic()[:2]))
        nearest_enemies = alive_enemies[:NUM_NEAREST_ENEMIES]

        offset = 8
        for target_sim in partner_sims + nearest_enemies:
            state = np.array(target_sim.get_property_values(state_var))
            cur_ned = LLA2NEU(
                *state[:3], self.center_lon, self.center_lat, self.center_alt)
            feature = np.array([*cur_ned, *(state[6:9])])
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)
            norm_obs[offset + 1] = (state[9] - ego_state[9]) / 340
            norm_obs[offset + 2] = (state[2] - ego_state[2]) / 1000
            norm_obs[offset + 3] = AO
            norm_obs[offset + 4] = TA
            norm_obs[offset + 5] = R / 10000
            norm_obs[offset + 6] = side_flag
            offset += 6

        norm_obs = np.clip(
            norm_obs,
            self.task.observation_space.low,
            self.task.observation_space.high,
        )

        # ── (3) 미사일 경고 ────────────────────────────────────────────
        missile_sim = ego_sim.check_missile_warning()
        if missile_sim is not None:
            missile_feature = np.concatenate(
                (missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(
                ego_feature, missile_feature, return_side=True)
            norm_obs[offset + 1] = (
                np.linalg.norm(missile_sim.get_velocity()) - ego_state[9]) / 340
            norm_obs[offset + 2] = (missile_feature[2] - ego_state[2]) / 1000
            norm_obs[offset + 3] = ego_AO
            norm_obs[offset + 4] = ego_TA
            norm_obs[offset + 5] = R / 10000
            norm_obs[offset + 6] = side_flag

        return norm_obs

    def _rl_action_ego(self, uid: str) -> np.ndarray:
        """아군 RL 정책 행동 (정책 없으면 임시 오토파일럿)."""
        if self.ego_policy is None:
            sim = self.agents[uid]
            enm_alive = [e for e in sim.enemies if e.is_alive]
            if enm_alive:
                lon, lat, _ = enm_alive[0].get_geodetic()
                return _autopilot_hierarchical_action(sim, lon, lat)
            return np.array([1, 2, 1, 0], dtype=np.int32)

        obs_arr = self._get_paired_obs(uid)[np.newaxis, :]
        obs_t = torch.from_numpy(obs_arr).float().to(self.device)
        mask_t = torch.ones((1, 1), dtype=torch.float32).to(self.device)
        rnn_t = torch.from_numpy(self._ego_rnn).to(self.device)
        with torch.no_grad():
            action_t, _, rnn_t = self.ego_policy(obs_t, rnn_t, mask_t, deterministic=True)
        self._ego_rnn = rnn_t.cpu().numpy()
        return action_t.cpu().numpy().squeeze(0).astype(np.int32)

    def _rl_action_enm(self, uid: str) -> np.ndarray:
        """적군 RL 정책 행동."""
        if self.enm_policy is None:
            sim = self.agents[uid]
            enm_alive = [e for e in sim.enemies if e.is_alive]
            if enm_alive:
                lon, lat, _ = enm_alive[0].get_geodetic()
                return _autopilot_hierarchical_action(sim, lon, lat)
            return np.array([1, 2, 1, 0], dtype=np.int32)

        obs_arr = self._get_paired_obs(uid)[np.newaxis, :]
        obs_t = torch.from_numpy(obs_arr).float().to(self.device)
        mask_t = torch.ones((1, 1), dtype=torch.float32).to(self.device)
        rnn_t = torch.from_numpy(self._enm_rnn).to(self.device)
        with torch.no_grad():
            action_t, _, rnn_t = self.enm_policy(obs_t, rnn_t, mask_t, deterministic=True)
        self._enm_rnn = rnn_t.cpu().numpy()
        return action_t.cpu().numpy().squeeze(0).astype(np.int32)

    # ------------------------------------------------------------------
    # 편대 중심 계산
    # ------------------------------------------------------------------

    def _formation_centroid(self, uids: List[str]) -> Tuple[float, float]:
        """살아있는 기체들의 위도/경도 평균."""
        alive = [u for u in uids if u in self.agents and self.agents[u].is_alive]
        if not alive:
            uids_use = uids
        else:
            uids_use = alive
        lons, lats = [], []
        for u in uids_use:
            if u in self.agents:
                lon, lat, _ = self.agents[u].get_geodetic()
                lons.append(lon)
                lats.append(lat)
        if not lons:
            return self.center_lon, self.center_lat
        return float(np.mean(lons)), float(np.mean(lats))

    # ------------------------------------------------------------------
    # 페어 거리 및 단계 전환
    # ------------------------------------------------------------------

    def _pair_min_distance_m(self, friendly_uids: List[str], enemy_uids: List[str]) -> float:
        """편대 쌍 최소 기체간 거리 (m)."""
        min_d = float("inf")
        for fu in friendly_uids:
            if fu not in self.agents or not self.agents[fu].is_alive:
                continue
            flon, flat, _ = self.agents[fu].get_geodetic()
            for eu in enemy_uids:
                if eu not in self.agents or not self.agents[eu].is_alive:
                    continue
                elon, elat, _ = self.agents[eu].get_geodetic()
                d = _haversine_m(flon, flat, elon, elat)
                min_d = min(min_d, d)
        return min_d

    def _update_pair_phases(self):
        for i, pair in enumerate(self.formation_pairs):
            if self.pair_phases[i] in ("done", "rtb_loss", "rtb_victory"):
                continue
            friendly_uids = pair["friendly_uids"]
            enemy_uids    = pair["enemy_uids"]
            if self.pair_phases[i] == "approach":
                dist = self._pair_min_distance_m(friendly_uids, enemy_uids)
                if dist <= APPROACH_DISTANCE_M:
                    logger.info(f"편대쌍 {i}: 접근 완료 ({dist/1000:.1f}km) → 교전 시작")
                    self.pair_phases[i] = "combat"

    # ------------------------------------------------------------------
    # 이벤트 감지
    # ------------------------------------------------------------------

    def _detect_events(self):
        step = self.current_step
        ts = step * self.time_interval

        # ── 50% 이상 손실 ──────────────────────────────────────────────
        if not self.event_major_loss:
            total_friendly = len(self.ego_ids)
            alive_friendly = sum(
                1 for uid in self.ego_ids
                if self.agents[uid].is_alive
            )
            if total_friendly > 0 and alive_friendly < total_friendly * 0.5:
                dead_list = [
                    uid for uid in self.ego_ids
                    if not self.agents[uid].is_alive
                ]
                details = {
                    "total_friendly": total_friendly,
                    "alive_friendly": alive_friendly,
                    "dead_aircraft": dead_list,
                    "loss_ratio": 1.0 - alive_friendly / total_friendly,
                }
                event_id = self.db.log_event(
                    self.sim_id, step, ts, "major_loss", details
                )
                self.event_major_loss = True
                self.event_major_loss_id = event_id
                logger.warning(
                    f"[이벤트] 아군 전력 50% 이상 손실 (event_id={event_id})"
                )

        # ── 개별 기체 무장 고갈 ────────────────────────────────────────
        for uid in self.ego_ids:
            if uid in self.event_ammo_depleted_ids:
                continue
            sim = self.agents[uid]
            if not sim.is_alive:
                continue
            if uid in self.reload_pending or uid in self.reloaded_returning:
                continue
            if sim.num_left_missiles == 0:
                details = {
                    "aircraft_uid": uid,
                    "missiles_left": 0,
                    "step": step,
                }
                event_id = self.db.log_event(
                    self.sim_id, step, ts, "ammo_depleted", details
                )
                self.event_ammo_depleted_ids[uid] = event_id
                self.event_ammo_depleted = True
                logger.warning(
                    f"[이벤트] {uid} 무장 고갈 (event_id={event_id})"
                )

        # ── 편대 단위 무장 고갈 ────────────────────────────────────────
        # 편대 내 생존 기체가 전부 미사일 0발이면 formation_ammo_depleted 발생
        for i, pair in enumerate(self.formation_pairs):
            if i in self.event_formation_ammo_depleted:
                continue
            if self.pair_phases[i] in ("rtb_loss", "rtb_ammo", "done", "rtb_victory"):
                continue
            alive_uids = [
                u for u in pair["friendly_uids"]
                if u in self.agents and self.agents[u].is_alive
                and u not in self.reload_pending
                and u not in self.reloaded_returning
            ]
            if not alive_uids:
                continue   # 생존 기체 없음 → 손실 이벤트가 이미 처리
            if all(self.agents[u].num_left_missiles == 0 for u in alive_uids):
                details = {
                    "pair_idx": i,
                    "friendly_base": pair["friendly_base"]["name"],
                    "enemy_base": pair["enemy_base"]["name"],
                    "alive_friendly": len(alive_uids),
                    "alive_enemy": sum(
                        1 for u in pair["enemy_uids"]
                        if u in self.agents and self.agents[u].is_alive
                    ),
                    "step": step,
                }
                event_id = self.db.log_event(
                    self.sim_id, step, ts, "formation_ammo_depleted", details
                )
                self.event_formation_ammo_depleted[i] = event_id
                logger.warning(
                    f"[이벤트] 편대 {i} ({pair['friendly_base']['name']}) "
                    f"전 기체 무장 고갈 (event_id={event_id})"
                )

        # ── 적 편대 전멸 (아군 승리) ────────────────────────────────────
        # enemy_uids 전원 사망, 아군 생존 기체 존재 → enemy_formation_destroyed 이벤트
        for i, pair in enumerate(self.formation_pairs):
            if i in self.event_enemy_formation_destroyed:
                continue
            if self.pair_phases[i] in ("rtb_loss", "rtb_ammo", "done", "rtb_victory"):
                continue
            alive_friendly = [
                u for u in pair["friendly_uids"]
                if u in self.agents and self.agents[u].is_alive
            ]
            if not alive_friendly:
                continue  # 아군도 없음 → formation_destroyed 쪽에서 처리
            alive_enemy = [
                u for u in pair["enemy_uids"]
                if u in self.agents and self.agents[u].is_alive
            ]
            if alive_enemy:
                continue  # 아직 적 생존
            details = {
                "pair_idx": i,
                "friendly_base": pair["friendly_base"]["name"],
                "enemy_base": pair["enemy_base"]["name"],
                "alive_friendly": len(alive_friendly),
                "friendly_uids": alive_friendly,
                "step": step,
            }
            event_id = self.db.log_event(
                self.sim_id, step, ts, "enemy_formation_destroyed", details
            )
            self.event_enemy_formation_destroyed[i] = event_id
            logger.info(
                f"[이벤트] 적 편대 {i} ({pair['enemy_base']['name']}) 전멸 — "
                f"아군 {len(alive_friendly)}대 생존 (event_id={event_id})"
            )

        # ── 아군 편대 전멸 ─────────────────────────────────────────────
        # friendly_uids 전원 사망 → formation_destroyed 이벤트
        for i, pair in enumerate(self.formation_pairs):
            if i in self.event_formation_destroyed:
                continue
            if self.pair_phases[i] in ("rtb_loss", "rtb_ammo", "done"):
                continue
            # 편대에 속한 전체 기체 (지원 포함) 가 모두 사망인지 확인
            all_uids = pair["friendly_uids"]
            if not all_uids:
                continue
            alive_any = any(
                u in self.agents and self.agents[u].is_alive
                for u in all_uids
            )
            if alive_any:
                continue
            # 전멸 확인. 아직 교전 중인 적이 남아있을 때만 이벤트 발생
            alive_enemy = [
                u for u in pair["enemy_uids"]
                if u in self.agents and self.agents[u].is_alive
            ]
            if not alive_enemy:
                continue  # 적도 없음 → 이미 종료
            details = {
                "pair_idx": i,
                "friendly_base": pair["friendly_base"]["name"],
                "enemy_base": pair["enemy_base"]["name"],
                "alive_enemy": len(alive_enemy),
                "enemy_uids": alive_enemy,
                "step": step,
            }
            event_id = self.db.log_event(
                self.sim_id, step, ts, "formation_destroyed", details
            )
            self.event_formation_destroyed[i] = event_id
            logger.warning(
                f"[이벤트] 편대 {i} ({pair['friendly_base']['name']}) 전멸 "
                f"(잔여 적기 {len(alive_enemy)}대, event_id={event_id})"
            )

    # ------------------------------------------------------------------
    # 이벤트 처리 (컨트롤러에서 LLM 결정 후 호출)
    # ------------------------------------------------------------------

    def handle_formation_ammo_rtb(self, pair_idx: int):
        """
        편대 단위 무장 고갈 처리: 해당 편대의 생존 기체 전부 기지 복귀.
        지원 편대가 스폰되면 이 편대를 교체.

        Parameters
        ----------
        pair_idx : 무장 고갈된 편대쌍 인덱스
        """
        pair = self.formation_pairs[pair_idx]
        home = pair["friendly_base"]
        enm_lon, enm_lat = self._formation_centroid(pair["enemy_uids"])

        for uid in pair["friendly_uids"]:
            if uid not in self.agents:
                continue
            sim = self.agents[uid]
            if not sim.is_alive:
                continue
            if uid in self.reload_pending or uid in self.reloaded_returning:
                continue
            # 재장착 없이 순수 RTB (무장 고갈 편대 복귀)
            self.reload_pending[uid] = {
                "home_lon": home["lon"],
                "home_lat": home["lat"],
                "combat_lon": enm_lon,
                "combat_lat": enm_lat,
                "_no_reload": True,   # 재장착 없이 그냥 복귀
            }
            logger.info(f"[처리] 편대{pair_idx} 무장고갈 RTB: {uid} → {home['name']}")

        self.pair_phases[pair_idx] = "rtb_ammo"

    def handle_major_loss_rtb(self):
        """아군 전체 RTB: 잔존 기체 기지 방향으로 전환."""
        logger.info("[처리] 아군 전체 RTB 명령")
        for i in range(len(self.formation_pairs)):
            if self.pair_phases[i] not in ("done",):
                self.pair_phases[i] = "rtb_loss"

    def handle_request_support(self, support_base: Dict, pair_idx: int):
        """
        지원 편대 스폰.

        Parameters
        ----------
        support_base : {"name":…, "lon":…, "lat":…}
        pair_idx     : 교전 중인 편대쌍 인덱스
        """
        if self._support_spawned.get(pair_idx, False):
            return
        pair = self.formation_pairs[pair_idx]
        combat_lon, combat_lat = self._formation_centroid(pair["enemy_uids"])

        # 새 UID (S=support 접두어)
        uid_a = f"S{pair_idx}100"
        uid_b = f"S{pair_idx}200"

        lon0, lat0, alt0 = (
            support_base["lon"],
            support_base["lat"],
            SUPPORT_SPAWN_ALT_M,
        )
        # JSBSim 초기 상태: 기지 위치에서 출발
        init_common = {
            "ic_long_gc_deg": lon0,
            "ic_lat_geod_deg": lat0,
            "ic_h_sl_ft": SUPPORT_SPAWN_ALT_M * 3.28084,
            "ic_psi_true_deg": float(_bearing_rad(lon0, lat0, combat_lon, combat_lat) * 180 / math.pi),
            "ic_u_fps": CRUISE_SPEED_FPS,
        }

        for uid in [uid_a, uid_b]:
            sim = AircraftSimulator(
                uid=uid,
                color="Blue",
                model="f16",
                init_state=init_common,
                origin=(self.center_lon, self.center_lat, self.center_alt),
                sim_freq=self.sim_freq,
                num_missiles=2,
            )
            # 적군을 enemies 로 연결
            for eu in pair["enemy_uids"]:
                if eu in self.agents and self.agents[eu].is_alive:
                    sim.enemies.append(self.agents[eu])
                    self.agents[eu].enemies.append(sim)
            self.add_temp_simulator(sim)
            # _save_states_to_db 은 self.agents + pair["friendly_uids"] 를 순회하므로
            # 두 곳 모두 등록해야 지도에 표시됨
            self.agents[uid] = sim
            pair["friendly_uids"].append(uid)

        self._support_spawned[pair_idx] = True
        self.pair_phases[pair_idx] = "support"
        logger.info(
            f"[처리] 지원 편대 스폰: {uid_a}, {uid_b} (기지: {support_base['name']})"
        )

    def handle_spawn_replacement(self, replacement_base: Dict, pair_idx: int):
        """
        전멸한 아군 편대를 대체하는 새 편대 스폰.

        handle_request_support() 와 달리 _support_spawned 제한 없이
        항상 스폰 가능하며, 고유 UID를 생성하기 위해 _spawn_counter 를 사용.

        Parameters
        ----------
        replacement_base : {"name":…, "lon":…, "lat":…}
        pair_idx         : 전멸된 편대쌍 인덱스
        """
        pair = self.formation_pairs[pair_idx]
        combat_lon, combat_lat = self._formation_centroid(pair["enemy_uids"])

        self._spawn_counter += 1
        uid_a = f"R{pair_idx}_{self._spawn_counter}A"
        uid_b = f"R{pair_idx}_{self._spawn_counter}B"

        lon0, lat0 = replacement_base["lon"], replacement_base["lat"]
        init_common = {
            "ic_long_gc_deg":  lon0,
            "ic_lat_geod_deg": lat0,
            "ic_h_sl_ft":      SUPPORT_SPAWN_ALT_M * 3.28084,
            "ic_psi_true_deg": float(
                _bearing_rad(lon0, lat0, combat_lon, combat_lat) * 180 / math.pi
            ),
            "ic_u_fps": CRUISE_SPEED_FPS,
        }

        for uid in [uid_a, uid_b]:
            sim = AircraftSimulator(
                uid=uid,
                color="Blue",
                model="f16",
                init_state=init_common,
                origin=(self.center_lon, self.center_lat, self.center_alt),
                sim_freq=self.sim_freq,
                num_missiles=2,
            )
            for eu in pair["enemy_uids"]:
                if eu in self.agents and self.agents[eu].is_alive:
                    sim.enemies.append(self.agents[eu])
                    self.agents[eu].enemies.append(sim)
            self.add_temp_simulator(sim)
            self.agents[uid] = sim
            pair["friendly_uids"].append(uid)

        self.pair_phases[pair_idx] = "approach"
        logger.info(
            f"[처리] 교체 편대 스폰: {uid_a}, {uid_b} "
            f"(기지: {replacement_base['name']}, pair={pair_idx})"
        )

    def handle_enemy_victory_engage(self, pair_idx: int, target_pair_idx: int):
        """
        적 편대 전멸 후 아군 편대를 다른 적 편대 방향으로 재배치.

        Parameters
        ----------
        pair_idx        : 승리한 아군 편대쌍 인덱스
        target_pair_idx : 새로 교전할 적 편대쌍 인덱스
        """
        pair        = self.formation_pairs[pair_idx]
        target_pair = self.formation_pairs[target_pair_idx]

        alive_target_enemies = [
            u for u in target_pair["enemy_uids"]
            if u in self.agents and self.agents[u].is_alive
        ]

        for uid in pair["friendly_uids"]:
            if uid not in self.agents or not self.agents[uid].is_alive:
                continue
            sim = self.agents[uid]
            # 죽은 적기 제거 후 새 적기 연결
            sim.enemies = [e for e in sim.enemies if e.is_alive]
            for eu in alive_target_enemies:
                enemy_sim = self.agents[eu]
                if enemy_sim not in sim.enemies:
                    sim.enemies.append(enemy_sim)
                if sim not in enemy_sim.enemies:
                    enemy_sim.enemies.append(sim)

        # pair의 enemy_uids를 타겟 편대의 것으로 업데이트
        pair["enemy_uids"] = list(target_pair["enemy_uids"])
        self.pair_phases[pair_idx] = "approach"
        logger.info(
            f"[처리] 편대{pair_idx} 승리 재배치: "
            f"{pair['friendly_base']['name']} → 적편대{target_pair_idx} "
            f"({target_pair['enemy_base']['name']}) 공격"
        )

    def handle_enemy_victory_rtb(self, pair_idx: int):
        """
        적 편대 전멸 후 아군 편대 기지 복귀.

        Parameters
        ----------
        pair_idx : 승리한 아군 편대쌍 인덱스
        """
        pair = self.formation_pairs[pair_idx]
        home = pair["friendly_base"]
        for uid in pair["friendly_uids"]:
            if uid not in self.agents or not self.agents[uid].is_alive:
                continue
            if uid in self.reload_pending or uid in self.reloaded_returning:
                continue
            self.reload_pending[uid] = {
                "home_lon":   home["lon"],
                "home_lat":   home["lat"],
                "combat_lon": home["lon"],
                "combat_lat": home["lat"],
                "_no_reload": True,
            }
        self.pair_phases[pair_idx] = "rtb_victory"
        logger.info(
            f"[처리] 편대{pair_idx} 승리 RTB: {home['name']}으로 복귀"
        )

    def handle_ammo_rtb(self, uid: str):
        """
        무장 고갈 기체 RTB → 기지 귀환 후 AIM-9L 5발 재장착 → 복귀 시작.
        """
        pair_idx = self._find_pair_for_uid(uid)
        if pair_idx is None:
            return
        pair = self.formation_pairs[pair_idx]
        home = pair["friendly_base"]
        enm_lon, enm_lat = self._formation_centroid(pair["enemy_uids"])

        self.reload_pending[uid] = {
            "home_lon": home["lon"],
            "home_lat": home["lat"],
            "combat_lon": enm_lon,
            "combat_lat": enm_lat,
        }
        logger.info(f"[처리] {uid} 재장착 RTB 시작 → {home['name']}")

    def _process_reload_aircraft(self):
        """
        reload_pending 기체가 기지에 도착하면 재장착 후 복귀 큐로 이동.
        reloaded_returning 기체가 교전 지역에 도착하면 대기 해제.
        """
        completed_reload = []
        for uid, info in self.reload_pending.items():
            if uid not in self.agents or not self.agents[uid].is_alive:
                completed_reload.append(uid)
                continue
            sim = self.agents[uid]
            lon, lat, _ = sim.get_geodetic()
            dist = _haversine_m(lon, lat, info["home_lon"], info["home_lat"])
            if dist <= RELOAD_DISTANCE_M:
                if info.get("_no_reload"):
                    # 무장 고갈 편대 복귀: 재장착·재출격 없이 임무 종료
                    completed_reload.append(uid)
                    logger.info(f"[처리] {uid} 무장고갈 복귀 완료. 임무 종료.")
                else:
                    # 일반 재장착 후 복귀
                    sim.num_left_missiles = RELOAD_MISSILES
                    sim.num_missiles = RELOAD_MISSILES
                    self.reloaded_returning[uid] = {
                        "combat_lon": info["combat_lon"],
                        "combat_lat": info["combat_lat"],
                        "missiles": RELOAD_MISSILES,
                    }
                    completed_reload.append(uid)
                    logger.info(f"[처리] {uid} 재장착 완료. 교전 지역 복귀 시작.")

        for uid in completed_reload:
            self.reload_pending.pop(uid, None)

        completed_return = []
        for uid, info in self.reloaded_returning.items():
            if uid not in self.agents or not self.agents[uid].is_alive:
                completed_return.append(uid)
                continue
            sim = self.agents[uid]
            lon, lat, _ = sim.get_geodetic()
            dist = _haversine_m(lon, lat, info["combat_lon"], info["combat_lat"])
            if dist <= APPROACH_DISTANCE_M:
                completed_return.append(uid)
                logger.info(f"[처리] {uid} 교전 지역 도착. 전투 재개.")
        for uid in completed_return:
            self.reloaded_returning.pop(uid, None)

    # ------------------------------------------------------------------
    # DB 저장
    # ------------------------------------------------------------------

    def _save_states_to_db(self):
        step = self.current_step
        ts = step * self.time_interval

        for i, pair in enumerate(self.formation_pairs):
            fid = self._friendly_formation_ids.get(i, i)
            eid = self._enemy_formation_ids.get(i, i)

            for uid in pair["friendly_uids"]:
                if uid not in self.agents:
                    continue
                self._save_one_aircraft(
                    uid=uid, team="friendly",
                    formation_id=fid,
                    base_name=pair["friendly_base"]["name"],
                    step=step, ts=ts,
                    phase=self._get_uid_phase(uid, i),
                )

            for uid in pair["enemy_uids"]:
                if uid not in self.agents:
                    continue
                self._save_one_aircraft(
                    uid=uid, team="enemy",
                    formation_id=eid,
                    base_name=pair["enemy_base"]["name"],
                    step=step, ts=ts,
                    phase=self.pair_phases[i],
                )

    def _save_one_aircraft(
        self, uid, team, formation_id, base_name, step, ts, phase
    ):
        sim = self.agents[uid]
        lon, lat, alt = sim.get_geodetic()
        _, _, heading_rad = sim.get_rpy()
        heading_deg = math.degrees(heading_rad) % 360
        vn, ve, vd = sim.get_velocity()
        speed = float(np.linalg.norm([vn, ve, vd]))
        missiles = getattr(sim, "num_left_missiles", 0)
        health = float(sim.bloods)

        if sim.is_alive:
            death_cause = "alive"
        elif health <= 0:
            death_cause = "shot_down"
        else:
            death_cause = "crashed"

        self.db.save_aircraft_state(
            sim_id=self.sim_id,
            step=step,
            timestamp=ts,
            aircraft_uid=uid,
            team=team,
            formation_id=formation_id,
            base_name=base_name,
            is_alive=sim.is_alive,
            lon=lon, lat=lat, alt=alt,
            heading_deg=heading_deg,
            speed_mps=speed,
            health=health,
            missiles_left=missiles,
            flight_phase=phase,
            death_cause=death_cause,
        )

    def _get_uid_phase(self, uid: str, pair_idx: int) -> str:
        if uid in self.reload_pending:
            return "reload"
        if uid in self.reloaded_returning:
            return "returning"
        return self.pair_phases[pair_idx]

    def _find_pair_for_uid(self, uid: str) -> Optional[int]:
        for i, pair in enumerate(self.formation_pairs):
            if uid in pair["friendly_uids"]:
                return i
        return None

    # ------------------------------------------------------------------
    # 임무완료 편대 재배정 (safe_return done=True 방지)
    # ------------------------------------------------------------------

    def _reassign_victorious_formations(self):
        """
        담당 적 편대를 전멸시킨 아군 편대를 아직 교전 중인 다른 적 편대로 재배정.

        self.step() 호출 전에 실행하여 safe_return.get_termination()이
        enemies 리스트를 검사하기 전에 enemies를 갱신함으로써 done=True 방지.

        동작:
          1. pair i 의 enemy_uids 가 전부 사망 → '임무완료' 편대로 판단
          2. 다른 pair j 에 생존 적기가 있으면:
             - pair i 아군 기체의 enemies 에 pair j 생존 적기 추가
             - pair j 적기의 enemies 에 pair i 생존 아군 기체 추가
             - pair i 의 enemy_uids 목록에 pair j 의 enemy_uids 병합
          3. pair i 를 _victory_reassigned 에 기록 → 중복 처리 방지
        """
        for i, pair in enumerate(self.formation_pairs):
            if i in self._victory_reassigned:
                continue
            if self.pair_phases.get(i) in ("rtb_loss", "rtb_ammo", "done", "rtb_victory"):
                continue

            # pair i 의 담당 적기 중 생존자 확인
            alive_enemy_in_pair = [
                u for u in pair["enemy_uids"]
                if u in self.agents and self.agents[u].is_alive
            ]
            if alive_enemy_in_pair:
                continue  # 아직 교전 중 → 재배정 불필요

            # pair i 아군 생존 기체
            alive_friendly = [
                u for u in pair["friendly_uids"]
                if u in self.agents and self.agents[u].is_alive
            ]
            if not alive_friendly:
                continue  # 아군도 없음

            # 생존 적기가 가장 많은 다른 pair 선택
            target_idx, target_enemy_sims = self._find_best_target_pair(i)
            if target_idx is None:
                continue  # 모든 적 전멸 → 진짜 임무 완료, done 허용

            # ── 아군 기체 enemies 갱신 ────────────────────────────────
            for fu in alive_friendly:
                f_sim = self.agents[fu]
                for e_sim in target_enemy_sims:
                    if e_sim not in f_sim.enemies:
                        f_sim.enemies.append(e_sim)

            # ── 적기 enemies 에 아군 추가 ─────────────────────────────
            friendly_sims = [self.agents[fu] for fu in alive_friendly]
            for e_sim in target_enemy_sims:
                for f_sim in friendly_sims:
                    if f_sim not in e_sim.enemies:
                        e_sim.enemies.append(f_sim)

            # ── pair enemy_uids 병합 (DB 추적용) ─────────────────────
            target_pair = self.formation_pairs[target_idx]
            for eu in target_pair["enemy_uids"]:
                if eu not in pair["enemy_uids"]:
                    pair["enemy_uids"].append(eu)

            self._victory_reassigned.add(i)
            logger.info(
                f"[재배정] 편대{i}({pair['friendly_base']['name']}) "
                f"담당 적 전멸 → 편대{target_idx} 지원 "
                f"(적기 {len(target_enemy_sims)}대, "
                f"아군 {len(alive_friendly)}대 추가)"
            )

    def _find_best_target_pair(self, exclude_idx: int) -> Tuple[Optional[int], List]:
        """생존 적기 수가 가장 많은 다른 pair 반환."""
        best_idx = None
        best_sims: List = []
        for j, pair in enumerate(self.formation_pairs):
            if j == exclude_idx:
                continue
            alive = [
                self.agents[eu]
                for eu in pair["enemy_uids"]
                if eu in self.agents and self.agents[eu].is_alive
            ]
            if alive and len(alive) > len(best_sims):
                best_idx = j
                best_sims = alive
        return best_idx, best_sims

    # ------------------------------------------------------------------
    # DB formation_id 등록 (컨트롤러에서 호출)
    # ------------------------------------------------------------------

    def register_formation_ids(
        self,
        friendly_ids: Dict[int, int],
        enemy_ids: Dict[int, int],
    ):
        self._friendly_formation_ids = friendly_ids
        self._enemy_formation_ids = enemy_ids

    # ------------------------------------------------------------------
    # 편대 타겟 배정 (LLM 결정 적용)
    # ------------------------------------------------------------------

    def apply_formation_assignment(self, assignment: List[Dict]):
        """
        LLM이 결정한 아군-적군 편대 배정을 실제 기체의 enemies 리스트에 적용.

        Parameters
        ----------
        assignment : list of dict
            [{"friendly_base": "성남공군기지", "enemy_base": "원산기지"}, ...]
            LLMCommander.assign_formation_targets() 의 반환값.

        동작:
          1. assignment 에서 아군 기지명 → 적군 기지명 매핑 추출
          2. formation_pairs 에서 각 pair 의 friendly_base.name 으로 매핑 검색
          3. 매핑된 enemy_base.name 을 가진 pair 의 enemy_uids 를 가져와
             아군 각 기체의 sim.enemies = [해당 적 편대 sim 목록] 으로 업데이트
          4. 적군 기체도 대칭 업데이트 (적군 기체의 enemies → 해당 아군 편대)
        """
        # 기지명 → pair_idx 역매핑
        friendly_base_to_idx: Dict[str, int] = {
            p["friendly_base"]["name"]: i for i, p in enumerate(self.formation_pairs)
        }
        enemy_base_to_idx: Dict[str, int] = {
            p["enemy_base"]["name"]: i for i, p in enumerate(self.formation_pairs)
        }

        for entry in assignment:
            f_base = entry.get("friendly_base", "")
            e_base = entry.get("enemy_base", "")
            f_idx = friendly_base_to_idx.get(f_base)
            e_idx = enemy_base_to_idx.get(e_base)
            if f_idx is None or e_idx is None:
                logger.warning(
                    f"apply_formation_assignment: 매핑 실패 — "
                    f"friendly_base={f_base!r}, enemy_base={e_base!r}"
                )
                continue

            f_pair = self.formation_pairs[f_idx]
            e_pair = self.formation_pairs[e_idx]

            # 아군 기체의 enemies → 해당 적군 편대 기체만
            enemy_sims = [
                self.agents[eu]
                for eu in e_pair["enemy_uids"]
                if eu in self.agents
            ]
            for fu in f_pair["friendly_uids"]:
                if fu in self.agents:
                    self.agents[fu].enemies = list(enemy_sims)

            # 적군 기체의 enemies → 해당 아군 편대 기체만
            friendly_sims = [
                self.agents[fu]
                for fu in f_pair["friendly_uids"]
                if fu in self.agents
            ]
            for eu in e_pair["enemy_uids"]:
                if eu in self.agents:
                    self.agents[eu].enemies = list(friendly_sims)

            logger.info(
                f"편대 배정 적용: {f_base} ({len(friendly_sims)}기) "
                f"↔ {e_base} ({len(enemy_sims)}기)"
            )
