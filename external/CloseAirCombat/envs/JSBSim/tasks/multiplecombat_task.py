import numpy as np
from gymnasium import spaces
from typing import Tuple
import torch

from ..tasks import SingleCombatTask
from ..core.catalog import Catalog as c
from ..core.simulatior import MissileSimulator
from ..reward_functions import AltitudeReward, PostureReward, EventDrivenReward, MissilePostureReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn
from ..utils.utils import get_AO_TA_R, LLA2NEU, get_root_dir
from ..model.baseline_actor import BaselineActor


class MultipleCombatTask(SingleCombatTask):
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            AltitudeReward(self.config),
            PostureReward(self.config),
            EventDrivenReward(self.config)
        ]

        self.termination_conditions = [
            SafeReturn(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]

    @property
    def num_agents(self) -> int:
        return 4

    def load_variables(self):
        self.state_var = [
            c.position_long_gc_deg,             # 0. lontitude  (unit: °)
            c.position_lat_geod_deg,            # 1. latitude   (unit: °)
            c.position_h_sl_m,                  # 2. altitude   (unit: m)
            c.attitude_roll_rad,                # 3. roll       (unit: rad)
            c.attitude_pitch_rad,               # 4. pitch      (unit: rad)
            c.attitude_heading_true_rad,        # 5. yaw        (unit: rad)
            c.velocities_v_north_mps,           # 6. v_north    (unit: m/s)
            c.velocities_v_east_mps,            # 7. v_east     (unit: m/s)
            c.velocities_v_down_mps,            # 8. v_down     (unit: m/s)
            c.velocities_u_mps,                 # 9. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 10. v_body_y  (unit: m/s)
            c.velocities_w_mps,                 # 11. v_body_z  (unit: m/s)
            c.velocities_vc_mps,                # 12. vc        (unit: m/s)
            c.accelerations_n_pilot_x_norm,     # 13. a_north   (unit: G)
            c.accelerations_n_pilot_y_norm,     # 14. a_east    (unit: G)
            c.accelerations_n_pilot_z_norm,     # 15. a_down    (unit: G)
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        self.obs_length = 9 + (self.num_agents - 1) * 6
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def get_obs(self, env, agent_id):
        norm_obs = np.zeros(self.obs_length)
        # (1) ego info normalization
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        norm_obs[0] = ego_state[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_state[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_state[3])           # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_state[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_state[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_state[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_state[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_state[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_state[12] / 340            # 8. ego vc   (unit: mh)(unit: 5G)
        # (2) relative inof w.r.t partner+enemies state
        offset = 8
        for sim in env.agents[agent_id].partners + env.agents[agent_id].enemies:
            state = np.array(sim.get_property_values(self.state_var))
            cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
            feature = np.array([*cur_ned, *(state[6:9])])
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)
            norm_obs[offset+1] = (state[9] - ego_state[9]) / 340
            norm_obs[offset+2] = (state[2] - ego_state[2]) / 1000
            norm_obs[offset+3] = AO
            norm_obs[offset+4] = TA
            norm_obs[offset+5] = R / 10000
            norm_obs[offset+6] = side_flag
            offset += 6
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value.
        """
        norm_act = np.zeros(4)
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
        norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
        return self._enforce_altitude_floor(env, agent_id, norm_act)

    def _enforce_altitude_floor(self, env, agent_id, norm_act: np.ndarray,
                                 min_altitude: float = None) -> np.ndarray:
        """에이전트가 최소 허용 고도 이하로 하강하지 못하도록 제어 입력을 보정한다.

        고도 <= min_altitude 이고 하강 중(v_down > 0)인 경우:
          - elevator를 nose-up 방향으로 클램프 (>= -0.5)
          - throttle을 최대값(0.9)으로 고정

        Args:
            env          : 환경 인스턴스
            agent_id     : 에이전트 ID
            norm_act     : 정규화된 연속 액션 [aileron, elevator, rudder, throttle]
            min_altitude : 최소 허용 고도 (m, MSL).
                           None 이면 config.min_altitude_floor 값 사용 (기본 3000 m).

        Returns:
            보정된 norm_act. 조건 미충족 시 원본을 그대로 반환.
        """
        agent = env.agents[agent_id]
        if not agent.is_alive:
            return norm_act

        if min_altitude is None:
            min_altitude = getattr(self.config, 'min_altitude_floor', 3000.0)

        altitude = agent.get_property_value(c.position_h_sl_m)
        v_down   = agent.get_property_value(c.velocities_v_down_mps)  # NED +값 = 하강

        if altitude <= min_altitude and v_down > 0:
            act = norm_act.copy()
            act[1] = min(act[1], -0.5)  # elevator: nose-up 방향으로 클램프
            act[3] = 0.9                # throttle: 최대 추력으로 고정
            return act
        return norm_act

    def get_reward(self, env, agent_id, info: dict = ...) -> Tuple[float, dict]:
        if env.agents[agent_id].is_alive:
            return super().get_reward(env, agent_id, info=info)
        else:
            return 0.0, info


class HierarchicalMultipleCombatTask(MultipleCombatTask):
    
    def __init__(self, config: str):
        super().__init__(config)
        self.lowlevel_policy = BaselineActor()
        self.lowlevel_policy.load_state_dict(torch.load(get_root_dir() + '/model/baseline_model.pt', map_location=torch.device('cpu')))
        self.lowlevel_policy.eval()
        self.norm_delta_altitude = np.array([0.1, 0, -0.1])
        self.norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, 0, np.pi / 12, np.pi / 6])
        self.norm_delta_velocity = np.array([0.05, 0, -0.05])

    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([3, 5, 3])

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        # generate low-level input_obs
        raw_obs = self.get_obs(env, agent_id)
        input_obs = np.zeros(12)
        # (1) delta altitude/heading/velocity
        input_obs[0] = self.norm_delta_altitude[action[0]]
        input_obs[1] = self.norm_delta_heading[action[1]]
        input_obs[2] = self.norm_delta_velocity[action[2]]
        # (2) ego info
        input_obs[3:12] = raw_obs[:9]
        input_obs = np.expand_dims(input_obs, axis=0)
        # output low-level action
        _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])
        action = _action.detach().cpu().numpy().squeeze(0)
        self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
        # normalize low-level action
        norm_act = np.zeros(4)
        norm_act[0] = action[0] / 20 - 1.
        norm_act[1] = action[1] / 20 - 1.
        norm_act[2] = action[2] / 20 - 1.
        norm_act[3] = action[3] / 58 + 0.4
        return self._enforce_altitude_floor(env, agent_id, norm_act)

    def go_dest(
        self,
        env,
        agent_id: str,
        dest_lon: float,
        dest_lat: float,
        target_alt_m: float = 6000.0,
        target_speed_mps: float = 200.0,
    ) -> np.ndarray:
        """
        목표 위도/경도로 기동하기 위한 high-level action을 계산한다.

        현재 기체의 헤딩·고도·속도와 목표 지점의 차이를 비교하여
        HierarchicalMultipleCombatShootTask 호환 action 인덱스를 반환한다.

        Parameters
        ----------
        env              : 환경 인스턴스
        agent_id         : 기체 UID
        dest_lon         : 목표 경도 (°)
        dest_lat         : 목표 위도 (°)
        target_alt_m     : 순항 목표 고도 (m, 기본 6000 m)
        target_speed_mps : 순항 목표 속도 (m/s, 기본 200 m/s)

        Returns
        -------
        action : np.ndarray, shape (4,), dtype int32
            [altitude_idx, heading_idx, velocity_idx, shoot_flag]

            altitude_idx  (0~2) :
                0 = 상승  (+0.1, norm_delta_altitude[0])
                1 = 유지  ( 0.0, norm_delta_altitude[1])
                2 = 하강  (-0.1, norm_delta_altitude[2])

            heading_idx   (0~4) :
                0 = 좌선회 강 (-30°, norm_delta_heading[0])
                1 = 좌선회 약 (-15°, norm_delta_heading[1])
                2 = 직진   (  0°, norm_delta_heading[2])
                3 = 우선회 약 (+15°, norm_delta_heading[3])
                4 = 우선회 강 (+30°, norm_delta_heading[4])

            velocity_idx  (0~2) :
                0 = 가속  (+0.05, norm_delta_velocity[0])
                1 = 유지  ( 0.0,  norm_delta_velocity[1])
                2 = 감속  (-0.05, norm_delta_velocity[2])

            shoot_flag    (0~1) :
                항법 중이므로 항상 0 (비사격)
        """
        agent = env.agents[agent_id]
        state = np.array(agent.get_property_values(self.state_var))

        cur_lat_deg     = state[1]   # 현재 위도 (°)
        cur_lon_deg     = state[0]   # 현재 경도 (°)
        cur_alt_m       = state[2]   # 현재 고도 (m)
        cur_heading_rad = state[5]   # 현재 헤딩 (rad, 0=North, CW)
        cur_speed_mps   = state[12]  # 현재 대기속도 (m/s)

        # ── 목표 방위각 계산 (정북 기준 CW, -π ~ π) ────────────────────
        cur_lat_rad  = np.deg2rad(cur_lat_deg)
        cur_lon_rad  = np.deg2rad(cur_lon_deg)
        dest_lat_rad = np.deg2rad(dest_lat)
        dest_lon_rad = np.deg2rad(dest_lon)

        dlon = dest_lon_rad - cur_lon_rad
        x = np.cos(dest_lat_rad) * np.sin(dlon)
        y = (np.cos(cur_lat_rad) * np.sin(dest_lat_rad)
             - np.sin(cur_lat_rad) * np.cos(dest_lat_rad) * np.cos(dlon))
        target_bearing_rad = np.arctan2(x, y)  # -π ~ π

        # ── 헤딩 오차 → 헤딩 액션 ───────────────────────────────────────
        # -π ~ π 로 정규화
        heading_err = (target_bearing_rad - cur_heading_rad + np.pi) % (2 * np.pi) - np.pi

        # 임계값: π/8 ≈ 22.5°(강), π/24 ≈ 7.5°(약)
        if heading_err < -np.pi / 8:
            heading_idx = 0   # 좌선회 강 (-30°)
        elif heading_err < -np.pi / 24:
            heading_idx = 1   # 좌선회 약 (-15°)
        elif heading_err > np.pi / 8:
            heading_idx = 4   # 우선회 강 (+30°)
        elif heading_err > np.pi / 24:
            heading_idx = 3   # 우선회 약 (+15°)
        else:
            heading_idx = 2   # 직진

        # ── 고도 오차 → 고도 액션 (데드밴드 ±200 m) ────────────────────
        alt_err = target_alt_m - cur_alt_m
        if alt_err > 200.0:
            altitude_idx = 0  # 상승
        elif alt_err < -200.0:
            altitude_idx = 2  # 하강
        else:
            altitude_idx = 1  # 유지

        # ── 속도 오차 → 속도 액션 (데드밴드 ±10 m/s) ───────────────────
        speed_err = target_speed_mps - cur_speed_mps
        if speed_err > 10.0:
            velocity_idx = 0  # 가속
        elif speed_err < -10.0:
            velocity_idx = 2  # 감속
        else:
            velocity_idx = 1  # 유지

        return np.array([altitude_idx, heading_idx, velocity_idx, 0], dtype=np.int32)

    def reset(self, env):
        """Task-specific reset, include reward function reset.
        """
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return super().reset(env)



class HierarchicalMultipleCombatShootTask(HierarchicalMultipleCombatTask):
    def __init__(self, config: str):
        super().__init__(config)
        self.max_attack_angle = getattr(self.config, 'max_attack_angle', 180)
        self.max_attack_distance = getattr(self.config, 'max_attack_distance', np.inf)
        self.min_attack_interval = getattr(self.config, 'min_attack_interval', 125)
        self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config)
        ]
    
    def load_observation_space(self):
        self.obs_length = 9 + self.num_agents  * 6
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))
    
    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([3, 5, 3, 2])


    def get_obs(self, env, agent_id):
        norm_obs = np.zeros(self.obs_length)
        # (1) ego info normalization
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        norm_obs[0] = ego_state[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_state[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_state[3])           # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_state[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_state[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_state[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_state[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_state[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_state[12] / 340            # 8. ego vc   (unit: mh)(unit: 5G)
        # (2) relative inof w.r.t partner+enemies state
        offset = 8
        for sim in env.agents[agent_id].partners + env.agents[agent_id].enemies:
            state = np.array(sim.get_property_values(self.state_var))
            cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
            feature = np.array([*cur_ned, *(state[6:9])])
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)
            norm_obs[offset+1] = (state[9] - ego_state[9]) / 340
            norm_obs[offset+2] = (state[2] - ego_state[2]) / 1000
            norm_obs[offset+3] = AO
            norm_obs[offset+4] = TA
            norm_obs[offset+5] = R / 10000
            norm_obs[offset+6] = side_flag
            offset += 6
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        # (3) missile info TODO: multiple missile and parnter's missile?
        missile_sim = env.agents[agent_id].check_missile_warning() #
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[offset + 1] = (np.linalg.norm(missile_sim.get_velocity()) - ego_state[9]) / 340
            norm_obs[offset + 2] = (missile_feature[2] - ego_state[2]) / 1000
            norm_obs[offset + 3] = ego_AO
            norm_obs[offset + 4] = ego_TA
            norm_obs[offset + 5] = R / 10000
            norm_obs[offset + 6] = side_flag
        return norm_obs

    def reset(self, env):
        """Reset fighter blood & missile status
        """
        self._last_shoot_time = {agent_id: -self.min_attack_interval for agent_id in env.agents.keys()}
        self._remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self._shoot_action = {agent_id: False for agent_id in env.agents.keys()}
        return super().reset(env)

    def normalize_action(self, env, agent_id, action):
        self._shoot_action[agent_id] = action[3] > 0
        return super().normalize_action(env, agent_id, action[:3])

    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]
            # Determine whether can launch missile at the nearest enemy aircraft
            target_list = list(map(lambda x: x.get_position() - agent.get_position(), agent.enemies))
            target_distance = list(map(np.linalg.norm, target_list))
            target_index = np.argmin(target_distance)
            target = target_list[target_index]
            heading = agent.get_velocity()
            distance = target_distance[target_index]
            attack_angle = np.rad2deg(np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
            shoot_interval = env.current_step - self._last_shoot_time[agent_id]

            shoot_flag = agent.is_alive and self._shoot_action[agent_id] and self._remaining_missiles[agent_id] > 0 \
                and attack_angle <= self.max_attack_angle and distance <= self.max_attack_distance and shoot_interval >= self.min_attack_interval
            if shoot_flag:
                new_missile_uid = agent_id + str(self._remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[target_index], uid=new_missile_uid))
                self._remaining_missiles[agent_id] -= 1
                self._last_shoot_time[agent_id] = env.current_step
