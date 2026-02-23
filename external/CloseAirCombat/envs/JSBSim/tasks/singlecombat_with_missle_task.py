import numpy as np
from gymnasium import spaces
from collections import deque

from .singlecombat_task import SingleCombatTask, HierarchicalSingleCombatTask
from ..reward_functions import AltitudeReward, PostureReward, MissilePostureReward, EventDrivenReward, ShootPenaltyReward
from ..core.simulatior import MissileSimulator
from ..utils.utils import LLA2NEU, get_AO_TA_R, in_range_rad


class SingleCombatDodgeMissileTask(SingleCombatTask):
    """This task aims at training agent to dodge missile attacking
    """
    def __init__(self, config):
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
        self.observation_space = spaces.Box(low=-10, high=10., shape=(21,))

    def load_action_space(self):
        # high-level control + shoot flag
        self.action_space = spaces.MultiDiscrete([3, 5, 3, 2])

    def normalize_action(self, env, agent_id, action):
        action = np.asarray(action)
        # [altitude, heading, velocity, shoot_flag] -> use first 3 for control
        flight_action = action[:3].astype(np.int32)
        return HierarchicalSingleCombatTask.normalize_action(self, env, agent_id, flight_action)

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space

        ------
        Returns: (np.ndarray)
        - ego info
            - [0] ego altitude           (unit: 5km)
            - [1] ego_roll_sin
            - [2] ego_roll_cos
            - [3] ego_pitch_sin
            - [4] ego_pitch_cos
            - [5] ego v_body_x           (unit: mh)
            - [6] ego v_body_y           (unit: mh)
            - [7] ego v_body_z           (unit: mh)
            - [8] ego_vc                 (unit: mh)
        - relative enm info
            - [9] delta_v_body_x         (unit: mh)
            - [10] delta_altitude        (unit: km)
            - [11] ego_AO                (unit: rad) [0, pi]
            - [12] ego_TA                (unit: rad) [0, pi]
            - [13] relative distance     (unit: 10km)
            - [14] side_flag             1 or 0 or -1
        - relative missile info
            - [15] delta_v_body_x
            - [16] delta altitude
            - [17] ego_AO
            - [18] ego_TA
            - [19] relative distance
            - [20] side flag
        """
        norm_obs = np.zeros(21)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *ego_obs_list[6:9]])
        enm_feature = np.array([*enm_cur_ned, *enm_obs_list[6:9]])
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000
        norm_obs[1] = np.sin(ego_obs_list[3])
        norm_obs[2] = np.cos(ego_obs_list[3])
        norm_obs[3] = np.sin(ego_obs_list[4])
        norm_obs[4] = np.cos(ego_obs_list[4])
        norm_obs[5] = ego_obs_list[9] / 340
        norm_obs[6] = ego_obs_list[10] / 340
        norm_obs[7] = ego_obs_list[11] / 340
        norm_obs[8] = ego_obs_list[12] / 340
        # (2) relative enm info (radar-gated)
        radar_hits = env.agents[agent_id].get_radar_detections()
        enm_detected = any(hit['target_id'] == env.agents[agent_id].enemies[0].uid for hit in radar_hits)
        if enm_detected:
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
            norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
            norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
            norm_obs[11] = ego_AO
            norm_obs[12] = ego_TA
            norm_obs[13] = R / 10000
            norm_obs[14] = side_flag
        # (3) relative missile info
        missile_sim = env.agents[agent_id].check_missile_warning()
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[15] = (np.linalg.norm(missile_sim.get_velocity()) - ego_obs_list[9]) / 340
            norm_obs[16] = (missile_feature[2] - ego_obs_list[2]) / 1000
            norm_obs[17] = ego_AO
            norm_obs[18] = ego_TA
            norm_obs[19] = R / 10000
            norm_obs[20] = side_flag
        return norm_obs

    def reset(self, env):
        """Reset fighter blood & missile status
        """
        self._last_shoot_time = {agent_id: -self.min_attack_interval for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.lock_duration = {agent_id: deque(maxlen=int(1 / env.time_interval)) for agent_id in env.agents.keys()}
        return super().reset(env)

    def mask_action(self, env, agent_id, action):
        """Mask high-level action [altitude, heading, velocity, shoot_flag].

        Action semantics (discrete):
            - altitude: 0 (down), 1 (hold), 2 (up)
            - heading: 0 (left) ... 4 (right)
            - velocity: 0 (min) ... 2 (max)

        Missile warning branch uses missile direction/speed to induce a
        perpendicular break turn so missile turn-demand/energy loss increases.
        """
        action_arr = np.array(action, copy=True)
        if action_arr.shape[-1] < 4:
            return action_arr

        flight_action = action_arr[:3].copy()
        shoot_flag = action_arr[3:4].copy()

        def _pack(masked_flight_action):
            return np.concatenate([masked_flight_action, shoot_flag], axis=-1)

        def _apply_max_speed(masked_flight_action):
            # velocity index semantics: 2 == max speed
            masked_flight_action[2] = 2
            return masked_flight_action

        missile_sim = env.agents[agent_id].check_missile_warning()
        if missile_sim is None or not missile_sim.is_alive:
            # No missile warning: steer toward the nearest alive enemy until 10 km.
            enemies = [enemy for enemy in env.agents[agent_id].enemies if enemy.is_alive]
            if not enemies:
                return _pack(flight_action)

            ego_pos = np.array(env.agents[agent_id].get_position(), dtype=np.float64)
            closest_enemy = min(enemies, key=lambda enemy: np.linalg.norm(np.array(enemy.get_position(), dtype=np.float64) - ego_pos))
            rel_vec = np.array(closest_enemy.get_position(), dtype=np.float64) - ego_pos
            rel_distance = np.linalg.norm(rel_vec)

            # Already close enough: keep policy action.
            if rel_distance <= 10000.0:
                return _pack(flight_action)

            rel_xy = rel_vec[:2]
            rel_xy_norm = np.linalg.norm(rel_xy)
            if rel_xy_norm < 1e-6:
                return _pack(flight_action)

            ego_vel = np.array(env.agents[agent_id].get_velocity(), dtype=np.float64)
            ego_xy = ego_vel[:2]
            ego_heading = np.arctan2(ego_xy[1], ego_xy[0]) if np.linalg.norm(ego_xy) > 1e-6 else 0.0
            enemy_heading = np.arctan2(rel_xy[1], rel_xy[0])
            azimuth = in_range_rad(enemy_heading - ego_heading)
            elevation = np.arctan2(rel_vec[2], rel_xy_norm)

            masked = flight_action.copy()
            # Turn toward enemy bearing.
            masked[1] = 0 if azimuth > 0 else 4
            # Climb/descend toward enemy altitude if vertical offset is meaningful.
            if elevation > np.deg2rad(5.0):
                masked[0] = 2
            elif elevation < -np.deg2rad(5.0):
                masked[0] = 0
            # Keep current velocity command from policy.
            return _pack(masked)

        ego_velocity = np.array(env.agents[agent_id].get_velocity(), dtype=np.float64)
        missile_velocity = np.array(missile_sim.get_velocity(), dtype=np.float64)

        ego_xy = ego_velocity[:2]
        missile_xy = missile_velocity[:2]
        missile_speed = np.linalg.norm(missile_velocity)

        masked = _apply_max_speed(flight_action.copy())

        # Degenerate case: missile direction unavailable -> keep max speed only.
        if np.linalg.norm(missile_xy) < 1e-6:
            return _pack(masked)

        # Choose a perpendicular heading (left/right) against missile approach direction.
        ego_heading = np.arctan2(ego_xy[1], ego_xy[0]) if np.linalg.norm(ego_xy) > 1e-6 else 0.0
        missile_heading = np.arctan2(missile_xy[1], missile_xy[0])
        candidate_left = missile_heading + np.pi / 2
        candidate_right = missile_heading - np.pi / 2

        left_delta = abs(in_range_rad(candidate_left - ego_heading))
        right_delta = abs(in_range_rad(candidate_right - ego_heading))
        turn_left = left_delta <= right_delta
        masked[1] = 0 if turn_left else 4

        # Missile speed-aware vertical break: faster missile -> stronger vertical split.
        if missile_speed >= 350.0:
            # If missile is climbing toward us, break down; if diving, break up.
            missile_vz = missile_velocity[2]
            if missile_vz > 0:
                masked[0] = 0
            elif missile_vz < 0:
                masked[0] = 2
            else:
                masked[0] = 0 if turn_left else 2
        else:
            masked[0] = 1

        return _pack(masked)

    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [Rule-based missile launch]
            target = agent.enemies[0].get_position() - agent.get_position()
            heading = agent.get_velocity()
            distance = np.linalg.norm(target)
            attack_angle = np.rad2deg(np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
            self.lock_duration[agent_id].append(attack_angle < self.max_attack_angle)
            shoot_interval = env.current_step - self._last_shoot_time[agent_id]

            shoot_flag = agent.is_alive and np.sum(self.lock_duration[agent_id]) >= self.lock_duration[agent_id].maxlen \
                and distance <= self.max_attack_distance and self.remaining_missiles[agent_id] > 0 and shoot_interval >= self.min_attack_interval
            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1
                self._last_shoot_time[agent_id] = env.current_step


class HierarchicalSingleCombatDodgeMissileTask(HierarchicalSingleCombatTask, SingleCombatDodgeMissileTask):

    def __init__(self, config: str):
        HierarchicalSingleCombatTask.__init__(self, config)

        self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config)
        ]

    def load_observation_space(self):
        return SingleCombatDodgeMissileTask.load_observation_space(self)

    def load_action_space(self):
        # high-level control + shoot flag
        self.action_space = spaces.MultiDiscrete([3, 5, 3, 2])

    def get_obs(self, env, agent_id):
        return SingleCombatDodgeMissileTask.get_obs(self, env, agent_id)

    def normalize_action(self, env, agent_id, action):
        action = np.asarray(action)
        flight_action = action[:3].astype(np.int32)
        return HierarchicalSingleCombatTask.normalize_action(self, env, agent_id, flight_action)

    def reset(self, env):
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return SingleCombatDodgeMissileTask.reset(self, env)

    def step(self, env):
        return SingleCombatDodgeMissileTask.step(self, env)


class SingleCombatShootMissileTask(SingleCombatDodgeMissileTask):
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            PostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config),
            ShootPenaltyReward(self.config)
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(21,))

    def load_action_space(self):
        # high-level control + shoot flag
        self.action_space = spaces.MultiDiscrete([3, 5, 3, 2])

    def normalize_action(self, env, agent_id, action):
        action = np.asarray(action)
        # [altitude, heading, velocity, shoot_flag] -> use first 3 for control
        flight_action = action[:3].astype(np.int32)
        return HierarchicalSingleCombatTask.normalize_action(self, env, agent_id, flight_action)

    def load_action_space(self):
        # aileron, elevator, rudder, throttle, shoot control
        self.action_space = spaces.Tuple([spaces.MultiDiscrete([41, 41, 41, 30]), spaces.Discrete(2)])
    
    def get_obs(self, env, agent_id):
        return super().get_obs(env, agent_id)
    
    def normalize_action(self, env, agent_id, action):
        self._shoot_action[agent_id] = action[-1]
        return super().normalize_action(env, agent_id, action[:-1].astype(np.int32))
    
    def reset(self, env):
        self._shoot_action = {agent_id: 0 for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        super().reset(env)
    
    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]
            shoot_flag = agent.is_alive and self._shoot_action[agent_id] and self.remaining_missiles[agent_id] > 0
            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1


class HierarchicalSingleCombatShootTask(HierarchicalSingleCombatTask, SingleCombatShootMissileTask):
    def __init__(self, config: str):
        HierarchicalSingleCombatTask.__init__(self, config)
        self.reward_functions = [
            PostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config),
            ShootPenaltyReward(self.config)
        ]

    def load_observation_space(self):
        return SingleCombatShootMissileTask.load_observation_space(self)

    def load_action_space(self):
        # altitude control + heading control + velocity control + shoot control
        self.action_space = spaces.Tuple([spaces.MultiDiscrete([3, 5, 3]), spaces.Discrete(2)])

    def get_obs(self, env, agent_id):
        return SingleCombatShootMissileTask.get_obs(self, env, agent_id)

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        self._shoot_action[agent_id] = action[-1]
        return HierarchicalSingleCombatTask.normalize_action(self, env, agent_id, action[:-1].astype(np.int32))

    def reset(self, env):
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        SingleCombatShootMissileTask.reset(self, env)

    def step(self, env):
        SingleCombatShootMissileTask.step(self, env)
