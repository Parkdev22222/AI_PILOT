import numpy as np
from typing import Tuple, Dict, Any, Optional
from gymnasium import spaces
from .env_base import BaseEnv
from ..tasks.multiplecombat_task import HierarchicalMultipleCombatShootTask, HierarchicalMultipleCombatTask, MultipleCombatTask
from ..utils.utils import LLA2NEU, get_AO_TA_R


class MultipleCombatEnv(BaseEnv):
    """
    MultipleCombatEnv is an multi-player competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        self._create_records = False

    @property
    def share_observation_space(self):
        return self.task.share_observation_space

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'multiplecombat':
            self.task = MultipleCombatTask(self.config)
        elif taskname == 'hierarchical_multiplecombat':
            self.task = HierarchicalMultipleCombatTask(self.config)
        elif taskname == 'hierarchical_multiplecombat_shoot':
            self.task = HierarchicalMultipleCombatShootTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (dict): {agent_id: initial observation}
            share_obs (dict): {agent_id: initial state}
        """
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        share_obs = self.get_state()
        return self._pack(obs), self._pack(share_obs)

    def reset_simulators(self):
        # Assign new initial condition here!
        for sim in self._jsbsims.values():
            sim.reload()
        self._tempsims.clear()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (dict): the agents' actions, each key corresponds to an agent_id

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                share_obs: agents' share observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}

        # apply actions
        action = self._unpack(action)
        for agent_id in self.agents.keys():
            a_action = self.task.normalize_action(self, agent_id, action[agent_id])
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in self._tempsims.values():
                sim.run()
        self.task.step(self)
        obs = self.get_obs()
        share_obs = self.get_state()

        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]
        ego_reward = np.mean([rewards[ego_id] for ego_id in self.ego_ids])
        enm_reward = np.mean([rewards[enm_id] for enm_id in self.enm_ids])
        for ego_id in self.ego_ids:
            rewards[ego_id] = [ego_reward]
        for enm_id in self.enm_ids:
            rewards[enm_id] = [enm_reward]

        dones = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]

        return self._pack(obs), self._pack(share_obs), self._pack(rewards), self._pack(dones), info


class MultipleCombatEnv_LLM(MultipleCombatEnv):
    """
    MultipleCombatEnv_LLM extends MultipleCombatEnv with dynamic N-aircraft support
    and explicit one-to-one partner pairing within each team.

    Key differences from MultipleCombatEnv:
    - Supports arbitrary N ego + M enemy aircraft (not fixed at 4).
    - At env init, ally aircraft are paired 1-to-1 in index order: [0↔1], [2↔3], …
      If a team has an odd count, the last aircraft gets no partner (partner obs = zeros).
    - Observation structure per agent:
        [0 :9 ] – ego state (9 features)
        [9 :15] – paired partner relative state (6 features, or all-zero if unpaired)
        [15:  ] – each enemy relative state (6 features × max_enemy_count, zero-padded)
    - All aircraft in _jsbsims are treated as RL agents (num_agents = total aircraft).
    """

    def __init__(self, config_name: str):
        super().__init__(config_name)
        self._build_partner_pairs()
        self._update_obs_space()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_agents(self) -> int:
        """All aircraft are RL agents."""
        return len(self._jsbsims)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _build_partner_pairs(self):
        """Pair ally aircraft one-to-one within each team (ego / enemy).

        Consecutive pairs: team[0]↔team[1], team[2]↔team[3], …
        If the team size is odd the last aircraft maps to None (no partner).
        The result is stored in ``self.partner_map``.
        """
        self.partner_map: Dict[str, Optional[str]] = {}
        for team_ids in [self.ego_ids, self.enm_ids]:
            for i in range(0, len(team_ids), 2):
                agent_a = team_ids[i]
                if i + 1 < len(team_ids):
                    agent_b = team_ids[i + 1]
                    self.partner_map[agent_a] = agent_b
                    self.partner_map[agent_b] = agent_a
                else:
                    # Odd aircraft out – no partner
                    self.partner_map[agent_a] = None

    def _update_obs_space(self):
        """Recompute and overwrite the task's observation space.

        obs_length = 9 (ego) + 6 (one partner slot) + 6 × max_enemy_count

        Using the maximum enemy count across teams keeps a uniform obs size
        even for asymmetric team compositions; unused enemy slots are zero-padded.
        """
        num_ego = len(self.ego_ids)
        num_enm = len(self.enm_ids)
        # Ego agents face enm_ids as enemies; enm agents face ego_ids.
        # Use the larger count so every agent has the same obs length.
        self._max_enemy_count = max(num_ego, num_enm)
        self._obs_length = 9 + 6 + self._max_enemy_count * 6

        # Patch task spaces so downstream code reads correct dimensions.
        self.task.obs_length = self._obs_length
        self.task.observation_space = spaces.Box(
            low=-10, high=10., shape=(self._obs_length,))
        self.task.share_observation_space = spaces.Box(
            low=-10, high=10., shape=(self.num_agents * self._obs_length,))

    # ------------------------------------------------------------------
    # Observation builders
    # ------------------------------------------------------------------

    def get_obs(self) -> Dict[str, np.ndarray]:
        """Return partner-paired observations for every agent."""
        return {
            agent_id: self._get_paired_obs(agent_id)
            for agent_id in self.agents.keys()
        }

    def get_state(self) -> Dict[str, np.ndarray]:
        """Return global shared state built from partner-paired observations."""
        obs = self.get_obs()
        state = np.hstack([obs[agent_id] for agent_id in self.agents.keys()])
        return {agent_id: state.copy() for agent_id in self.agents.keys()}

    def _get_paired_obs(self, agent_id: str) -> np.ndarray:
        """Build a single agent's observation using its assigned partner + all enemies.

        Observation layout
        ------------------
        Indices  0–8   : ego state (9 features)
        Indices  9–14  : paired partner relative state (6 features)
                         → all zeros when the agent has no partner (odd team)
        Indices 15–end : each enemy's relative state (6 features × max_enemy_count)
                         → trailing slots are zero-padded for agents with fewer enemies
        """
        norm_obs = np.zeros(self._obs_length)

        # ── (1) Ego state ──────────────────────────────────────────────
        ego_state = np.array(
            self.agents[agent_id].get_property_values(self.task.state_var))
        ego_cur_ned = LLA2NEU(
            *ego_state[:3], self.center_lon, self.center_lat, self.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])

        norm_obs[0] = ego_state[2] / 5000          # altitude      (unit: 5 km)
        norm_obs[1] = np.sin(ego_state[3])          # roll_sin
        norm_obs[2] = np.cos(ego_state[3])          # roll_cos
        norm_obs[3] = np.sin(ego_state[4])          # pitch_sin
        norm_obs[4] = np.cos(ego_state[4])          # pitch_cos
        norm_obs[5] = ego_state[9] / 340            # v_body_x      (unit: Mach)
        norm_obs[6] = ego_state[10] / 340           # v_body_y      (unit: Mach)
        norm_obs[7] = ego_state[11] / 340           # v_body_z      (unit: Mach)
        norm_obs[8] = ego_state[12] / 340           # vc            (unit: Mach)

        offset = 9

        # ── (2) Paired partner (6 features, zeros if no partner) ───────
        partner_id = self.partner_map.get(agent_id)
        if partner_id is not None:
            p_state = np.array(
                self.agents[partner_id].get_property_values(self.task.state_var))
            p_ned = LLA2NEU(
                *p_state[:3], self.center_lon, self.center_lat, self.center_alt)
            p_feature = np.array([*p_ned, *(p_state[6:9])])
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, p_feature, return_side=True)
            norm_obs[offset + 0] = (p_state[9] - ego_state[9]) / 340   # Δv_body_x
            norm_obs[offset + 1] = (p_state[2] - ego_state[2]) / 1000  # Δaltitude
            norm_obs[offset + 2] = AO
            norm_obs[offset + 3] = TA
            norm_obs[offset + 4] = R / 10000
            norm_obs[offset + 5] = side_flag
        # else: no partner → slot remains zero (masked)
        offset += 6

        # ── (3) Enemy relative states (6 features each, zero-padded) ──
        for enm_sim in self.agents[agent_id].enemies:
            e_state = np.array(
                enm_sim.get_property_values(self.task.state_var))
            e_ned = LLA2NEU(
                *e_state[:3], self.center_lon, self.center_lat, self.center_alt)
            e_feature = np.array([*e_ned, *(e_state[6:9])])
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, e_feature, return_side=True)
            norm_obs[offset + 0] = (e_state[9] - ego_state[9]) / 340   # Δv_body_x
            norm_obs[offset + 1] = (e_state[2] - ego_state[2]) / 1000  # Δaltitude
            norm_obs[offset + 2] = AO
            norm_obs[offset + 3] = TA
            norm_obs[offset + 4] = R / 10000
            norm_obs[offset + 5] = side_flag
            offset += 6
        # Remaining enemy slots (asymmetric teams) stay at zero automatically.

        norm_obs = np.clip(
            norm_obs, self.task.observation_space.low, self.task.observation_space.high)
        return norm_obs
