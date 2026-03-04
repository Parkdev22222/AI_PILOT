from .reward_function_base import BaseRewardFunction


class EventDrivenReward(BaseRewardFunction):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Shot down by missile: -200
    - Crash accidentally: -200
    - Missile launch event: +100
    - Enemy shotdown: +100
    """
    def __init__(self, config):
        super().__init__(config)
        self.pre_launch_counts = {}
        self.pre_enemy_shotdowns = {}

    def reset(self, task, env):
        self.pre_launch_counts = {
            agent_id: len(agent.launch_missiles)
            for agent_id, agent in env.agents.items()
        }
        self.pre_enemy_shotdowns = {
            agent_id: sum(enemy.is_shotdown for enemy in agent.enemies)
            for agent_id, agent in env.agents.items()
        }
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the events.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        reward = 0
        agent = env.agents[agent_id]
        if env.agents[agent_id].is_shotdown:
            reward -= 200
        elif env.agents[agent_id].is_crash:
            reward -= 200

        launch_count = len(agent.launch_missiles)
        new_launches = max(0, launch_count - self.pre_launch_counts.get(agent_id, launch_count))
        reward += 100 * new_launches
        self.pre_launch_counts[agent_id] = launch_count

        enemy_shotdown_count = sum(enemy.is_shotdown for enemy in agent.enemies)
        new_enemy_shotdowns = max(0, enemy_shotdown_count - self.pre_enemy_shotdowns.get(agent_id, enemy_shotdown_count))
        reward += 100 * new_enemy_shotdowns
        self.pre_enemy_shotdowns[agent_id] = enemy_shotdown_count

        return self._process(reward, agent_id)
