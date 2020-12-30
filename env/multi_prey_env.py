import random

from ray.rllib import MultiAgentEnv

from env.prey_env import PreyEnv


class MultiPreyEnv(MultiAgentEnv):
    def __init__(self, config):
        num = config.pop("num_agents", 1)
        self.agents = [PreyEnv(config) for _ in range(num)]
        self.dones = set()
        self.observation_space = self.agents[0].observation_space
        self.action_space = self.agents[0].action_space

    def reset(self):
        self.dones = set()
        return {i: a.reset() for i, a in enumerate(self.agents)}

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = self.agents[i].step(action, [random.randint(0, 3), random.randint(0, 3)])
            if done[i]:
                self.dones.add(i)
        done["__all__"] = len(self.dones) == len(self.agents)
        return obs, rew, done, info

    def get_pos(self):
        loc = [        ]
        for agent in self.agents:
            loc.append(agent.get_position())
        return loc

