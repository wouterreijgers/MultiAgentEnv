import math
import random

from ray.rllib import MultiAgentEnv

from env.hunter_env import HunterEnv
from env.multi_prey_env import MultiPreyEnv


class MultiHunterEnv(MultiAgentEnv):
    """
    By training this model you only train the hunters.
    The preys will behave randomly
    """
    def __init__(self, config):
        num = config.pop("num_hunters", 1)
        self.agents = [HunterEnv(config) for _ in range(num)]
        self.dones = set()
        self.observation_space = self.agents[0].observation_space
        self.action_space = self.agents[0].action_space
        self.multi_preys_env = MultiPreyEnv(config)
        self.preys = self.multi_preys_env.reset()

    def reset(self):
        self.dones = set()
        return {i: a.reset() for i, a in enumerate(self.agents)}

    def step(self, action_dict):
        """
        Make the agents perform a step, notice that first we let the preys take steps
        :param action_dict:
        :return:
        """

        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            dist_x = random.randint(0, 20)
            dist_y = random.randint(0, 20)
            obs[i], rew[i], done[i], info[i] = self.agents[i].step(action, [dist_x, dist_y])
            if done[i]:
                self.dones.add(i)
        done["__all__"] = len(self.dones) == len(self.agents)
        return obs, rew, done, info


