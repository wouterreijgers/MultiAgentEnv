import math
import random
import pandas as pd
import ray
from dataclasses import dataclass

import gym
import torch
from gym import spaces, logger
import numpy as np
from gym.utils import seeding

"""
https://docs.ray.io/en/master/rllib-env.html
"""


# class HunterEnv(gym.Env):
class HunterEnv:
    """
    Description:
        The hunter will try to catch preys in order to survive, as soon as the hunter collected enough energy
        it can reproduce and make sure the species survives.
    Observation:
        Type: Box(4)
        Num     Observation               Min                Max
        0       Age                       0                  max_age defined in the simulation_param file
        1       Energy level              0                  100
        2       rel x to closest prey     0                  width defined in the simulation_param file
        3       rel y to closest prey     0                  Height defined in the simulation_param file
    Actions:
        Type: Discrete(5)
        Num   Action
        0     Reproduce, this is only possible if he has enough energy.
        1     Move up
        2     Move right
        3     Move down
        4     Move left
        Note: The hunter can not move out of the field defined in the simulation parameters, it can still perform the
        actions but it would result in no movement.
    Starting state:
        The hunter will start with age 0 and an energy level three times the amount of energy it receives
        from eating a prey. The other parameters will be defined randomly.
    Reward:
        to be decided
    Termination:
        The hunter dies when his energy is zero or when his age reaches the maximum age.
    """

    def __init__(self, config):
        # Static configurations
        self.max_age = config['hunters']['max_age']
        self.max_energy = 100
        self.width = config['sim']['width']
        self.height = config['sim']['height']
        high = np.array([self.max_age, self.max_energy, self.width, self.height], dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.action_shape = self.action_space.n
        self.observation_space = spaces.Box(np.array([0, 0, -self.width, -self.height]), high, dtype=np.float32)
        # print(self.observation_space)
        self.energy_to_reproduce = config['hunters']['energy_to_reproduce']
        self.energy_per_prey_eaten = config['hunters']['energy_per_prey_eaten']
        # self.preys = preys
        # Hunter specific
        self.age = 0
        self.energy = 3 * self.energy_per_prey_eaten
        self.x = random.randint(0, self.width)
        self.y = random.randint(0, self.height)
        # Used in the multiagent env
        self.type = "hunter"

        self.state = None
        self.steps_beyond_done = None
        self.done = False

    def step(self, action, dist_to_prey):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # print('step')

        # print(self.state)
        age, energy, x_to_prey, y_to_prey = self.state
        reproduce = False

        # cost of living
        age += 1
        energy -= 1

        reward = 0

        # perform the action
        if energy >= self.energy_to_reproduce + 1:
            # if action == 0 and self.energy >= self.energy_to_reproduce:
            energy -= self.energy_to_reproduce
            reward += .1
            reproduce = True
        if action == 1 and self.y < self.height - 2:
            self.y += 1
        if action == 2 and self.x < self.width - 2:
            self.x += 1
        if action == 3 and self.y > 0:
            self.y -= 1
        if action == 4 and self.x > 0:
            self.x -= 1

        # find closest prey and 'eat' if close enough
        x_to_prey, y_to_prey = dist_to_prey[0], dist_to_prey[1]  # self.preys.get_rel_x_y([self.x, self.y])
        if (abs(x_to_prey) + abs(y_to_prey)) <=1:
            reward += .1
            energy += self.energy_per_prey_eaten

        self.state = (age, energy, x_to_prey, y_to_prey)
        self.done = bool(
            age >= self.max_age
            or energy <= 0
        )

        if not self.done:
            reward += 1
        elif self.steps_beyond_done is None:
            # Hunter just died
            self.steps_beyond_done = 0
            reward += 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, self.done, {"reproduce": reproduce}

    def reset(self):
        self.state = (0, self.energy, random.randint(0, self.width), random.randint(0, self.height))
        self.done = False
        self.x = self.state[2]
        self.y = self.state[3]
        self.steps_beyond_done = None
        #print("reset hunter ", self.state)
        return np.array(self.state)

    def get_position(self):
        return torch.tensor(np.array([self.x, self.y]))

    def render(self, mode='human'):
        pass

    def close(self):
        pass
