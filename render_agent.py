import time

import ray
import json
import gym
import numpy as np

from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from dqn.dqn import DQNTrainer
from dqn.dqn_model import DQNModel
from dqn.hunter_model import DQNHunterModel
from dqn.hunter_policy import DQNHunterPolicy
from dqn.prey_model import DQNPreyModel
from dqn.prey_policy import DQNPreyPolicy
from env.multi_prey_hunter import MultiPreyHunterEnv


def env_creator(env_config):
    # return MultiHunterEnv(env_config)
    return MultiPreyHunterEnv(env_config)


if __name__ == "__main__":

    # Settings
    folder = "/home/wouter/ray_results/DQNAlgorithm_2020-12-19_15-04-15/DQNAlgorithm_MultiHunterEnv-v0_75e3d_00000_0_2020-12-19_15-04-15"
    env_name = "MultiHunterEnv-v0"
    checkpoint = 10
    num_episodes = 1

    # Def env
    env = register_env(env_name, env_creator)
    print(folder + "/params.json")

    ray.init()
    ModelCatalog.register_custom_model("DQNPreyModel", DQNPreyModel)
    ModelCatalog.register_custom_model("DQNHunterModel", DQNHunterModel)

    env_config = {
        'num_hunters': 20,
        'num_preys': 100,
        'training': False,
        'hunters': {
            'start_amount': 20,
            'energy_to_reproduce': 30,
            'energy_per_prey_eaten': 10,
            'max_age': 20, },
        'preys': {
            'start_amount': 100,
            'birth_rate': 5,
            'max_age': 20},
        'sim': {
            'width': 200,
            'height': 200}
    }

    # test_env = MultiPreyHunterEnv(env_config)
    # policies = {"hunter": (DQNHunterPolicy,
    #                        test_env.observation_space_hunter,
    #                        test_env.action_space_hunter,
    #                        policy_config["hunter_policy_config"]),
    #             "prey": (DQNPreyPolicy,
    #                      test_env.observation_space_prey,
    #                      test_env.action_space_prey,
    #                      policy_config["prey_policy_config"])}
    #
    #
    # def policy_mapping_fn(agent_id):
    #     if "hunter" in agent_id:
    #         return "hunter"
    #     else:
    #         return "prey"

    policy_config = {
        "hunter_policy_config": {
            "num_gpus": 0,
            "num_workers": 1,
            "framework": "torch",
            # "sample_batch_size": 50,
            "env": "MultiHunterEnv-v0",

            ########################################
            # Parameters Agent
            ########################################
            "lr": 4e-3,
            # "lr": tune.grid_search([5e-3, 2e-3, 1e-3, 5e-4]),
            "gamma": 0.985,
            # "gamma": tune.grid_search([0.983, 0.985, 0.986, 0.987, 0.988, 0.989]),
            "epsilon": 1,
            "epsilon_decay": 0.99998,
            "epsilon_min": 0.01,
            "buffer_size": 20000,
            "batch_size": 2000,
            "env_config": env_config,
            "dqn_model": {
                "custom_model": "DQNHunterModel",
                "custom_model_config": {
                    "network_size": [32, 64, 32],
                },  # extra options to pass to your model
            },
        },
        "prey_policy_config": {
            "num_gpus": 0,
            "num_workers": 1,
            "framework": "torch",
            # "sample_batch_size": 50,
            "env": "MultiHunterEnv-v0",

            ########################################
            # Parameters Agent
            ########################################
            "lr": 4e-3,
            # "lr": tune.grid_search([5e-3, 2e-3, 1e-3, 5e-4]),
            "gamma": 0.985,
            # "gamma": tune.grid_search([0.983, 0.985, 0.986, 0.987, 0.988, 0.989]),
            "epsilon": 1,
            "epsilon_decay": 0.99998,
            "epsilon_min": 0.01,
            "buffer_size": 20000,
            "batch_size": 2000,
            "env_config": env_config,
            "dqn_model": {
                "custom_model": "DQNPreyModel",
                "custom_model_config": {
                    "network_size": [32, 64, 32],
                },  # extra options to pass to your model
            },
        }

    }


    def policy_mapping_fn(agent_id):
        if "hunter" in agent_id:
            return "hunter"
        else:
            return "prey"


    # Load config
    with open(folder + "/params.json") as json_file:
        config = json.load(json_file)
    print(config)
    # Sommigconfigs moeten opnieuw gedaan worden -> opgeslagen als string, word niet meer herkernd
    test_env = MultiPreyHunterEnv(env_config)
    policies = {"hunter": (DQNHunterPolicy,
                           test_env.observation_space_hunter,
                           test_env.action_space_hunter,
                           policy_config["hunter_policy_config"]),
                "prey": (DQNPreyPolicy,
                         test_env.observation_space_prey,
                         test_env.action_space_prey,
                         policy_config["prey_policy_config"])}

    config["multiagent"] = {
            "policy_mapping_fn": policy_mapping_fn,
            "policies": policies,
            "policies_to_train": policies
        }
    trainer = DQNTrainer(env=env_name,
                         config=config)
    # Restore checkpoint
    trainer.restore(folder + "/checkpoint_{}/checkpoint-{}".format(checkpoint, checkpoint))

    avg_reward = 0
    for episode in range(num_episodes):
        step = 0
        total_reward = 0
        hunter_reward = 0
        prey_reward = 0
        done = False
        observation = test_env.reset()

        while not done:
            step += 1
            time.sleep(0.2)
            test_env.render()
            #print(observation)
            action = {}
            for i, obs in observation.items():
                if step>1:
                    if not dones[i]:
                        action[i] = trainer.get_policy(policy_mapping_fn(i)).compute_actions([obs], [])[0][0]
                else:
                    action[i] = trainer.get_policy(policy_mapping_fn(i)).compute_actions([obs], [])[0][0]
            #action, _, _ = trainer.get_policy().compute_actions([observation], [])
            observation, reward, dones, info = test_env.step(action)
            for i, rew in reward.items():
                if not dones[i]:
                    if "hunter" in i:
                        hunter_reward += rew
                    else:
                        prey_reward += rew
            total_reward += hunter_reward + prey_reward

            print(prey_reward)

            if dones['__all__']:
                done = True
                test_env.render()
        avg_reward += total_reward
    print('avg reward after {} episodes {}'.format(avg_reward / num_episodes, num_episodes))
    test_env.close()
    del trainer
