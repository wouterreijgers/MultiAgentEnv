import ray
import os
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from dqn.dqn import DQNTrainer
from dqn.dqn_model import DQNModel
from env.hunter_env import HunterEnv
from env.multi_hunter_env import MultiHunterEnv
from env.multi_prey_hunter import MultiPreyHunterEnv


def env_creator(env_config):
    return MultiHunterEnv(env_config)
    #return MultiPreyHunterEnv(env_config)


if __name__ == "__main__":
    ray.init()
    ModelCatalog.register_custom_model("DQNModel", DQNModel)

    env_config = {
        'num_hunters': 20,
        'num_preys': 100,
        'hunters': {
            'start_amount': 20,
            'energy_to_reproduce': 30,
            'energy_per_prey_eaten': 10,
            'max_age': 20, },
        'preys': {
            'start_amount': 100,
            'birth_rate': 17,
            'max_age': 20},
        'sim': {
            'width': 200,
            'height': 200}
    }

    env = register_env("MultiHunterEnv-v0", env_creator)

    tune.run(
        DQNTrainer,
        # checkpoint_freq=10,
        checkpoint_at_end=True,
        stop={"timesteps_total": 200000},
        config={
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
                "custom_model": "DQNModel",
                "custom_model_config": {
                    "network_size": [32, 64, 32],
                },  # extra options to pass to your model
            },


            ########################################
            # Envaluation parameters
            ########################################
            "evaluation_interval": 100, # based on training iterations
            "evaluation_num_episodes": 100,
            "evaluation_config": {
                "epsilon": -1,
            },
        }
    )
