from torch import nn, cat
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from gym.spaces import Discrete, Box


class DQNHunterModel(nn.Module, TorchModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.obs_space = obs_space
        self.action_space = action_space
        self.model_config = model_config
        self.name = name

        if isinstance(self.obs_space, Box):
            self.obs_shape = obs_space.shape[0]
        else:
            self.obs_shape = self.obs_space

        self.layer_config = model_config["custom_model_config"]["layers"]
        self.layers = nn.Sequential()
        linear_count = 0
        relu_count = 0
        for layer in self.layer_config:
            print("layer: ", layer["type"])
            if layer["type"] == "linear":
                linear_count += 1
                self.layers.add_module("linear_" + str(linear_count), nn.Linear(layer["input"], layer["output"]))
            elif layer["type"] == "relu":
                relu_count += 1
                self.layers.add_module("relu_" + str(relu_count), nn.ReLU())
        
    @override(TorchModelV2)
    def forward(self, obs):
        return self.layers(obs)
