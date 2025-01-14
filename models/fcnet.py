#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import Sequence

import gym
import numpy as np
from ray.rllib.models import ModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, List, ModelConfigDict

from typing import Union
import tensorflow as tf
TensorType = Union[np.array, "tf.Tensor", "torch.Tensor"]

torch, nn = try_import_torch()


class MyFullyConnectedNetwork(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **cfg,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.n_agents = len(obs_space.original_space)
        self.outputs_per_agent = num_outputs // self.n_agents
        self.trainer = cfg["trainer"]
        self.pos_dim = cfg["pos_dim"]
        self.pos_start = cfg["pos_start"]
        self.vel_start = cfg["vel_start"]
        self.vel_dim = cfg["vel_dim"]
        self.use_beta = cfg["use_beta"]
        self.add_agent_index = cfg["add_agent_index"]
        self.use_beta = False

        assert not cfg["share_observations"]

        self.obs_shape = obs_space.original_space[0].shape[0]
        # Remove position
        self.obs_shape -= self.pos_dim
        if self.add_agent_index:
            self.obs_shape += 1

        self.agent_networks = nn.ModuleList(
            [ MyFullyConnectedNetworkInner( self.obs_shape, self.outputs_per_agent, model_config ) ]
        )
        self.share_init_hetero_networks()

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        batch_size = input_dict["obs"][0].shape[0]
        device = input_dict["obs"][0].device

        obs = torch.stack(input_dict["obs"], dim=1)
        if self.add_agent_index:
            agent_index = ( torch.arange(self.n_agents, device=device).repeat(batch_size, 1).unsqueeze(-1) )
            obs = torch.cat((obs, agent_index), dim=-1)
        pos = ( obs[..., self.pos_start : self.pos_start + self.pos_dim] if self.pos_dim > 0 else None )
        vel = ( obs[..., self.vel_start : self.vel_start + self.vel_dim] if self.vel_dim > 0 else None )
        obs_no_pos = torch.cat( [ obs[..., : self.pos_start], obs[..., self.pos_start + self.pos_dim :], ], dim=-1 ).view(
            batch_size, self.n_agents, self.obs_shape )
        obs = obs_no_pos

        logits, state = self.agent_networks[0](obs, state)
        value = self.agent_networks[0].value_function()
        self._cur_value = value
        logits = logits.view(batch_size, self.n_agents * self.outputs_per_agent)
        
        return logits, state

    @override(ModelV2)
    def value_function(self):
        return self._cur_value

    def share_init_hetero_networks(self):
        for child in self.children():
            for agent_index, agent_model in enumerate(child.children()):
                if agent_index == 0:
                    state_dict = agent_model.state_dict()
                else:
                    agent_model.load_state_dict(state_dict)


class MyFullyConnectedNetworkInner(nn.Module):
    def __init__(
        self,
        obs_shape: Sequence[int],
        num_outputs: int,
        model_config: ModelConfigDict,
    ):
        nn.Module.__init__(self)

        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.free_log_std = model_config.get("free_log_std")

        if self.free_log_std:
            num_outputs = num_outputs // 2

        layers = []
        prev_layer_size = int(np.product(obs_shape))
        self._logits = None

        for size in hiddens[:-1]:
            layers.append(
                SlimFC( in_size=prev_layer_size, out_size=size, initializer=normc_initializer(1.0), activation_fn=activation )
            )
            prev_layer_size = size

        if no_final_linear and num_outputs:
            layers.append(
                SlimFC( in_size=prev_layer_size, out_size=num_outputs, initializer=normc_initializer(1.0), activation_fn=activation )
            )
            prev_layer_size = num_outputs
        else:
            if len(hiddens) > 0:
                layers.append(
                    SlimFC( in_size=prev_layer_size, out_size=hiddens[-1], initializer=normc_initializer(1.0), activation_fn=activation )
                )
                prev_layer_size = hiddens[-1]
            if num_outputs:
                self._logits = SlimFC( in_size=prev_layer_size, out_size=num_outputs, initializer=normc_initializer(0.01), activation_fn=None )
            else:
                self.num_outputs = ([int(np.product(obs_shape))] + hiddens[-1:])[-1]

        if self.free_log_std and self._logits:
            self._append_free_log_std = AppendBiasLayer(num_outputs)

        self._hidden_layers = nn.Sequential(*layers)

        self._value_branch_separate = None
        if not self.vf_share_layers:
            prev_vf_layer_size = int(np.product(obs_shape))
            vf_layers = []
            for size in hiddens:
                vf_layers.append(
                    SlimFC( in_size=prev_vf_layer_size, out_size=size, activation_fn=activation, initializer=normc_initializer(1.0) )
                )
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = SlimFC( in_size=prev_layer_size, out_size=1, initializer=normc_initializer(0.01), activation_fn=None )
        self._features = None
        self._last_flat_in = None

    @override(TorchModelV2)
    def forward(
        self,
        obs: torch.Tensor,
        state: List[TensorType],
    ) -> (TensorType, List[TensorType]):
        self._last_flat_in = obs
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else self._features
        if self.free_log_std:
            logits = self._append_free_log_std(logits)
        return logits, state

    def value_function(self) -> TensorType:
        if self._value_branch_separate:
            return self._value_branch( self._value_branch_separate(self._last_flat_in) ).squeeze(-1)
        else:
            return self._value_branch(self._features).squeeze(-1)
