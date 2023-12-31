from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import torch as th
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

def init_layer(layer):
    nn.init.orthogonal_(layer.weight)
    return layer

class DroneNetwork(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 4,
            last_layer_dim_vf: int = 4,
        ):
            super().__init__()

            # IMPORTANT:
            # Save output dimensions, used to create the distributions
            self.latent_dim_pi = last_layer_dim_pi
            self.latent_dim_vf = last_layer_dim_vf

            # Policy network
            self.policy_net = nn.Sequential(
                init_layer(nn.Linear(feature_dim, 128)), nn.Tanh(),
                init_layer(nn.Linear(128, 128)), nn.Tanh(),
                init_layer(nn.Linear(128, last_layer_dim_pi)), nn.Tanh()
            )
             
            # Value network
            self.value_net = nn.Sequential(
                init_layer(nn.Linear(feature_dim, 128)), nn.Tanh(),
                init_layer(nn.Linear(128, 128)), nn.Tanh(),
                init_layer(nn.Linear(128, last_layer_dim_vf)), nn.Tanh()
            )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class DronePolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = True
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = DroneNetwork(self.features_dim)
