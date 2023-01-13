from functools import reduce
from typing import (
    Union,
    List,
    Dict,
    Tuple,
    Iterator,
    Type,
    Callable,
    Optional
)
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import numba as nb
import scipy.signal

import torch as th
from torch import nn
from torch.utils.data import Dataset

import gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.ppo.ppo import PPO
from IPython import display as ipythondisplay

def df_to_image(df: pd.DataFrame) -> np.array:
    """Transform configuration to positions where the image centre is (0, 0) dot.

    Args:
        df (pd.DataFrame): Pandas DataFrame of Christmas card.

    Returns:
        image (np.array): Image of Christmas card.
    """

    side = int(len(df) ** 0.5)  # assumes a square image
    image = df.set_index(["x", "y"]).to_numpy().reshape(side, side, -1)
    return image

def transform_conf_to_pos(config: Union[List[List[int]], np.array]) -> List[int]:
    """Transform configuration to positions where the image centre is (0, 0) dot.

    Args:
        config (np.array): Configuration of robot arm.
                        Example: [
                                    [64,  0],
                                    [-32, 0],
                                    [-16, 0],
                                    [-8,  0],
                                    [-4,  0],
                                    [-2,  0],
                                    [-1,  0],
                                    [-1,  0]
                                ]

    Returns:
        position (List[int]): List of x and y coordinates.
    """
    
    position = reduce(lambda p, q: (p[0] + q[0], p[1] + q[1]), config, (0, 0))
    return position

@nb.njit
def get_position(config):
    return config.sum(0)

@nb.njit 
def cartesian_to_array(x: int, y: int, center_x: int=128, center_y: int=128) -> Tuple[int]:
    """Transform cartesian coordinates to python image coordiantes
    (Example: cartesian (0, 0) is center of image but in python this is (image.shape[0] // 2, image.shape[1] // 2)).

    Args:
        x (int): x coord
        y (int): y coord
        center_x (int): Image center by x-axis
        center_y (int): Image center by x-axis

    Raises:
        ValueError: Coordinates not within given image dimensions.

    Returns:
        position (Tuple[int]): List of x and y coordinates.
    """

    i = center_x - y
    j = center_y + x
    return i, j

def reconfiguration_cost(from_config: List[List[int]], to_config: List[List[int]]) -> int:
    """Calculate reconfiguration cost.

    Args:
        from_config (List[List[int]]): Starting configuration
        to_config (List[List[int]]): Ending configuration

    Returns:
        cost (int): Reconfiguration cost.
    """

    diffs = np.abs(np.asarray(from_config) - np.asarray(to_config)).sum(axis=1)
    cost = np.sqrt(diffs.sum())
    return cost

@nb.njit
def color_cost(from_position: List[int], to_position: List[int], image: np.array, color_scale: int=3.0) -> int:
    """Calculate color cost from positions.

    Args:
        from_position (List[int]): List of x, y position
        to_position (List[int]): List of x, y position

    Returns:
        cost (int): Color cost.
    """

    cost = np.abs(image[to_position] - image[from_position]).sum() * color_scale
    return cost

def color_cost_from_conf(from_config: List[List[int]], to_config: List[List[int]], image, color_scale=3.0) -> int:
    """Calculate color cost from conf.

    Args:
        from_position (List[int]]): List of x, y position
        to_position (List[int]]): List of x, y position

    Returns:
        cost (int): Color cost.
    """

    from_position = cartesian_to_array(*get_position(from_config))
    to_position = cartesian_to_array(*get_position(to_config))
    cost = np.abs(image[to_position] - image[from_position]).sum() * color_scale
    return cost

def step_cost(from_config: List[List[int]], to_config: List[List[int]], image: np.array) -> int:
    """Calculate step cost.

    Args:
        from_config (List[List[int]]): Starting configuration
        to_config (List[List[int]]): Ending configuration
        image (np.array): Image of Christmas card

    Returns:
        cost (int): Step cost.
    """
    from_position = cartesian_to_array(*get_position(from_config))
    to_position = cartesian_to_array(*get_position(to_config))
    cost = (
        reconfiguration_cost(from_config, to_config) +
        color_cost(from_position, to_position, image)
    )
    return cost

def get_possible_possitions_mask_for_coords(x_coords: Iterator[int], y_coords: Iterator[int], num_arm_links: int=8) -> np.array:
    """Creates a mask of all possible pixels that can be visited by the robot 
        arm with num_arm_links number of links for given x and y coordinates.

    Args:
        x_coords (Iterator[int]): x_indexes
        y_coords (Iterator[int]): y indexes
        num_arm_links (int): number of robot links

    Returns:
        mask (np.array): Binary mask with True values on possible postions.
    """

    mask = np.zeros((num_arm_links*2+1, num_arm_links*2+1), dtype=np.bool_)

    for x_coord in x_coords:
        for y_coord in y_coords: 
            if abs(x_coord)+abs(y_coord)<num_arm_links+1:
                mask[x_coord+num_arm_links,y_coord+num_arm_links] = True

    return mask

def get_possible_possitions_mask_for_conf(conf: List[List[int]]) -> np.array:
    """Creates a mask of all possible pixels that can be visited by the robot 
        arm whose current configuration is conf.

    Args:
        conf (List[List[int]]): Current robot arm configuration.

    Returns:
        mask (np.array): Binary mask with True values on possible postions.
    """

    max_conf_values = [2**(i-1 if i-1 > 0 else 0) for i in range(len(conf))][::-1]
    x_min = y_min = -len(conf)
    x_max = y_max = len(conf) + 1

    for max_conf_value, conf_value in zip(max_conf_values, conf):

        if conf_value[0]==conf_value[1]:
            if conf_value[0]>0:
                x_max -= 1
                y_min += 1
            else:
                x_min += 1
                y_max -= 1

        elif conf_value[0]+conf_value[1]==0:
            if conf_value[0]>0:
                x_max -= 1
                y_max -= 1
            else:
                x_min += 1
                y_min += 1

        elif abs(conf_value[0])==max_conf_value:
            x_max -= 1
            x_min += 1

        else:
            y_max -= 1
            y_min += 1

    mask = get_possible_possitions_mask_for_coords(range(x_min, x_max), range(y_min, y_max))
    return mask

def get_possible_confs(conf: List[List[int]]) -> List[List[List[int]]]:
    """Creates a list of all possible confs that can be obtained from input conf.

    Args:
        conf (List[List[int]]): Current robot arm configuration.

    Returns:
        new_confs (List[List[List[int]]]): List of possible confs.
    """
    new_confs = []
    max_conf_values = [2**(i-1 if i-1 > 0 else 0) for i in range(len(conf))][::-1]
    for conf_entry, max_conf_value in zip(conf, max_conf_values):
        new_entry_1 = conf_entry
        if conf_entry[0] == conf_entry[1]:
            if conf_entry[0] > 0:
                new_entry_2 = [conf_entry[0]-1, conf_entry[1]]
                new_entry_3 = [conf_entry[0], conf_entry[1]-1]
            else:
                new_entry_2 = [conf_entry[0]+1, conf_entry[1]]
                new_entry_3 = [conf_entry[0], conf_entry[1]+1]

        elif conf_entry[0] + conf_entry[1] == 0:
            if conf_entry[0] > 0:
                new_entry_2 = [conf_entry[0]-1, conf_entry[1]]
                new_entry_3 = [conf_entry[0], conf_entry[1]+1]
            else:
                new_entry_2 = [conf_entry[0]+1, conf_entry[1]]
                new_entry_3 = [conf_entry[0], conf_entry[1]-1]

        elif abs(conf_entry[0]) == max_conf_value:
            new_entry_2 = [conf_entry[0], conf_entry[1]-1]
            new_entry_3 = [conf_entry[0], conf_entry[1]+1]

        elif abs(conf_entry[1]) == max_conf_value:
            new_entry_2 = [conf_entry[0]-1, conf_entry[1]]
            new_entry_3 = [conf_entry[0]+1, conf_entry[1]]

        new_confs.append([new_entry_1, new_entry_2, new_entry_3])
        
    return new_confs

def record_video(
    new_env: gym.Env,
    model: PPO,
    video_length: int=500,
    prefix: str="",
    video_folder: str="./videos/"
):
    """
    :param new_env: (gym.Env)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env = DummyVecEnv([lambda: new_env])
    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                              record_video_trigger=lambda step: step == 0, video_length=video_length,
                              name_prefix=prefix)

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = eval_env.step(action)


    # Close the video recorder
    eval_env.close()

def show_videos(video_path: str="", prefix: str=""):
    """
    Taken from https://github.com/eleurent/highway-env

    :param video_path: (str) Path to the folder containing videos
    :param prefix: (str) Filter the video, showing only the only starting with this prefix
    """
    html = []
    for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append("""<video alt="{}" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>""".format(mp4, video_b64.decode("ascii")))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


def discounted_cumulative_sums(x: np.array, discount: float) -> np.array:
    """Discounted cumulative sums of vectors for computing
    rewards-to-go and advantage estimates

    Args:
        x (np.array): Input array of rewards.
        discount (float): Discount factor

    Returns:
        advantages (np.array): Array of advantages
    """

    advantages = scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    return advantages

class FeaturesExtractorCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: Dict,
        features_dim: int = 128,
        n_input_channels: int = 3
    ):
        super(FeaturesExtractorCNN, self).__init__(observation_space, features_dim)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.rand(1, 3, 257, 257).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        
        self.conf_linear = nn.Sequential(nn.Linear(16, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        image = observations["image"].permute(0, 3, 1, 2)
        conf = observations["conf"]
        
        image_features = self.linear(self.cnn(image))
        conf_features = self.conf_linear(conf)
        
        features = th.cat((image_features, conf_features), 1)
        return features

class CustomPPONetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 3**8,
        last_layer_dim_vf: int = 1,
    ):
        super().__init__()
        
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim*2, feature_dim*4), nn.ReLU(),
            nn.Linear(feature_dim*4, feature_dim*8), nn.ReLU(),
            nn.Linear(feature_dim*8, last_layer_dim_pi)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim*2, feature_dim), nn.ReLU(),
            nn.Linear(feature_dim, feature_dim//2), nn.ReLU(),
            nn.Linear(feature_dim//2, last_layer_dim_vf)
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

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomPPONetwork(self.features_dim)

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        n_input_channels: int = 3,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64
    ):
        super(CustomNetwork, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.rand(1, 3, 257, 257).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, feature_dim), nn.ReLU())
        
        self.conf_linear = nn.Sequential(nn.Linear(16, feature_dim), nn.ReLU())
        
        

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim*2, last_layer_dim_pi)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim*2, last_layer_dim_vf)
        )

    def forward(self, images: th.Tensor, confs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        image_features = self.linear(self.cnn(images))
        conf_features = self.conf_linear(confs)
        
        features = th.cat((image_features, conf_features), 1)

        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

class SantaDataset(Dataset):
    """Santa dataset."""

    def __init__(self, observations, actions, values):

        self.observations = observations
        self.actions = actions
        self.values = values

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        observation = self.observations[idx]
        image = observation["image"]
        conf = observation["conf"]
        action = self.actions[idx]
        reward = self.values[idx]

        return image, conf, action, reward
