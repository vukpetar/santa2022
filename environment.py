from typing import List, Dict, Union
import numpy as np

import gym
from gym import spaces
from itertools import product

from utils import *

class Santa2022Environment(gym.Env):
    """Basic Environment for Santa 2022.
    The main API methods that users of this class need to know are:

        step
        reset
        render
        close
        seed
    """
    metadata = {'render.modes': ['rgb_array']}

    def __init__(
        self,
        image: np.array,
        starting_confs = [[[64, 0], [-32, 0], [-16, 0], [-8, 0], [-4, 0], [-2, 0], [-1, 0], [-1, 0]]],
        max_iter: int = 500
    ):
        """Constructs all the necessary attributes for the Santa2022Environment object.

        Args:
            image (np.array): Image of Christmas card
            max_iter (int): maximum number of iterations in one episode.

        Returns:
            None
        """
        super(Santa2022Environment, self).__init__()
        self.max_iter = max_iter
        self.current_step = 0
        self.total_cost = 0
        self.image = image.copy()
        self.original_image = image.copy()
        self.conf_len = len(starting_confs[0])
        self.is_visited_array = np.zeros(self.image.shape[:2])

        conf_index = np.random.randint(low=0, high=len(starting_confs))
        self.starting_confs = starting_confs
        self.starting_conf = starting_confs[conf_index]
        self.conf = self.starting_conf.copy()
        self.new_confs = [[]]*(3**self.conf_len)
        self.obs_matrix = self.get_observation()
        
        self.action_space = spaces.Discrete(3**self.conf_len)
        
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=-1.0, high=1.0, shape=self.image.shape, dtype=np.float32
            ),
            'conf': spaces.Box(low=0.0, high=64.0, shape=(self.conf_len*2,), dtype=np.float32)
        })
        
    def get_reward(self, cost: float) -> float:
        """Reward calculation function

        Returns:
            reward (float): amount of reward
        """
        reward = -cost + 1

        if np.sum(self.is_visited_array == 0) == 0:
            reward += 10
        
        return reward


    def step(self, action: np.array) -> List[Union[np.array, float, bool, Dict]]:
        """Reward calculation method

        Args:
            action (int): the index of chosen action.

        Returns:
            obs (np.array): all possible links between nodes (image pixels)
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.current_step += 1
        old_conf = self.conf.copy()
        old_pos = cartesian_to_array(*get_position(np.asarray(old_conf)))
        new_conf = self._take_action(action).copy()
        new_pos = cartesian_to_array(*get_position(np.asarray(new_conf)))
        self.conf = new_conf
        if self.image[new_pos[0], new_pos[1], 0] == -1 and self.image[old_pos[0], old_pos[1], 0] == -1:
            cost = 10
        else:
            cost = step_cost(np.asarray(old_conf), np.asarray(new_conf), self.image)
        self.total_cost += cost
        
        reward = self.get_reward(cost)
        if self.is_visited_array[new_pos[0], new_pos[1]] == 1:
            self.image[new_pos[0], new_pos[1], :] = -1.0
        else:
            self.is_visited_array[new_pos[0], new_pos[1]] = 1
        obs = self.get_observation()
        
        done = self.current_step > self.max_iter

        info = {}
        return obs, reward, done, info
    
    def _take_action(self, action: np.array):
        """Change graph weights by action values

        Args:
            action (np.array): Array of actions that represent the values for which 
                            the weights of the elements in the electric circuit should be changed

        Returns:
            None
        """
        return self.new_confs[action]
        
    def reset(self):
        """Resets environment on initial state.

        Returns:
            obs_matrix (np.array): the array that contains the feauters of all possible links between nodes (image pixels)
                                for current node
        """
        
        self.current_step = 0
        self.total_cost = 0
        if np.sum(self.is_visited_array == 0) == 0:
            self.is_visited_array = np.zeros(self.image.shape[:2])
            self.image = self.original_image.copy()
            conf_index = np.random.randint(low=0, high=len(self.starting_confs))
            self.conf = self.starting_confs[conf_index].copy()

        self.obs_matrix = self.get_observation()
        
        return self.obs_matrix

    def render(self, mode='rgb_array', close=False):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Args:
            mode (str): the mode to render with

        Returns:
            None
        """
        mask_img = self.original_image * self.is_visited_array[:, :, np.newaxis]
        render_img = (mask_img * 255).astype(np.uint8)
        return render_img
    
    def get_observation(self):

        confs = get_possible_confs(self.conf)
        conf_obs = np.array(self.conf).reshape(-1) / 64

        for index, new_conf in enumerate(product(*confs)):
            self.new_confs[index] = list(new_conf)
            
        return {
            "image": self.image,
            "conf": conf_obs
        }