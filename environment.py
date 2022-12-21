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
        starting_conf = [[64, 0], [-32, 0], [-16, 0], [-8, 0], [-4, 0], [-2, 0], [-1, 0], [-1, 0]],
        max_iter: int = 1e9
    ):
        """Constructs all the necessary attributes for the Santa2022Environment object.

        Args:
            image (np.array): Image of Christmas card
            max_iter (int): maximum number of iterations in one episode.

        Returns:
            None
        """
        super(Santa2022Environment, self).__init__()
        self.new_confs = []
        self.max_iter = max_iter
        self.reconfiguration_costs = []
        self.current_step = 0
        self.total_cost = 0
        self.image = image
        self.is_visited_array = np.zeros(self.image.shape[:2], dtype=np.bool_)
        self.is_visited_array[128, 128] = 1
        self.double_visit_count = 0
        self.obs_shape = [3**len(starting_conf), 3]
        self.starting_conf = starting_conf
        self.conf = starting_conf.copy()
        self.obs = [[]]*(3**len(starting_conf))
        self.new_confs = [[]]*(3**len(starting_conf))
        self.obs_matrix = self.get_observation()

        confs = get_possible_confs(self.conf)
        for new_conf in product(*confs):
            r_cost = reconfiguration_cost(self.conf, new_conf)
            self.reconfiguration_costs.append(r_cost)
        
        self.action_space = spaces.Discrete(2**8)
        
        self.observation_space = spaces.Box(
          low=-50, high=50, shape=self.obs_shape, dtype=np.float16)
        
    def get_reward(self, cost: float, new_pos: List[int]) -> float:
        """Reward calculation function

        Returns:
            reward (float): amount of reward
        """
        reward = cost + 3
        
        if self.is_visited_array[new_pos]:
            self.double_visit_count += 1
            reward -= 30
        
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
        old_conf = self.conf.copy()
        new_conf = self._take_action(action).copy()
        new_pos = cartesian_to_array(*get_position(np.asarray(new_conf)))
        self.conf = new_conf
        cost = step_cost(np.asarray(old_conf), np.asarray(new_conf), self.image)
        self.total_cost += cost
        
        is_done = np.sum(self.is_visited_array == 0) == 0
        done = is_done or (self.current_step > self.max_iter)
        
        reward = self.get_reward(cost, new_pos)
        done = done or (self.double_visit_count > (self.current_step // 7))
        obs = self.get_observation()
        self.is_visited_array[new_pos] = 1
        
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
        self.is_visited_array = np.zeros(self.image.shape[:2])
        self.is_visited_array[128, 128] = 1
        self.double_visit_count = 0
        self.conf = self.starting_conf
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
        mask_img = self.image * self.is_visited_array[:, :, np.newaxis]
        render_img = (mask_img * 255).astype(np.uint8)
        return render_img
    
    def get_observation(self):

        confs = get_possible_confs(self.conf)
        from_position = cartesian_to_array(*get_position(np.asarray(self.conf)))
        for index, (new_conf, r_cost) in enumerate(zip(product(*confs), self.reconfiguration_costs)):
            self.new_confs[index] = list(new_conf)
            to_position = cartesian_to_array(*get_position(np.asarray(new_conf)))

            c_cost = color_cost(from_position, to_position, self.image)
            self.obs[index] = [r_cost, c_cost, self.is_visited_array[to_position]]
            
        return self.obs