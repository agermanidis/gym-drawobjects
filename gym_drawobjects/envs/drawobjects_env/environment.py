from io import StringIO
import sys
import time
from PIL import Image

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

from .gui import GUI
from .labels import LABELS
from . import inception

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
PAINT_BLACK = 4
PAINT_WHITE = 5

def truncate(s):
    if len(s) > 10:
        return s[:7]+'...'
    else:
        return s

class DrawObjectsEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self):
        super(DrawObjectsEnv, self).__init__()
        self._is_rendering = False

        self.width = 200
        self.height = 200
        self.label_idx = 1

        self.last_reward = 0

        self._reset()

    def action_to_human_readable(self, action):
        if action == UP:
            return "UP"

        elif action == DOWN:
            return "DOWN"

        elif action == RIGHT:
            return "RIGHT"

        elif action == LEFT:
            return "LEFT"

        elif action == PAINT_WHITE:
            return "PAINT_WHITE"

        else:
            return "PAINT_BLACK"

    @property
    def observation_space(self):
        return spaces.Box(low=0.0, high=1.0, shape=(200, 200))

    @property
    def action_space(self):
        # UP, DOWN, LEFT, RIGHT, PAINT_WHITE, PAINT_BLACK
        return spaces.Discrete(6)

    def _create_new_canvas(self):
        return Image.new('L', (self.width, self.height), 'white')

    def _init_gui(self):
        self.gui = GUI()
        self.gui.start()

    def _reset(self):
        inception.init()
        # start with an empty white canvas
        self.current_canvas = self._create_new_canvas()
        self.current_pixel_data = self.current_canvas.load()
        # start at center
        self.current_pos = (self.width // 2, self.height // 2)
        return self._observe()

    def _redraw(self):
        if not self._is_rendering:
            return
        self.gui.update(self.current_canvas, self.last_reward, truncate(LABELS[self.label_idx-1]))

    def _render(self, mode="human", close=False):
        if close: return
        if not self._is_rendering:
            self._is_rendering = True
            self._init_gui()

        self._redraw()
        return None
        
    def _apply_action(self, action):
        if action == UP:
            self.current_pos = (self.current_pos[0], max(0, self.current_pos[1]-1))

        elif action == DOWN:
            self.current_pos = (self.current_pos[0], min(self.height-1, self.current_pos[1]+1))

        elif action == RIGHT:
            self.current_pos = (min(self.width-1, self.current_pos[0]+1), self.current_pos[1])

        elif action == LEFT:
            self.current_pos = (max(0, self.current_pos[0]-1), self.current_pos[1])

        elif action == PAINT_WHITE:
            self.current_pixel_data[self.current_pos] = 255

        else:
            self.current_pixel_data[self.current_pos] = 0

    def _observe(self):
        return np.array(self.current_canvas, dtype=np.float)/255.0
    
    def _reward(self, action):
        self.last_reward = inception.get_prediction(self.current_canvas, self.label_idx)
        return self.last_reward

    def _step(self, action):
        self._apply_action(action)
        reward = self._reward(action)

        obs = self._observe()
        done = False
        info = {}

        return obs, reward, done, info
