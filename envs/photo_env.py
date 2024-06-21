import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
import os
from edit_photo import PhotoEditor


DATASET_DIR = "./dataset/"
TARGET_DIR = "expertC/"
ORIGINAL_DIR = "original/"


class PhotoEnhancementEnv(gym.Env):
    metadata = {
            'render.modes': ['human', 'rgb_array'],
        }
    def __init__(self,
                    batch_size,
                    logger=None,
                    imsize=512,
                    max_episode_steps=10,
                    training_mode=True):
            super().__init__()
            self.tags = {'max_episode_steps': max_episode_steps}
            self.logger = logger or logging.getLogger(__name__)
            self.imsize = imsize
            self.batch_size = batch_size
            self.training_mode = training_mode
            try:
                self.file_names
            except:
                self.file_names = []
                with open(os.path.join(DATASET_DIR, "trainSource_jgp.txt")) as f:
                    s = f.read()
                self.file_names.extend(s.split("\n")[:-1])
                self.file_names = \
                    list(map(lambda x: os.path.join(DATASET_DIR, ORIGINAL_DIR, x), self.file_names))
            self.photo_editor = PhotoEditor()
            self.num_parameters = self.photo_editor.num_parameters
            self.action_space = spaces.Dict({
            'parameters':
            spaces.Box(low=-1.0, high=1.0,
                            shape=(self.batch_size, self.num_parameters), dtype=np.float32),
        })
            self.observation_space = spaces.Dict({
            'image':
            spaces.Box(low=0,
                       high=255,
                       shape=(self.batch_size, self.imsize, self.imsize, 3),
                       dtype=np.uint8),
            'enhanced_image':spaces.Box(low=0,
                       high=255,
                       shape=(self.batch_size, self.imsize, self.imsize, 3),
                       dtype=np.uint8)
        }
        )
    
    def reset (self):
         