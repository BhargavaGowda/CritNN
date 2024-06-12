import numpy as np
from gymnasium.envs.mujoco.inverted_double_pendulum_v4 import InvertedDoublePendulumEnv

class myPendulum(InvertedDoublePendulumEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.render_mode = "human"


    
    