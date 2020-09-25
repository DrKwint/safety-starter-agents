import gym
import numpy as np
from gym import spaces


class Env2048Wrapper(gym.Wrapper):
    def __init__(self, env, idxs=11):
        super(Env2048Wrapper, self).__init__(env)
        self._idxs = idxs  # 2**idxs = max represented value
        old_shape = self.observation_space.shape
        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=old_shape + (self._idxs, ),
                                            dtype=np.int64)

    def process_obs(self, obs):
        idxs = np.clip(np.log2(obs).astype(np.int64), 0, self._idxs - 1)
        return np.eye(self._idxs)[idxs]

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return self.process_obs(obs), rew, done, info

    def reset(self):
        return self.process_obs(self.env.reset())
