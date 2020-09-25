import numpy as np
import os
import atexit

from gym.core import Wrapper


class LogBuffer(object):
    """Automatically growing (by doubling) numpy array for logging"""

    def __init__(self, initial_buffer_size, object_shape, dtype=np.uint8):
        """
        Args:
            initial_buffer_size: size of buffer log initially
            object_shape: shape of objects to be stored in log
        """
        self.buffer = np.zeros((initial_buffer_size, ) + object_shape, dtype)
        self.next_step = 0

    def log(self, item):
        try:
            self.buffer[self.next_step] = item
        except IndexError:
            self.buffer = np.concatenate(
                [self.buffer, np.zeros_like(self.buffer)])
            self.buffer[self.next_step] = item
        self.next_step += 1
        return self.next_step

    def save(self, filename):
        np.save(filename, self.buffer[:self.next_step - 1])


class ConstraintStepMonitor(Wrapper):
    def __init__(self, env, filename, initial_log_size=10000):
        Wrapper.__init__(self, env=env)
        self.filename = filename
        self.log_size = initial_log_size
        self.action_log = None
        self.reward_log = LogBuffer(initial_log_size, (), dtype=np.float32)
        self.raw_reward_log = LogBuffer(initial_log_size, (), dtype=np.float32)
        self.done_log = LogBuffer(initial_log_size, (), dtype=np.int32)
        self.reset_counter = 0
        atexit.register(self.save)

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        self.update(ob, action, rew, done, info)
        return (ob, rew, done, info)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def update(self, ob, act, rew, done, info):
        if self.action_log is None:
            self.action_log = LogBuffer(
                self.log_size, act.shape, dtype=np.int32)
        act_ns = self.action_log.log(act)
        rew_ns = self.reward_log.log(rew)
        _ = self.raw_reward_log.log(info['raw_reward'])
        don_ns = self.done_log.log(done)
        # assert that the logs are staying in step
        assert act_ns == rew_ns
        assert act_ns == don_ns

        if done:
            self.reset_counter += 1
            if self.reset_counter % 1000 == 0:
                self.save()

    def save(self):
        self.action_log.save(os.path.join(self.filename, 'action'))
        self.reward_log.save(os.path.join(self.filename, 'reward'))
        self.raw_reward_log.save(os.path.join(self.filename, 'raw_reward'))
        self.done_log.save(os.path.join(self.filename, 'done'))
