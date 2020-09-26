import atexit
import collections
import os

import gym
import numpy as np
from gym.spaces.box import Box
import gym.spaces as spaces
import tensorflow as tf

import constraint
from constraint.bench.step_monitor import LogBuffer
from constraint.constraint import SoftDenseConstraint


class ConstraintEnv(gym.Wrapper):
    def __init__(self,
                 env,
                 constraints,
                 augmentation_type=None,
                 log_dir=None,
                 action_history_size=10):
        gym.Wrapper.__init__(self, env)
        if augmentation_type == 'constraint_state_concat' and isinstance(
                env.observation_space, spaces.Dict):
            spaces_dict = dict(env.observation_space.spaces)
            spaces_dict['constraint_state'] = spaces.Tuple(
                [spaces.MultiBinary(c.num_states) for c in constraints])
            self.observation_space = spaces.Dict(spaces_dict)
        """
        elif augmentation_type == 'constraint_state_concat' and isinstance(
                env.observation_space, Box):
            constraint_shape_len = sum([c.num_states for c in constraints])
            new_shape = list(env.observation_space.shape)
            new_shape[-1] = new_shape[-1] + constraint_shape_len
            self.observation_space = Box(-np.inf, np.inf, tuple(new_shape),
                                         env.observation_space.dtype)
        """
        self.constraints = constraints
        self.augmentation_type = augmentation_type
        self.prev_obs = self.env.reset()
        if log_dir is not None:
            self.log_dir = log_dir
            self.save_cost_log = False
            self.cost_log = LogBuffer(1024, (), dtype=np.float32)
            self.viol_log_dict = dict([(c, LogBuffer(1024, (), dtype=np.bool))
                                       for c in constraints])
            self.state_log_dict = dict([(c, LogBuffer(1024, (),
                                                      dtype=np.int32))
                                        for c in constraints])
            self.rew_mod_log_dict = dict([
                (c, LogBuffer(1024, (), dtype=np.float32)) for c in constraints
            ])
        else:
            self.logs = None
        self.reset_counter = 0
        atexit.register(self.save)

    def augment_obs(self, ob):
        if self.augmentation_type == 'constraint_state_concat' and isinstance(
                self.env.observation_space, spaces.Dict):
            ob['constraint_state'] = tuple([
                np.eye(c.num_states)[c.current_state] for c in self.constraints
            ])
        elif self.augmentation_type == 'constraint_state_concat':
            ob = np.concatenate([ob] + np.array([
                np.eye(c.num_states)[c.current_state] for c in self.constraints
            ]))
        elif self.augmentation_type == 'constraint_state_product':
            ob = (ob, np.array([c.current_state for c in self.constraints]))
        elif self.augmentation_type == 'action_history_product':
            ob = (ob, np.array(self.action_history))
        return ob

    def save(self):
        if self.save_cost_log:
            self.cost_log.save(os.path.join(self.log_dir, 'cost'))
        [
            log.save(os.path.join(self.log_dir, c.name + '_viols'))
            for (c, log) in self.viol_log_dict.items() if c.is_hard == False
        ]
        [
            log.save(os.path.join(self.log_dir, c.name + '_state'))
            for (c, log) in self.state_log_dict.items()
        ]
        [
            log.save(os.path.join(self.log_dir, c.name + '_rew_mod'))
            for (c, log) in self.rew_mod_log_dict.items() if c.is_hard == False
        ]

    def reset(self, **kwargs):
        [c.reset() for c in self.constraints]
        self.reset_counter += 1
        if self.reset_counter % 1000 == 0:
            self.save()
        ob = self.env.reset(**kwargs)
        ob = self.augment_obs(ob)
        self.prev_obs = ob
        return ob

    def step(self, action):
        # assumes that action is pre-filtered using the hard constraints
        ob, rew, done, info = self.env.step(action)
        if 'cost' in info:
            self.save_cost_log = True
            self.cost_log.log(info['cost'])
        for c in self.constraints:
            is_vio, rew_mod = c.step(self.prev_obs, action, done)
            if isinstance(c, SoftDenseConstraint):
                info['constraint_cost'] = info['cost'] + rew_mod
            if self.viol_log_dict is not None:
                self.viol_log_dict[c].log(is_vio)
                self.state_log_dict[c].log(c.current_state)
                self.rew_mod_log_dict[c].log(rew_mod)
                self.cost_log

        ob = self.augment_obs(ob)
        self.prev_obs = ob

        return ob, rew, done, info

    def __del__(self):
        self.reset()
