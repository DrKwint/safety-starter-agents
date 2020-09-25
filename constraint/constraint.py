from collections import Counter

import numpy as np
import tensorflow as tf
from copy import copy

from constraint.dfa import DFA


class Constraint(object):
    def __init__(self,
                 name,
                 dfa_string,
                 is_hard,
                 violation_reward=None,
                 translation_fn=lambda x: x,
                 inv_translation_fn=None):
        self.name = name
        self.dfa = DFA.from_string(dfa_string)
        if is_hard:
            assert inv_translation_fn is not None
        else:
            assert violation_reward is not None
        self.violation_reward = violation_reward
        self.translation_fn = translation_fn
        self.is_hard = is_hard
        self.inv_translation_fn = inv_translation_fn

    def step(self, obs, action, done):
        token = self.translation_fn(obs, action, done)
        is_viol = self.dfa.step(token)
        if is_viol and self.is_hard:
            raise Exception('Hard violation')
        rew_mod = self.violation_reward if is_viol else 0.
        return is_viol, rew_mod

    def reset(self):
        self.dfa.reset()

    @property
    def current_state(self):
        return self.dfa.current_state

    @property
    def num_states(self):
        return len(self.dfa.states)

    def is_violating(self, obs, action, done):
        return self.dfa.step(self.translation_fn(obs, action, done),
                             hypothetical=True)

    def violating_mask(self, num_actions):
        mask = np.zeros(num_actions)
        for v in self.dfa.violating_inputs:
            for i in self.inv_translation_fn(v):
                mask += np.eye(num_actions)[i]
        return mask


class SoftDenseConstraint(Constraint):
    def __init__(self,
                 name,
                 dfa_string,
                 violation_reward,
                 translation_fn,
                 gamma,
                 alpha=0.025,
                 target_time=40):
        super(SoftDenseConstraint,
              self).__init__(name, dfa_string, False, violation_reward,
                             translation_fn)
        self.alpha = alpha
        self.gamma = gamma
        self.target_time = target_time
        # counters for tracking value of each DFA state
        self.prev_state = self.current_state

        self.current_step = 0
        self.state_buffer = list()
        self.expected_hitting_time = np.ones(self.num_states) * target_time
        for accept_state in self.dfa.accepting_states:
            self.expected_hitting_time[accept_state] = 0.
        self.empirical_hitting_times = []

    def step(self, obs, action, done):
        is_viol, _ = super().step(obs, action, done)

        # record state
        self.state_buffer.append(self.current_state)

        # update reward
        current_cost_val = (1 / 2)**(
            self.expected_hitting_time[self.current_state] / self.target_time)
        prev_cost_val = (1 / 2)**(self.expected_hitting_time[self.prev_state] /
                                  self.target_time)
        if self.prev_state in self.dfa.accepting_states: prev_cost_val = 0.
        rew_mod = (self.gamma * current_cost_val -
                   prev_cost_val) * self.violation_reward

        # update hitting times
        if is_viol:
            self.state_buffer = np.array(self.state_buffer)
            ep_expected_hitting_time = np.zeros(self.num_states)
            for i in range(self.num_states):
                ep_expected_hitting_time[i] = len(self.state_buffer) - np.mean(
                    np.argwhere(self.state_buffer == i)) - 1
            self.empirical_hitting_times.append(ep_expected_hitting_time)
            self.state_buffer = list()

        self.prev_state = self.current_state

        if done:
            if len(self.empirical_hitting_times) == 0:
                self.state_buffer = list()
                self.empirical_hitting_times = list()
                return is_viol, rew_mod
            if len(self.empirical_hitting_times) == 1:
                self.empirical_hitting_times = np.array(
                    self.empirical_hitting_times)
            else:
                self.empirical_hitting_times = np.stack(
                    self.empirical_hitting_times)

            ep_hitting_times = [
                np.mean([
                    v for v in self.empirical_hitting_times[:, i]
                    if not np.isnan(v)
                ]) for i in range(self.num_states)
            ]
            for i in reversed(range(len(self.expected_hitting_time))):
                if not np.isnan(ep_hitting_times[i]):
                    self.expected_hitting_time[i] = (
                        self.alpha * ep_hitting_times[i]) + (
                            (1 - self.alpha) * self.expected_hitting_time[i])
                if i != len(self.expected_hitting_time):
                    self.expected_hitting_time[i] = max(
                        self.expected_hitting_time[i],
                        max(self.expected_hitting_time[i:]))

            self.state_buffer = list()
            self.empirical_hitting_times = list()

        return is_viol, rew_mod

    def reset(self):
        self.dfa.reset()
        self.prev_state = self.current_state


def float_counter_mul(f, c):
    for (s, v) in dict(c).items():
        c[s] = f * v
    return c
