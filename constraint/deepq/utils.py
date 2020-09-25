from baselines.common.tf_util import adjust_shape
from baselines.constraint.common.input import constraint_state_input, action_history_input
from baselines.deepq.utils import ObservationInput
import tensorflow as tf
import numpy as np


class ActionHistoryAugmentedInput(ObservationInput):
    def __init__(self, observation_space, action_space, history_len,
                 name=None):
        super().__init__(observation_space)
        self.action_space = action_space
        self.history_len = history_len
        self.action_history_placeholder = action_history_input(
            action_space, history_len)[1]

    def get(self):
        return [super().get(), self.action_history_placeholder]

    def make_feed_dict(self, data):
        """
        Assumes data is an interable whose second entry is an iterable of
        action history
        """
        obs = np.array(list(data[:, 0]))
        # list of lists with shape [batch_size, history_len]
        action_histories = list(data[:, 1])
        batch_size = obs.shape[0]

        feed_dict = super().make_feed_dict(obs)

        one_hot_histories = np.zeros(
            [batch_size, self.history_len, self.action_space.n])
        for i in range(batch_size):
            for j in range(self.history_len):
                one_hot_histories[i, j, action_histories[i][j]] = 1

        feed_dict[self.action_history_placeholder] = np.reshape(
            one_hot_histories, [batch_size, -1])
        return feed_dict

    def batch_size(self):
        return tf.shape(self.get()[0])[0]


class ConstraintStateAugmentedInput(ObservationInput):
    def __init__(self, observation_space, constraints, name=None):
        super().__init__(observation_space)
        self.constraints = constraints
        self.constraint_state_phs = [
            constraint_state_input(c, name=c.name)[1] for c in constraints
        ]

    def get(self):
        return [
            super().get(),
        ] + self.constraint_state_phs

    def make_feed_dict(self, data):
        """
        Assumes data is an interable whose second entry is an iterable of
        constraint states.
        """
        if type(data) is not np.ndarray:
            data = np.asarray(data)
        obs = np.array(list(data[:, 0]))
        # list of lists with shape [batch_size, # constraints]
        constraint_states = data[:, 1]
        batch_size = obs.shape[0]

        feed_dict = super().make_feed_dict(obs)
        c_one_hots = [
            np.zeros([batch_size, c.num_states]) for c in self.constraints
        ]
        for i, ph in enumerate(self.constraint_state_phs):
            for n in range(len(constraint_states)):
                c_one_hots[i][n, constraint_states[n][i]] = 1
            feed_dict[ph] = c_one_hots[i]

        return feed_dict

    def batch_size(self):
        return tf.shape(self.get()[0])[0]
