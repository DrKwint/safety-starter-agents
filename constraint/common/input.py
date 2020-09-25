import tensorflow as tf
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box


def constraint_state_placeholder(constraint, batch_size, name='ContrSt'):
    return tf.placeholder(shape=[batch_size], dtype=tf.int32, name=name)


def constraint_state_input(constraint, batch_size=None, name='ContrSt'):
    placeholder = constraint_state_placeholder(constraint, batch_size, name)
    return placeholder, tf.to_float(
        tf.one_hot(placeholder, constraint.num_states))


def action_history_placeholder(action_space, batch_size, name='ActHist'):
    if isinstance(action_space, Discrete):
        return tf.placeholder(shape=[batch_size, None], dtype=tf.int32)
    elif isinstance(action_space, Box):
        return tf.placeholder(shape=[batch_size, None] + action_space.shape,
                              dtype=action_space.dtype)
    else:
        raise Exception()


def action_history_input(action_space,
                         history_len,
                         batch_size=None,
                         name='ActHist'):
    placeholder = action_history_placeholder(action_space, batch_size, name)
    if isinstance(action_space, Discrete):
        return placeholder, tf.reshape(
            tf.to_float(tf.one_hot(placeholder, action_space.n)),
            [-1, history_len * action_space.n])
    elif isinstance(action_space, Box):
        return placeholder, tf.reshape(tf.to_float(placeholder),
                                       [batch_size, -1])
    else:
        raise Exception()
