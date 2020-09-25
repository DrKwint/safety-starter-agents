import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np
import shutil
from functools import reduce

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines.constraint import ConstraintStepMonitor, ConstraintEnv, get_constraint
from baselines import logger
from importlib import import_module

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro', 'envs'}:
        if alg == 'deepq':
            if args.augmentation is not None: args.augmentation += '_product'
            env = make_env(env_id,
                           env_type,
                           seed=seed,
                           wrapper_kwargs={
                               'frame_stack': True,
                               'clip_rewards': False
                           },
                           logger_dir=logger.get_dir())
        elif alg == 'trpo_mpi':
            if args.augmentation is not None:
                args.augmentation += '_not_implemented'
            env = make_env(env_id, env_type, seed=seed)
        else:
            if args.augmentation is not None: args.augmentation += '_concat'
            frame_stack_size = 4
            env = make_vec_env(env_id,
                               env_type,
                               nenv,
                               seed,
                               gamestate=args.gamestate,
                               reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id,
                           env_type,
                           args.num_env or 1,
                           seed,
                           reward_scale=args.reward_scale,
                           flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True)

    constraints = []
    if args.constraints is not None:
        if not args.is_hard:
            assert args.reward_shaping is not None
            assert len(args.constraints) == len(
                args.reward_shaping)  # should be parallel lists
            reward_shaping = args.reward_shaping
        else:
            reward_shaping = [0.] * len(args.constraints)
        constraints = [
            get_constraint(s)(args.is_hard, args.is_dense, r)
            for (s, r) in zip(args.constraints, reward_shaping)
        ]
        env = ConstraintStepMonitor(
            ConstraintEnv(env,
                          constraints,
                          augmentation_type=args.augmentation,
                          log_dir=logger.get_dir()), logger.get_dir())

    return env, constraints


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(
            env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k, v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    import json
    with open(osp.join(logger.get_dir(), 'args.json'), 'w') as arg_record_file:
        json.dump(args.__dict__, arg_record_file)
    env, constraints = build_env(args)
    hard_constraints = [c for c in constraints if c.is_hard]

    from baselines.deepq.deepq import ActWrapper
    model = ActWrapper.load_act(args.save_path)
    q_vals = np.zeros((int(args.num_timesteps), env.action_space.n))

    # if we have already collected trajectories
    if 'experience_dir' in args.__dict__:
        logger.log("Loading collected experiences")
        # TODO: fix by adding loading of constraint state and finding the right files
        states = np.load(osp.join(args.experience_dir, 'states'))
        if file_exists:
            constraint_states = np.load(
                osp.join(args.experience_dir, 'constraint_states'))
        else:
            constraint_states = []
    else:
        print(extra_args)
        if 'collect_states' in extra_args:
            states = np.zeros((int(args.num_timesteps), ) +
                              env.observation_space.shape)
        constraint_states = []
        episode_rewards = []

        logger.log("Running loaded model")
        obs = env.reset()

        state = model.initial_state if hasattr(model,
                                               'initial_state') else None
        dones = np.zeros((1, ))

        episode_rew = np.zeros(env.num_envs) if isinstance(
            env, VecEnv) else np.zeros(1)
        timestep = 0
        ready_to_exit = False
        while True:
            timestep += 1
            if timestep >= args.num_timesteps:
                ready_to_exit = True

            if hard_constraints:
                constraint_mask = reduce(lambda x, y: x + y, [
                    c.violating_mask(env.action_space.n)
                    for c in hard_constraints
                ])
            else:
                constraint_mask = None

            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones, hard_constraint_mask=constraint_mask)
            else:
                actions, _, _, _ = model.step(obs, hard_constraint_mask=constraint_mask)

            obs, rew, done, _ = env.step(actions)
            if 'collect_states' in extra_args:
                if type(obs) is tuple:  # with augmentation
                    states[i] = obs[0]
                    constraint_states.append(obs[1])
                else:  # without aug
                    states[i] = obs
            episode_rew += rew
            done_any = done.any() if isinstance(done, np.ndarray) else done
            if done_any:
                for i in np.nonzero(done)[0]:
                    episode_rewards.append(episode_rew[0])
                    episode_rew[i] = 0
                if ready_to_exit:
                    break
                env.reset()

        np.save(osp.join(logger.get_dir(), 'episode_rewards'), episode_rewards)
        if 'collect_states' in extra_args:
            np.save(osp.join(logger.get_dir(), 'states'), states)
        if len(constraint_states) > 0:
            np.save(osp.join(logger.get_dir(), 'constraint_states'),
                    np.array(constraint_states))
        env.close()

    # calculate q values
    if 'collect_states' in extra_args:
        for i, s in enumerate(states):
            if len(constraint_states) > 0:  # with augmentation
                q_input = [(s, constraint_states[i])]
            else:
                q_input = s
            q_vals[i] = model.q(q_input)
        np.save(osp.join(logger.get_dir(), 'q_vals'), q_vals)

    shutil.copyfile(osp.join(logger.get_dir(), 'log.txt'),
                    osp.join(logger.get_dir(), 'final_log.txt'))


if __name__ == '__main__':
    main(sys.argv)
