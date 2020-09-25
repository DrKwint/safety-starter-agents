#!/usr/bin/env python
import gym
import pathlib
from gym.wrappers import FlattenObservation
import safety_gym
import safe_rl
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork

from constraint.constraint_wrapper import ConstraintEnv
from constraint.constraints.register import get_constraint


def main(robot, task, algo, seed, exp_name, cpu, constraint, use_aug, dense_coeff):

    # Verify experiment
    robot_list = ['point', 'car', 'doggo']
    task_list = ['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']
    algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']

    algo = algo.lower()
    task = task.capitalize()
    robot = robot.capitalize()
    assert algo in algo_list, "Invalid algo"
    assert task.lower() in task_list, "Invalid task"
    assert robot.lower() in robot_list, "Invalid robot"

    # Hyperparameters
    #exp_name = algo + '_' + robot + task
    if robot=='Doggo':
        num_steps = 1e8
        steps_per_epoch = 60000
    else:
        num_steps = 1e7
        steps_per_epoch = 30000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01
    cost_lim = 25

    # Fork for parallelizing
    mpi_fork(cpu)

    # Prepare Logger
    exp_name = exp_name or (algo + '_' + robot.lower() + task.lower())
    logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir=str(pathlib.Path('../tests', exp_name)), datestamp=False)

    # Algo and Env
    algo = eval('safe_rl.'+algo)
    env_name = 'Safexp-'+robot+task+'-v0'

    def env_fn():
        env = gym.make(env_name)
        if constraint != 'none':
            if use_aug:
                augmentation_type = 'constraint_state_concat'
            else:
                augmentation_type = 'None'
            use_dense = dense_coeff > 0.
            env = ConstraintEnv(env, [get_constraint(constraint)(False, use_dense, dense_coeff)], augmentation_type=augmentation_type, log_dir='../tests/'+exp_name)
        fcenv = FlattenObservation(env)
        return fcenv

    algo(env_fn=env_fn,
         ac_kwargs=dict(
             hidden_sizes=(256, 256),
            ),
         epochs=epochs,
         steps_per_epoch=steps_per_epoch,
         save_freq=save_freq,
         target_kl=target_kl,
         cost_lim=cost_lim,
         seed=seed,
         logger_kwargs=logger_kwargs
         )
    (pathlib.Path('../data') / exp_name / 'final.txt').touch()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='Point')
    parser.add_argument('--task', type=str, default='Goal1')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--constraint', type=str)
    parser.add_argument('--use_aug', type=bool, default=False)
    parser.add_argument('--dense_coeff', type=float)
    args = parser.parse_args()
    exp_name = args.exp_name if not(args.exp_name=='') else None
    main(args.robot, args.task, args.algo, args.seed, exp_name, args.cpu, args.constraint, args.use_aug, args.dense_coeff)