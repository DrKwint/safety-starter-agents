import itertools
import os
import os.path as osp
import pathlib
import subprocess

import luigi


class TrainTask(luigi.Task):
    algo = luigi.Parameter()
    robot = luigi.Parameter()
    task = luigi.Parameter()
    constraint = luigi.Parameter()
    use_aug = luigi.BoolParameter()
    dense_coefficient = luigi.FloatParameter()
    seed = luigi.IntParameter()

    def get_exp_name(self):
        exp_name = self.algo
        exp_name += '-' + self.robot
        exp_name += '-' + self.task
        exp_name += '-' + self.constraint
        exp_name += '-' + str(self.use_aug)
        exp_name += '-' + str(self.dense_coefficient)
        exp_name += '-' + str(self.seed)
        return exp_name


    def run(self):
        cmd_str = 'python experiment.py'
        cmd_str += ' --algo ' + self.algo
        cmd_str += ' --robot ' + self.robot
        cmd_str += ' --task ' + self.task
        cmd_str += ' --constraint ' + self.constraint + '_' + self.robot + self.task
        if self.use_aug:
            cmd_str += ' --use_aug ' + str(self.use_aug)
        cmd_str += ' --dense_coeff ' + str(self.dense_coefficient)
        cmd_str += ' --seed ' + str(self.seed)
        cmd_str += ' --cpu 4'
        cmd_str += ' --exp_name ' + self.get_exp_name()
        print(cmd_str)
        output = subprocess.check_output(cmd_str.split(' '))
    
    def complete(self):
        return pathlib.Path('../tests', self.get_exp_name(), 'final.txt').exists()

def create_tasks():
    algo = ['ppo_lagrangian']
    robot = [
        'point',
        'car',
        #'doggo'
    ]
    task = [
        'goal1',
        'goal2',
        'button1',
        'button2',
        #'push1',
        #'push2'
    ]
    constraint = ['proximity']
    use_aug = [True, False]
    dense_coefficient = [0]
    seed = [0]
    arg_names = [
        'algo',
        'robot',
        'task',
        'constraint',
        'use_aug',
        'dense_coefficient',
        'seed'
    ]
    args = [
        dict(zip(arg_names, arg_vals)) for arg_vals in itertools.product(
            algo, robot, task, constraint, use_aug, dense_coefficient,
            seed)
    ]
    args = [
        d for d in args
    ]
    return [TrainTask(**a) for a in args]


if __name__ == "__main__":
    tasks = create_tasks()
    luigi.build(tasks, local_scheduler=True, workers=2)
    #luigi.build(tasks, scheduler_url="http://localhost:8082", workers=1)
