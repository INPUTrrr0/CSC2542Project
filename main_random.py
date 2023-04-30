#!/usr/bin/env python

import logging
import numpy as np
import os
import sys
import traceback
import yaml
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

from utils.util import Struct
import environment


TOTAL_EPS = 1e6
EVAL_FREQ = 1e2
EVAL_EPS = 0


def main():
    config = configure("experiments/config_get_gold_ood.yaml")
    env = environment.CraftEnv(config, random_seed=101)
    # env.set_alg_name('Random')

    n_eps = 0
    avg_steps = 0
    while n_eps < TOTAL_EPS:
        obs = env.reset()
        done = False
        while not done:
            action = np.random.choice(range(env.action_space.n))
            obs, reward, done, _ = env.step(action)

        n_eps += 1
        avg_steps += (env.n_step - avg_steps) / n_eps
        print(f'Average steps: {avg_steps}')


def configure(file_name):
    # load config
    with open(file_name) as config_f:
        config = Struct(**yaml.load(config_f, Loader=yaml.SafeLoader))

    # set up experiment
    config.experiment_dir = os.path.join("experiments/%s" % config.name)
    # assert not os.path.exists(config.experiment_dir), \
    #         "Experiment %s already exists!" % config.experiment_dir
    # if not os.path.exists(config.experiment_dir):
    #     os.mkdir(config.experiment_dir)

    # # set up logging
    # log_name = os.path.join(config.experiment_dir, "run.log")
    # logging.basicConfig(filename=log_name, level=logging.DEBUG,
    #         format='%(asctime)s %(levelname)-8s %(message)s')
    # def handler(type, value, tb):
    #     logging.exception("Uncaught exception: %s", str(value))
    #     logging.exception("\n".join(traceback.format_exception(type, value, tb)))
    # sys.excepthook = handler

    # logging.info("BEGIN")
    # logging.info(str(config))
    return config

if __name__ == "__main__":
    main()
