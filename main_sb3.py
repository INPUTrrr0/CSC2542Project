#!/usr/bin/env python

import logging
import numpy as np
import os
import sys
import traceback
import yaml
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import DQN, PPO, A2C

from utils.util import Struct
import models
import trainers
import environment


def main():
    config = configure()
    env = environment.CraftEnv(config)
    
    # model = DQN("MlpPolicy", env, verbose=3)
    # model = PPO("MlpPolicy", env, verbose=3)
    model = A2C("MlpPolicy", env, verbose=3)
    model.learn(total_timesteps=50000000, log_interval=100000)

def configure():
    # load config
    with open("config.yaml") as config_f:
        config = Struct(**yaml.load(config_f, Loader=yaml.SafeLoader))

    # set up experiment
    config.experiment_dir = os.path.join("experiments/%s" % config.name)
    # assert not os.path.exists(config.experiment_dir), \
    #         "Experiment %s already exists!" % config.experiment_dir
    if not os.path.exists(config.experiment_dir):
        os.mkdir(config.experiment_dir)

    # set up logging
    log_name = os.path.join(config.experiment_dir, "run.log")
    logging.basicConfig(filename=log_name, level=logging.DEBUG,
            format='%(asctime)s %(levelname)-8s %(message)s')
    def handler(type, value, tb):
        logging.exception("Uncaught exception: %s", str(value))
        logging.exception("\n".join(traceback.format_exception(type, value, tb)))
    sys.excepthook = handler

    logging.info("BEGIN")
    logging.info(str(config))
    return config

if __name__ == "__main__":
    main()
