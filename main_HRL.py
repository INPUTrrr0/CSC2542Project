#!/usr/bin/env python

import logging
import numpy as np
import os
import sys
import traceback
import yaml
from torch.utils.tensorboard import SummaryWriter
from trainers.option_critic import main

from utils.util import Struct
import environment


class OptionCritic:
    def __init__(self, env) -> None:
        self.args = main.parser.parse_args()
        self.env = env
    
    def learn(self):
        self.args.env = self.env.config.name
        self.args.exp = self.env.alg_name
        main.run(self.args, self.env)


def run_HRL():
    config = configure()
    env = environment.CraftEnv(config)
    
    # TODO: add more HRL entries
    trainer = OptionCritic(env)
    env.set_alg_name(trainer.__class__.__name__)
    trainer.learn()


def configure():
    # load config
    with open("experiments/config_build_plank_ood.yaml") as config_f:
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
    run_HRL()
