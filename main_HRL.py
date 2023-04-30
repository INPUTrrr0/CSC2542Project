#!/usr/bin/env python

import logging
import numpy as np
import os
import sys
import traceback
import yaml
import copy
from torch.utils.tensorboard import SummaryWriter
from trainers.option_critic import run_oc
from trainers.feudalnets_MLP import run_fun
from trainers.skill_chaining import SkillChainingAgentClass

from utils.util import Struct
import environment


class OptionCritic:
    def __init__(self, env, eval_env) -> None:
        self.args = run_oc.parser.parse_args()
        self.env = env
        self.eval_env = eval_env
    
    def learn(self):
        args = copy.deepcopy(self.args)
        args.env = self.env.config.name
        args.exp = self.env.alg_name
        run_oc.run(args, self.env, self.eval_env)
    
    def eval(self, oc, eval_episodes=30):
        args = copy.deepcopy(self.args)
        args.env = self.env.config.name
        args.exp = self.env.alg_name
        eval_episode_steps = run_oc.eval(args, self.eval_env, oc, eval_episodes)
        print(eval_episode_steps)


class DSC:
    def __init__(self, env) -> None:
        self.args = SkillChainingAgentClass.parser.parse_args()
        self.env = env
    
    def learn(self):
        SkillChainingAgentClass.skill(self.args, self.env)


class FuN:
    def __init__(self, env) -> None:
        self.args = run_fun.parser.parse_args()
        self.env = env
    
    def learn(self):
        run_fun.main(self.args, self.env)


def run_HRL():
    config = configure("experiments/config_build_bridge.yaml")
    env = environment.CraftEnv(config, random_seed=101)
    eval_env = environment.CraftEnv(config, random_seed=1017, eval=True, scenario=env.scenario)  # must use different rnd seed!
    
    # TODO: add more HRL entries
    trainer = OptionCritic(env, eval_env)
    # trainer = DSC(env)
    # trainer = FuN(env)
    env.set_alg_name(trainer.__class__.__name__)
    trainer.learn()
    # trainer.eval()


def configure(file_name):
    # load config
    with open(file_name) as config_f:
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
