from environment.cookbook import Cookbook

from collections import defaultdict, namedtuple
import itertools
import logging
import numpy as np
import yaml

N_ITERS = 3000000
N_UPDATE = 500
N_BATCH = 100
IMPROVEMENT_THRESHOLD = 0.8

Task = namedtuple("Task", ["goal", "steps"])

class Trainer(object):
    def __init__(self, config):
        # load configs
        self.config = config
        self.cookbook = Cookbook(config.recipes)

        # initialize randomness
        self.random = np.random.RandomState(0)

    def train(self):
        # add DQN code here
        pass