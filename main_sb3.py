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


TOTAL_EPS = int(1e4)+10
EVAL_FREQ = int(1e2)
EVAL_EPS = 30


def main():
    config = configure("experiments/config_build_plank.yaml")
    env = environment.CraftEnv(config, random_seed=101)
    eval_env = environment.CraftEnv(config, random_seed=1017, eval=True, scenario=env.scenario)  # must use different rnd seed!
    
    # trainer = DQN("MlpPolicy", env, verbose=0, exploration_initial_eps=0.2, exploration_final_eps=0.05, exploration_fraction=5e-2)
    # trainer = DQN("MlpPolicy", env, verbose=0, exploration_initial_eps=0.1, exploration_final_eps=0.1)  # no annealing
    # trainer = PPO("MlpPolicy", env, verbose=0)
    trainer = A2C("MlpPolicy", env, verbose=0)
    # print(trainer.policy)
    env.set_alg_name(trainer.__class__.__name__)
    # trainer.learn(total_timesteps=int(1e7), log_interval=100000)

    n_eps = 0
    while n_eps < TOTAL_EPS:
        # eval
        if EVAL_EPS and n_eps > 0:
            eval_steps = np.zeros(EVAL_EPS)
            for i in range(EVAL_EPS):
                obs = eval_env.reset()
                done = False
                while not done:
                    action, _ = trainer.predict(obs, deterministic=False)
                    # deterministic=True will stuck at some state
                    action = action.item()
                    obs, reward, done, _ = eval_env.step(action)
                eval_steps[i] = eval_env.n_step

            mean_steps = np.mean(eval_steps)
            std_steps = np.std(eval_steps)
            if env.writer is not None:
                env.writer.add_scalar('Avg Eval (timesteps)', mean_steps, n_eps/EVAL_FREQ)
                env.writer.add_scalar('Std Eval (timesteps)', std_steps, n_eps/EVAL_FREQ)
            print(f'Eval stage {int(n_eps/EVAL_FREQ)}, Avg Eval (timesteps): {mean_steps}, Std Eval (timesteps): {std_steps}')
            print('-------------------------------')
        n_eps += EVAL_FREQ

        callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=EVAL_FREQ, verbose=0)  # refresh it, eval every EVAL_FREQ eps
        env.set_episode(env.n_episode-1)  # offset the one more resetting
        trainer.learn(total_timesteps=int(1e7), log_interval=100000, callback=callback_max_episodes)
        # Not working if using linear annealing of epsilon; epsilon will be back and forth


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
    main()
