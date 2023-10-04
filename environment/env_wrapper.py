# import gymnasium as gym
from time import sleep
from sys import gettrace
import gym
from torch.utils.tensorboard import SummaryWriter
from environment.cookbook import Cookbook
import numpy as np
from environment.craft import CraftWorld, dir_to_str


isDebug = True if gettrace() else False


class CraftEnv(gym.Env):
    def __init__(self, config, random_seed=2, eval=False, scenario=None):
        super(CraftEnv, self).__init__()
        self.alg_name = None
        self.eval = eval
        self.n_truncate = 20000
        self.config = config
        self.cookbook = Cookbook(config.recipes)
        self.world = CraftWorld(config, random_seed)
        self.writer = None

        self.n_action = self.world.n_actions
        self.n_features = self.world.n_features
        self.action_space = gym.spaces.Discrete(n=self.n_action)
        self.observation_space = gym.spaces.Box(low=-np.ones(self.n_features) * np.inf,
                                                high=np.ones(self.n_features) * np.inf,
                                                dtype=np.float64)
        # self.spec = gym.envs.registration.EnvSpec('CraftEnv-v0')

        self.goal = self.config.world.goal
        if scenario is None:
            self.scenario = self.world.sample_scenario_with_goal(self.cookbook.index[self.goal])  # goal
        else:
            self.scenario = scenario
        self.state_before = None
        self.n_step = 0
        self.n_episode = 0
        self.n_total_step = 0
        self.done_ = False

        if not self.eval:
            print(f'Indices: {self.cookbook.index.contents}')
            print(f'Grabbable indices: {self.world.grabbable_indices},\
                    workshop indices: {self.world.workshop_indices}')
            print(f'Goal: {self.goal}')
            print('----------------- Start -----------------')

    def set_episode(self, episode):
        self.n_episode = episode
    
    def set_alg_name(self, name):
        self.alg_name = name
        # if not isDebug:
        #     self.writer = SummaryWriter(self.config.tensorboard_dir.rstrip('/') + f'/{self.config.name}/log/{name}')

    def step(self, action):
        self.n_step += 1
        reward, state = self.state_before.step(action)
        truncated = True if self.n_step >= self.n_truncate else False
        info = {'truncated': truncated}
        sat = state.satisfies(self.goal, self.cookbook.index[self.goal])

        # reward = 1 if sat else 0
        reward += 1 if sat else -0.8 / self.n_truncate
        # reward += 3 if sat else -2.4 / self.n_truncate
        done = sat or truncated
        self.state_before = state

        state_feats = state.features()
        if isDebug:
            print(f'Ep {self.n_episode}, step: {self.n_step}, action: {dir_to_str(action)}, reward: {reward},\nstate:{state}')
        if done and not self.eval:
            self.n_total_step += self.n_step
            if truncated:
                print(f'Ep {self.n_episode}: Timeout ({self.n_step} steps)!\t\tTotal steps: {self.n_total_step}.')
            else:
                self.done_ = True
                print(f'Ep {self.n_episode}: Goal Reached within {self.n_step} steps!\t\tTotal steps: {self.n_total_step}.')
            if not isDebug:
                pass
                # if self.writer is not None:
                #     self.writer.add_scalar('Time steps', self.n_step, self.n_episode)
            else:
                print('------------------------------------------')
                sleep(2)
        if done and self.eval:
            self.n_total_step += self.n_step
            if not truncated:
                self.done_ = True
        return state_feats, reward, done, info

    def reset(self):
        self.n_step = 0
        self.n_episode += 1
        self.done_ = False

        if self.config.world.procgen_ood:
            self.sample_another_scenario()  # sample again
        init_state = self.scenario.init()
        self.state_before = init_state
        if isDebug:
            print(f'Map:\n{self.scenario}')

        init_state_feats = init_state.features()
        return init_state_feats
    
    def sample_another_scenario(self):
        self.scenario = self.world.sample_scenario_with_goal(self.cookbook.index[self.goal])

    def render(self, mode='human'):
        print(f'Ep {self.n_episode}, step: {self.n_step}\nstate:{self.state_before}')
        print('------------------------------------------')
