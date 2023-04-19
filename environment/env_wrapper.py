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
    def __init__(self, config):
        super(CraftEnv, self).__init__()
        self.n_truncate = 500000
        self.config = config
        self.cookbook = Cookbook(config.recipes)
        self.world = CraftWorld(config)
        if not isDebug:
            self.writer = SummaryWriter(config.tensorboard_dir)

        self.n_action = self.world.n_actions
        self.n_features = self.world.n_features
        self.action_space = gym.spaces.Discrete(n=self.n_action)
        self.observation_space = gym.spaces.Box(low=-np.ones(self.n_features) * np.inf,
                                                high=np.ones(self.n_features) * np.inf,
                                                dtype=np.float64)
        # self.spec = gym.envs.registration.EnvSpec('CraftEnv-v0')

        self.goal = self.config.world.goal
        self.scenario = self.world.sample_scenario_with_goal(self.cookbook.index[self.goal])  # goal
        self.state_before = None
        self.n_step = 0
        self.n_episode = 0

        print(f'Indices: {self.cookbook.index.contents}')
        print(f'Grabbable indices: {self.world.grabbable_indices},\
                workshop indices: {self.world.workshop_indices}')
        print(f'Goal: {self.goal}')
        print('----------------- Start -----------------')


    def step(self, action):
        self.n_step += 1
        reward, state = self.state_before.step(action)
        truncated = True if self.n_step >= self.n_truncate else False
        info = {'truncated': truncated}
        sat = state.satisfies(self.goal, self.cookbook.index[self.goal])
        # reward = 1
        reward = 1 if sat else -0.8 / self.n_truncate
        # reward = 1 + 4 * (self.n_truncate - self.n_step) / self.n_truncate
        done = sat or truncated
        self.state_before = state

        state_feats = state.features()
        if isDebug:
            print(f'step: {self.n_step}, action: {dir_to_str(action)}, reward: {reward},\nstate:{state}')
        if done:
            if truncated:
                print(f'Ep {self.n_episode}: Timeout ({self.n_step} steps)!')
            else:
                print(f'Ep {self.n_episode}: Goal Reached within {self.n_step} steps!')
            if not isDebug:
                self.writer.add_scalar('Time steps', self.n_step, self.n_episode)
            else:
                sleep(3)
        return state_feats, reward, done, info

    def reset(self):
        self.n_step = 0
        self.n_episode += 1

        init_state = self.scenario.init()
        self.state_before = init_state
        if isDebug:
            print(f'Map:\n{self.scenario}')

        init_state_feats = init_state.features()
        return init_state_feats

    def render(self, mode='human'):
        print("[not supporting rendering]")
        raise NotImplementedError
