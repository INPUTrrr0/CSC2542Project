# import gymnasium as gym
from time import sleep
import gym
from pygame import init
from environment.cookbook import Cookbook
import numpy as np
from environment.craft import CraftWorld, dir_to_str


class CraftEnv(gym.Env):
    def __init__(self, config):
        super(CraftEnv, self).__init__()
        self.n_truncate = 1000000
        self.config = config
        self.cookbook = Cookbook(config.recipes)
        self.world = CraftWorld(config)

        self.n_action = self.world.n_actions
        self.n_features = self.world.n_features
        self.action_space = gym.spaces.Discrete(n=self.n_action)
        self.observation_space = gym.spaces.Box(low=-np.ones(self.n_features) * np.inf,
                                                high=np.ones(self.n_features) * np.inf,
                                                dtype=np.float64)
        # self.spec = gym.envs.registration.EnvSpec('CraftEnv-v0')

        self.scenario = self.world.sample_scenario_with_goal(self.cookbook.index["wood"])  # goal
        self.state_before = None
        self.n_step = 0

        print(f'Indices: {self.cookbook.index.contents}')
        print(f'Grabbable indices: {self.world.grabbable_indices},\
                workshop indices: {self.world.workshop_indices}')
        print('----------------- Start -----------------')


    def step(self, action):
        self.n_step += 1
        reward, state = self.state_before.step(action)
        truncated = True if self.n_step > self.n_truncate else False
        info = {'truncated': truncated}
        done = state.satisfies('wood', self.cookbook.index["wood"]) or truncated
        self.state_before = state

        state_feats = state.features()
        print(f'step: {self.n_step}, action: {dir_to_str(action)}, reward: {reward},\nstate:{state}')
        if done:
            print(f'Goal Reached within {self.n_step} steps!')
            sleep(3)
        return state_feats, reward, done, info

    def reset(self):
        self.n_step = 0
        init_state = self.scenario.init()
        print(f'Map:\n{self.scenario}')
        self.state_before = init_state
        init_state_feats = init_state.features()
        return init_state_feats

    def render(self, mode='human'):
        print("[not supporting rendering]")
        raise NotImplementedError
