import gym
from pygame import init
from environment.cookbook import Cookbook
import numpy as np
from environment.craft import CraftWorld


class CraftEnv(gym.Env):
    def __init__(self, config):
        super(CraftEnv, self).__init__()
        self.config = config
        self.cookbook = Cookbook(config.recipes)
        self.world = CraftWorld(config)

        self.n_action = self.world.n_actions
        self.n_features = self.world.n_features
        self.action_space = gym.spaces.Discrete(n=self.n_action)
        self.observation_space = gym.spaces.Box(low=-np.ones(self.n_features) * np.inf,
                                                high=np.ones(self.n_features) * np.inf)
        self.spec = gym.envs.registration.EnvSpec('CraftEnv-v0')

        self.scenario = self.world.sample_scenario_with_goal(self.cookbook.index["wood"])  # goal
        self.state_before = None


    def step(self, action):
        reward, state = self.state_before.step(action)
        info = {}
        done = state.satisfies('wood', self.cookbook.index["wood"])
        state_feats = state.features()
        return state_feats, reward, done, info

    def reset(self):
        init_state = self.scenario.init()
        self.state_before = init_state
        init_state_feats = init_state.features()
        return init_state_feats

    def render(self, mode='human'):
        print("[not supporting rendering]")
        raise NotImplementedError
