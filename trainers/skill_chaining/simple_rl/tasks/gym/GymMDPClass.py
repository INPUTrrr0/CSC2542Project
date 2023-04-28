'''
GymMDPClass.py: Contains implementation for MDPs of the Gym Environments.
'''

# Python imports.
import random
import sys
import os
import random
# import numpy as np
# Other imports.
import gym
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.gym.GymStateClass import GymState


class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)

class GymMDP(MDP):
    ''' Class for Gym MDPs '''

    def __init__(self, env, render=False):
        '''
        Args:
            env_name (str)
        '''
        self.env_name = 'Mine'
        self.env = env
        self.render = render
        MDP.__init__(self, range(self.env.action_space.n), self._transition_func, self._reward_func, init_state=None)

    def _reward_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (float)
        '''
        obs, reward, is_terminal, info = self.env.step(action)

        if self.render:
            self.env.render()

        self.next_state = GymState(obs, is_terminal=is_terminal)

        return reward

    def _transition_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (State)
        '''
        return self.next_state

    def reset(self):
        self.init_state = GymState(self.env.reset())
        self.cur_state = self.init_state
    
    def is_goal_state(self):
        return self.env.done_

    def __str__(self):
        return "gym-" + str(self.env_name)
    
    # def execute_agent_action(self,action):
    #     if random.random() > 0.001:
    #         action_chosen = random.randint(0,4)
    #     else:
    #         action_chosen = np.argmax(action)
    #     next_state, reward, _, _ = self.env.step(action_chosen)
        
    #     return next_state, reward
    def execute_agent_action(self, action, option_idx=None):
        reward, next_state = super(GymMDP, self).execute_agent_action(action)
        return reward, next_state
    
    @staticmethod
    def is_primitive_action(self,action):
        return -1. <= action.all() <= 1.
    
