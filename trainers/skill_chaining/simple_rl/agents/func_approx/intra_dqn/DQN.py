"""
DQN Agent for Vector Observation Learning
Example Developed By:
Michael Richardson, 2018
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code expanded and adapted from code examples provided by Udacity DRL Team, 2018.
"""

# Import Required Packages
import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
from collections import namedtuple, deque

from .dqnmodel import QNetwork
#from replay_memory import ReplayBuffer

# Determine if CPU or GPU computation should be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from simple_rl.agents.AgentClass import Agent
#from simple_rl.agents.func_approx.ddpg.model import Actor, Critic, OrnsteinUhlenbeckActionNoise
from simple_rl.agents.func_approx.intra_dqn.replay_memory import DQN_ReplayBuffer
from simple_rl.agents.func_approx.intra_dqn.hyperparameters import *
from simple_rl.agents.func_approx.intra_dqn.utils import *

"""
##################################################
Agent Class
Defines DQN Agent Methods
Agent interacts with and learns from an environment.
"""
class DQNAgents():

    """
    Initialize Agent, inclduing:
        DQN Hyperparameters
        Local and Targat State-Action Policy Networks
        Replay Memory Buffer from Replay Buffer Class (define below)
    """
    def __init__(self,name, state_size, action_size, dqn_type, replay_memory_size=1e5, batch_size=64, gamma=0.99,
    	learn_rate=1e-3, target_tau=2e-3, update_rate=4, seed=0):
        
        """
        DQN Agent Parameters
        ====== 
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            dqn_type (string): can be either 'DQN' for vanillia dqn learning (default) or 'DDQN' for double-DQN.
            replay_memory size (int): size of the replay memory buffer (typically 5e4 to 5e6)
            batch_size (int): size of the memory batch used for model updates (typically 32, 64 or 128)
            gamma (float): paramete for setting the discoun ted value of future rewards (typically .95 to .995)
            learning_rate (float): specifies the rate of model learing (typically 1e-4 to 1e-3))
            seed (int): random seed for initializing training point.
        """
        self.dqn_type = dqn_type
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = int(replay_memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_rate = learn_rate
        self.tau = target_tau
        self.update_rate = update_rate
        self.seed = random.seed(seed)
        self.name = name
        self.epsilon = 1.0
       
        # self.replay_buffer = ReplayBuffer(buffer_size=BUFFER_SIZE, name_buffer="{}_replay_buffer".format(name))
        """
        # DQN Agent Q-Network
        # For DQN training, two nerual network models are employed;
        # (a) A network that is updated every (step % update_rate == 0)
        # (b) A target network, with weights updated to equal the network at a slower (target_tau) rate.
        # The slower modulation of the target network weights operates to stablize learning.
        """
        self.network = QNetwork(state_size, action_size, seed).to(device)
        self.target_network = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)
        
        # Replay memory
        #self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
        self.replay_buffer = DQN_ReplayBuffer(action_size, self.buffer_size, self.name, self.batch_size, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    ########################################################
    # STEP() method
    #
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.replay_buffer) > self.batch_size:
                experiences = self.replay_buffer.sample(BATCH_SIZE)
                self.learn(experiences, self.gamma)


	########################################################
    def update_epsilon(self):
        if "global" in self.name.lower():
            self.epsilon = max(0., self.epsilon - GLOBAL_LINEAR_EPS_DECAY)
        else:
            self.epsilon = max(0., self.epsilon - OPTION_LINEAR_EPS_DECAY)

    # ACT() method
    def act(self, state, eps=0.1, evaluation_mode=False):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.network.eval()
        with torch.no_grad():
            action_values = self.network(state)
        self.network.train()

        # Epsilon-greedy action selection
        # if random.random() > eps or evaluation_mode:  # TODO: Eval mode fixed action
        if random.random() > eps:    
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


	########################################################
    # LEARN() method
    # Update value parameters using given batch of experience tuples.
    def learn(self, experiences, gamma, DQN=True):
        
        """
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences
        #print(states)
        # Get Q values from current observations (s, a) using model nextwork
        Qsa = self.network(states).gather(1, actions)


        if (self.dqn_type == 'DDQN'):
        # Double DQN
        #************************
            Qsa_prime_actions = self.network(next_states).detach().max(1)[1].unsqueeze(1)
            # Qsa_prime_targets = self.target_network(next_states)[Qsa_prime_actions].unsqueeze(1)
            Qsa_prime_targets_ = self.target_network(next_states)
            Qsa_prime_targets = torch.tensor([Qsa_prime_targets_[i, idx.item()] for i, idx in enumerate(Qsa_prime_actions)]).unsqueeze(1).to(device)

        else:
        # Regular (Vanilla) DQN
        #************************
            # Get max Q values for (s',a') from target model
            Qsa_prime_target_values = self.target_network(next_states).detach()
            Qsa_prime_targets = Qsa_prime_target_values.max(1)[0].unsqueeze(1)

        
        # Compute Q targets for current states 
        Qsa_targets = rewards + (gamma * Qsa_prime_targets * (1 - dones))
        
        # Compute loss (error)
        loss = F.mse_loss(Qsa, Qsa_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network, self.target_network, self.tau)


    ########################################################
    """
    Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    """
    def soft_update(self, local_model, target_model, tau):
        """
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def get_qvalues(self, states, actions):
        self.network.eval()
        with torch.no_grad():
            q_values = self.network(states).gather(1, actions)
        self.network.train()
        return q_values