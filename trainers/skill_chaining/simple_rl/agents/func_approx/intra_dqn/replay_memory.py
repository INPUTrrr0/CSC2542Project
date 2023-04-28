"""
Replay Memory Class for DQN Agent for Vector Observation Learning
Example Developed By:
Michael Richardson, 2018
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code expanded and adapted from code examples provided by Udacity DRL Team, 2018.
"""

# Import Required Packages
import torch
import numpy as np
import random
from collections import namedtuple, deque
from .dqnmodel import QNetwork
from simple_rl.agents.func_approx.intra_dqn.hyperparameters import BUFFER_SIZE, BATCH_SIZE
# Determine if CPU or GPU computation should be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
##################################################
ReplayBuffer Class
Defines  a Replay Memeory Buffer for a DQN or DDQN agent
The buffer holds memories of: [sate, action reward, next sate, done] tuples
Random batches of replay memories are sampled for learning. 
"""
class DQN_ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self,  action_size, buffer_size, name_buffer,batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        #self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.name = name_buffer
        self.num_exp = 0
        self.buffer_size = buffer_size
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.num_exp += 1
    
    def sample(self,batch_size=BATCH_SIZE):
        #self.num = num
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        # if self.num_exp < batch_size:
        #     batch = random.sample(self.memory, self.num_exp)
        # else:
        #     batch = random.sample(self.memory, batch_size)

        #states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return (states, actions, rewards, next_states, dones)
    def size(self):
        return self.buffer_size

    def __len__(self):
        return self.num_exp

    # def __len__(self):
    #     """Return the current size of internal memory."""
    #     return len(self.memory)
    def clear(self):
        self.memory = deque(maxlen=self.buffer_size)
        self.num_exp = 0