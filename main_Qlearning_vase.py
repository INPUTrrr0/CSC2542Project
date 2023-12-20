#!/usr/bin/env python

import logging
import numpy as np
import os
import sys
import traceback
import yaml
import warnings
import pandas as pd
from random import choice
from collections import deque

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
from torch.utils.tensorboard import SummaryWriter
from utils.util import Struct
import environment


TOTAL_EPS = int(1e4)+10
EVAL_FREQ = 10
# EVAL_FREQ = 10000
EVAL_EPS = 1
TOTAL_TIMESTEPS=int(1e7)
REPLAY=0
REPLAY_BATCHSIZE=2



def main():
    eval_data = pd.DataFrame(columns=["mean_steps", "std_steps", "mean_reward", "std_reward"])
    #config = configure("experiments/vase.yaml")
    config = configure("experiments/vase_person.yaml")
    env = environment.CraftEnv(config, random_seed=101, n_truncate=25000)
    eval_env = environment.CraftEnv(config, random_seed=1017, eval=True, n_truncate=25000, scenario=env.scenario)  # must use different rnd seed!
    env.set_alg_name("Basic_Q_Learning")
    eval_env.set_alg_name("Basic_Q_Learning")
    # action_space=env.action_space
    #trainer.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=100000, callback=[eval_callback, callback_max_episodes])
    Q_learning_agent = Q_learning(env=env, eval_env=eval_env,tabular=True,tensorboard_log='./experiments/vase_person/Q-learning',save_eval_dir="./experiments/vase_person/tabular_Q_learning_eval_steps_reward.csv") #save_eval_dir="./experiments/vase/tabular_Q_learning_eval_steps_reward.csv"
    Q_learning_agent.run()


    # epsilons = [0.25, 0.1, 0.005]
    # learning_rates = [0.25, 0.125, 0.0625]

    # rand_seeds = [1,10,15,24,42,100,300,520,900,1729] #using random seed to ensure reproducibility

    # #compute rewards for different combination of epsilon, learning rates and random seed
    # #len(rewards[0][0][0])= EPISODES (default 100)
    # rewards = []
    # for e in epsilons:
    #     epsilon_rewards = []
    #     for alpha in learning_rates:
    #         learning_rewards = []
    #         for i in rand_seeds:
    #             r = q_learning(env, alpha, e, i,EPISODES=1000)
    #             learning_rewards.append(r)
    #         epsilon_rewards.append(learning_rewards)
    #     rewards.append(epsilon_rewards)
    # rewards = np.array(rewards)



    #     callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=EVAL_FREQ, verbose=0)  # refresh it, eval every EVAL_FREQ eps
    #     # env.set_episode(env.n_episode-1)  # offset the one more resetting
    #     trainer.learn(total_timesteps=int(1e7), log_interval=100000, callback=callback_max_episodes)
    #     # Not working if using linear annealing of epsilon; epsilon will be back and forth


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

    return config



class Q_learning:
    def __init__(self, env, eval_env, random_seed=2, gamma=0.9, learning_rate=0.0125, epsilon=0.3, epsilon_min = 0.01, epsilon_decay = 0.995, tensorboard_log='./experiments/ppo', tabular=False, save_eval_dir=None):
        current_random = np.random.RandomState(seed=random_seed)
        self.random_seed = random_seed
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.env = env
        self.eval_env=eval_env
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tensorboard_log =tensorboard_log
        self.w = current_random.uniform(-0.001, 0.001, (self.env.config.world.height*self.env.config.world.width+4+env.cookbook.n_kinds, env.n_action))
        self.Q_table= np.zeros(((self.env.config.world.height-2)*(self.env.config.world.width-2)*env.cookbook.n_kinds**2, env.n_action))
        self.total_timestep=0 
        self.n_eps = 0
        self.save_eval_dir=save_eval_dir
        self.replay_buffer=deque() #state 
        self.tabular=tabular
        self.columns = ["mean_steps", "std_steps", "mean_reward", "std_reward"]
        self.save_eval_data = []


        
    def apply_weight(self, state): #dot product for linear approximation function with weights vector
        return np.dot(state, self.w)
    
    def update_weight(self, error, state, action, learning_rate):
        self.w[:, action] += learning_rate * error * state




    def encode_state(self, state):
        NUM_ROWS = 8  # Grid rows
        NUM_COLS = 8  # Grid columns
        NUM_ITEMS = 3  # Number of items
        NUM_STATES = NUM_ROWS * NUM_COLS * 2**NUM_ITEMS  # 8x8 grid and 4 combinations of items
        x, y = state.pos
        """Encode the state as a single integer."""
        item_state = state.inventory[self.env.cookbook.index["person"]] * 2 + state.inventory[self.env.cookbook.index["key"]] * 2 + state.inventory[self.env.cookbook.index["vase"]]  # Convert binary flags to an integer
        return (x-1) * NUM_COLS * 2**NUM_ITEMS + (y-1) * 2**NUM_ITEMS + int(item_state)

    def act_with_cautious(self, states, epsilon:int, action_to_avoid, with_prob, bellman=False, T=1, normalization=False): #Q_policy 
            if bellman: #if using bellman probability 
                e_x = np.exp((states - np.max(states))/T)
                action_probabilities = e_x / e_x.sum(axis=0)
                if np.random.rand() < with_prob: #choose a diff action with this prob
                    action_probabilities+=action_probabilities[action_to_avoid]/(len(action_probabilities)-1)
                    action_probabilities[action_to_avoid]=0
                
                action = np.random.choice(np.arange(len(states)), p=action_probabilities)
                if np.random.rand() < epsilon:
                    index=choice([i for i in range(self.env.n_action) if i != action_to_avoid])
                    return index, states[index]
                return action, states[action]
            if normalization:
                min_val = np.min(states)
                max_val = np.max(states)
                normalized = (states - min_val) / (states - states)
                # Adjust to exclude 0 and 1
                epsilon = 1e-10  # Small constant
                normalized = (normalized * (1 - 2 * epsilon)) + epsilon
                action = np.random.choice(np.arange(len(states)), p=normalized)
                return action, states[action]
            else:
                return states.argmax(), states[states.argmax()]


    def act(self, states, epsilon:int, bellman=False, T=1, normalization=False): #Q_policy 
        if np.random.rand() < epsilon:
            index=np.random.randint(self.env.n_action)
            return index, states[index]
        else:
            if bellman: #if using bellman probability 
                e_x = np.exp((states - np.max(states))/T)
                action_probabilities = e_x / e_x.sum(axis=0)
                action = np.random.choice(np.arange(len(states)), p=action_probabilities)
                return action, states[action]
            if normalization:
                min_val = np.min(states)
                max_val = np.max(states)
                normalized = (states - min_val) / (states - states)
                # Adjust to exclude 0 and 1
                epsilon = 1e-10  # Small constant
                normalized = (normalized * (1 - 2 * epsilon)) + epsilon
                action = np.random.choice(np.arange(len(states)), p=normalized)
                return action, states[action]
            else:
                return states.argmax(), states[states.argmax()]

    def run(self):
        while self.n_eps < TOTAL_EPS:
            if REPLAY>0 and self.n_eps>0 and self.n_eps%REPLAY==0:
                replay()

            if EVAL_FREQ and self.n_eps%EVAL_FREQ==0 and self.n_eps>0:
                eval_episodes = self.evaluate()

                eval_episodes = np.array(eval_episodes)

                mean_steps = np.mean(eval_episodes[:,0])
                std_steps = np.std(eval_episodes[:,0])
                mean_reward = np.mean(eval_episodes[:,1])
                std_reward = np.std(eval_episodes[:,1])

                reward_pertimestep = np.zeros(len(eval_episodes))
                non_zero_indices = (eval_episodes[:, 0] != 0) & (eval_episodes[:, 1] != 0)
                reward_pertimestep[non_zero_indices]=eval_episodes[non_zero_indices, 1]/eval_episodes[non_zero_indices, 0]
                mean_reward_pertimestep = np.mean(reward_pertimestep)
                std_reward_pertimestep = np.std(reward_pertimestep)

                if self.save_eval_dir is not None:
                    
                    row_data = [mean_steps, std_steps, mean_reward, std_reward]
                    self.save_eval_data.append(row_data)

                    save_eval_df = pd.DataFrame(self.save_eval_data, columns=self.columns)
                    save_eval_df.to_csv(self.save_eval_dir, index=False)


                if self.env.writer is not None:
                    self.env.writer.add_scalar('Avg Eval (timesteps)', mean_steps, self.n_eps/EVAL_FREQ)
                    self.env.writer.add_scalar('Std Eval (timesteps)', std_steps, self.n_eps/EVAL_FREQ)
                    self.env.writer.add_scalar('Avg Eval (mean reward)', mean_reward, self.n_eps/EVAL_FREQ)
                    self.env.writer.add_scalar('Avg Eval (mean reward/timesteps)', mean_reward_pertimestep, self.n_eps/EVAL_FREQ)

                print(f'Eval stage {int(self.n_eps/EVAL_FREQ)}, Avg Eval (timesteps): {mean_steps}, Avg Eval (rewards): {mean_reward}')
                print('-------------------------------')
            
            else:
                if self.epsilon>self.epsilon_min and self.n_eps%10==0:
                    self.epsilon*=self.epsilon_decay
                self.train()

            if self.total_timestep>TOTAL_TIMESTEPS:
                return 

    def train(self):
        self.n_eps+=1 
        episode_buffer=[]
        state = self.env.reset(basic=True) #start at random state for each episode
        if self.tabular:
            state_encoding=self.encode_state(state)
            action_values=self.Q_table[state_encoding]
        else:
            state_encoding = state.q_learning_state_encoding()    #discretize to bins
            action_values = self.apply_weight(state_encoding)
        steps = 0
        while True:
            steps += 1
            if self.env.world.cookbook.precaution:
                if state.lookup() is not None:
                    item=state.lookup()
                    if item=="vase":
                        action, action_value=self.act_with_cautious(action_values,self.epsilon,action_to_avoid=1,with_prob=0.5,bellman=True)
                    elif item=="person":
                        action, action_value=self.act_with_cautious(action_values,self.epsilon,action_to_avoid=1,with_prob=0.9,bellman=True)
                    if self.tabular:
                        self.Q_table[state_encoding, 1]=self.Q_table[state_encoding, 1]*0.99 #basically like adding a little bit of negative reward!
                if state.lookdown() is not None:
                    item=state.lookdown()
                    if item=="vase":
                        action, action_value=self.act_with_cautious(action_values,self.epsilon,action_to_avoid=0,with_prob=0.5,bellman=True)
                    elif item=="person":
                        action, action_value=self.act_with_cautious(action_values,self.epsilon,action_to_avoid=0,with_prob=0.9,bellman=True)
                    if self.tabular:
                        self.Q_table[state_encoding, 0]=self.Q_table[state_encoding, 1]*0.99
                if state.lookleft() is not None:
                    item=state.lookleft()
                    if item=="vase":
                        action, action_value=self.act_with_cautious(action_values,self.epsilon,action_to_avoid=2,with_prob=0.5,bellman=True)
                    elif item=="person":
                        action, action_value=self.act_with_cautious(action_values,self.epsilon,action_to_avoid=2,with_prob=0.9,bellman=True)
                    if self.tabular:
                        self.Q_table[state_encoding, 2]=self.Q_table[state_encoding, 1]*0.99
                if state.lookright() is not None:
                    item=state.lookright()
                    if item=="vase":
                        action, action_value=self.act_with_cautious(action_values,self.epsilon,action_to_avoid=3,with_prob=0.5,bellman=True)
                    elif item=="person":
                        action, action_value=self.act_with_cautious(action_values,self.epsilon,action_to_avoid=3,with_prob=0.9,bellman=True)
                    if self.tabular:
                        self.Q_table[state_encoding, 3]=self.Q_table[state_encoding, 1]*0.99
                    
            #state_value = Q.apply_weight(state_encoding)
            action, action_value = self.act(action_values, self.epsilon, bellman=True) #pick an action given the values and epsilons
            state, reward, terminated, truncated = self.env.step(action, basic=True) #next state of the chosen action # probability？？
            if self.tabular:
                new_state=self.encode_state(state)
                best_future_value = np.max(self.Q_table[new_state])
                self.Q_table[state_encoding, action] = self.Q_table[state_encoding, action] + self.learning_rate * (reward + self.gamma * best_future_value - self.Q_table[state_encoding, action])
            else:
                new_state=state.q_learning_state_encoding()
                next_value = np.max(self.apply_weight(new_state)) #value of next state based on linear approximation function
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('error') 
                        q_error = reward + self.gamma * next_value * (not terminated and not truncated) - action_value
                except Exception as e:
                    print(f"A runtime warning occurred: {e}")
                    print(f"reward: {reward}, gamma: {self.gamma}, next_value:{next_value}, action_value={action_value}, gamma*next_value-action_value={self.gamma * next_value * (not terminated and not truncated) - action_value}")
                    q_error=0
                self.update_weight(q_error, state_encoding, action, self.learning_rate) #update weight using current state

            
            episode_buffer.append([state_encoding,action,reward,terminated,truncated]) 

            if terminated or truncated: 
                #print(f"reward: {reward}, action_values: {action_values}")
                break


            if self.tabular:
                state_encoding=new_state
                action_values=self.Q_table[state_encoding]
            else:
                state_encoding = new_state #update state
                action_values = self.apply_weight(state_encoding) #update weight using new state


    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.replay_buffer, REPLAY_BATCHSIZE)
        #compare trajectories 
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def evaluate(self):
        total_rewards = 0
        eval_episodes = []
        self.n_eps+=1
        #self.n_eps += EVAL_EPS
        for episode in range(EVAL_EPS):
        #     obs = eval_env.reset()
        #     done = False
        #     while not done:
        #         self.total_timestep+=1
        #         action, _ = trainer.predict(obs, deterministic=False)
        #         # deterministic=True will stuck at some state
        #         action = action.item()
        #         obs, reward, done, _ = eval_env.step(action)
        #         eval_episodes[episode] = eval_env.n_step

        #     state = env.reset(basic=True).q_learning_state_encoding()
        #     done = False
        #     rewards_current_episode = 0

        #     for step in range(max_steps_per_episode): 
        #         # Choose action with highest Q-value for current state
        #         action = np.argmax(self.apply_weight(state)) 
        #         new_state, reward, done, info = env.step(action, basic=True)

        #         rewards_current_episode += reward

        #         if done:
        #             break

        #         state = new_state

        #     total_rewards += rewards_current_episode

        # average_reward = total_rewards / num_episodes
        # print(f"Average Reward per episode: {average_reward:.2f}")
            episode_buffer=[] #steps, reward 
            state = self.eval_env.reset(basic=True) #start at random state for each episode
            steps = 0
            accu_reward=0
            while True:
                steps += 1
                if self.tabular:
                    action, _ = self.act(self.Q_table[self.encode_state(state)],epsilon=0, bellman=True)
                    state, reward, terminated, truncated = self.eval_env.step(action, basic=True) 

                else:
                    state_encoding = state.q_learning_state_encoding()    #discretize to bins
                    action, action_value = self.act(self.apply_weight(state_encoding), epsilon=0, bellman=True)
                    #action=np.argmax(self.apply_weight(state_encoding))
                    state, reward, terminated, truncated = self.eval_env.step(action, basic=True) #next state of the chosen action # probability？？
                    
                accu_reward+=reward
                if terminated or truncated:
                    print(f"eval eps {self.n_eps}, steps:{steps}, reward:{accu_reward}")
                    eval_episodes.append([steps,accu_reward])
                    break
        return eval_episodes


# def q_learning(self,env=env, alpha=0.1, epsilon=0.5, seed=52, gamma=0.99, EPISODES = 100):
    
#     np.random.seed(seed)
#     Q = Q_linear_approx(seed)
#     rewards = []

#     for episode in range(EPISODES):
#         state, info = env.reset() #start at random state for each episode
#         state_encoding, _ = discretize_states(state) #discretize to bins
#         state_value = Q.apply_weight(state_encoding)
#         steps = 0
#         while True:
#             steps += 1
#             #state_value = Q.apply_weight(state_encoding)
#             action = Q_policy(state_value, epsilon) #pick an action given the current state value and epsilons
#             cur_value = state_value[action] #state value of the chosen action
#             state, reward, terminated, truncated, probs = env.step(action) #next state of the chosen action
#             next_state, _ = discretize_states(state)
#             next_value = np.max(Q.apply_weight(next_state)) #value of next state based on linear approximation function
#             q_error = reward + gamma * next_value * (not terminated and not truncated) - cur_value
#             Q.update_weight(q_error, state_encoding, action, alpha) #update weight using current state
#             state_encoding = next_state #update state
#             state_value = Q.apply_weight(state_encoding) #update weight using new state
#             if terminated or truncated:
#                 break
#         rewards.append(steps) #number of successfully taken steps

#     return rewards









# class DQN_Agent:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.memory = deque(maxlen=2000)
#         self.gamma = 0.95    # discount rate
#         self.epsilon = 1.0   # exploration rate
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.995
#         self.learning_rate = 0.001
#         self.model = self._build_model()

#     def _build_model(self):
#         # Neural Net for Deep-Q learning Model
#         model = Sequential()
#         model.add(Dense(24, input_dim=self.state_size, activation='relu'))
#         model.add(Dense(24, activation='relu'))
#         model.add(Dense(self.action_size, activation='linear'))
#         model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
#         return model

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def act(self, state):
#         if np.random.rand() <= self.epsilon:
#             return random.randrange(self.action_size)
#         act_values = self.model.predict(state)
#         return np.argmax(act_values[0])  # returns action

#     def replay(self, batch_size):
#         minibatch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state, done in minibatch:
#             target = reward
#             if not done:
#                 target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
#             target_f = self.model.predict(state)
#             target_f[0][action] = target
#             self.model.fit(state, target_f, epochs=1, verbose=0)
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

# Initialize environment
# env = gym.make('CartPole-v1')
# state_size = env.observation_space.shape[0]
# action_size = env.action_space.n
# agent = DQN_Agent(state_size, action_size)

# # Train the agent
# for e in range(1000):  # Number of episodes
#     state = env.reset()
#     state = np.reshape(state, [1, state_size])
    
#     for time in range(500):  # Time steps
#         action = agent.act(state)
#         next_state, reward, done, _ = env.step(action)
#         next_state = np.reshape(next_state, [1, state_size])
#         agent.remember(state, action, reward, next_state, done)
#         state = next_state
#         if done:
#             print("episode: {}/{}, score: {}".format(e, 1000, time))
#             break

#     if len(agent.memory) > batch_size:
#         agent.replay(batch_size)





if __name__ == "__main__":
    main()

