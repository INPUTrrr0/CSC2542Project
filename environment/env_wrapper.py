# import gymnasium as gym
#%%
from time import sleep
from sys import gettrace
import gym
import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from environment.cookbook import Cookbook
import numpy as np
from environment.craft import CraftWorld, dir_to_str

from PIL import Image, ImageDraw, ImageFont


isDebug = True if gettrace() else False


class CraftEnv(gym.Env):
    def __init__(self, config, random_seed=2, eval=False, scenario=None, n_truncate=20000,getgifflag=False, save_eval_dir=None):
        super(CraftEnv, self).__init__()
        self.alg_name = None
        self.eval = eval
        self.n_truncate = n_truncate
        self.getgifflag=getgifflag
        self.config = config
        self.cookbook = Cookbook(config.recipes)
        self.world = CraftWorld(config, random_seed)
        self.writer = None
        self.save_eval= None
        self.image_sequence = []
        self.inventory_sequence=[]
        self.save_eval_dir= save_eval_dir
        self.color_map = {
        1: (0, 0, 0),    # Black
        0: (255, 255, 255),  # White
        -1: (255, 0, 0),  # Bright Red
        2: (0, 0, 255),  # Yellow
        3: (173, 216, 230),  # Light Blue
        4: (255, 255, 0)
        }   


        self.n_action = self.world.n_actions
        self.n_features = self.world.n_features
        self.action_space = gym.spaces.Discrete(n=self.n_action)
        self.observation_space = gym.spaces.Box(low=-np.ones(self.n_features) * np.inf,
                                                high=np.ones(self.n_features) * np.inf,
                                                dtype=np.float64)
        # self.spec = gym.envs.registration.EnvSpec('CraftEnv-v0')
        #this is where i can also have the dictionary / automata for the probability of chained consequences ? 
        self.goal = self.config.world.goal
        if scenario is None:
            self.scenario = self.world.sample_scenario_with_goal(self.cookbook.index[self.goal])  # goal
        else:
            self.scenario = scenario
        self.state_before = None
        self.n_step = 0
        self.accum_reward=0
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
        if not isDebug:
            self.writer = SummaryWriter(self.config.tensorboard_dir.rstrip('/') + f'/{self.config.name}/log/{name}')

    def step(self, action, basic=False): #the basic flag denotes returning a basic state encoding rather than the state features (partially observed?)
        self.n_step += 1
        reward, state = self.state_before.step(action)

        og_map, flat_map, inv = state.get_ogmap_map_invetory()
        if self.eval:
            self.image_sequence.append(flat_map)
            self.inventory_sequence.append(inv)

        truncated = True if self.n_step >= self.n_truncate else False
        info = {'truncated': truncated}
        sat = state.satisfies(self.goal, self.cookbook.index[self.goal])

        # reward = 1 if sat else 0
        reward += 10 if sat else -0.8 / self.n_truncate
        self.accum_reward+=reward
        # reward += 3 if sat else -2.4 / self.n_truncate
        done = sat or truncated
        self.state_before = state

        state_feats = state.features()
        if isDebug:
            pass
            # print(f'Ep {self.n_episode}, step: {self.n_step}, action: {dir_to_str(action)}, reward: {reward},\nmap:\n{flat_map}')
            # print("Inventory is: " + ', '.join(f"{count} x {idx}" for count, idx in zip(inv[2:],self.cookbook.index.ordered_contents[1:])))
        if done and not self.eval:
            #print("at done but not self.eval")
            #print(self.image_sequence)
            self.n_total_step += self.n_step
            if truncated:
                print(f'training Ep {self.n_episode}: Timeout ({self.n_step} steps)!\t\tTotal steps: {self.n_total_step}.')
            else:
                self.done_ = True
                #print(f'training Ep {self.n_episode}: Goal Reached within {self.n_step} steps!\t\tTotal steps: {self.n_total_step}.')
            
            if not isDebug:
                pass
                # if self.writer is not None:
                #     self.writer.add_scalar('Time steps', self.n_step, self.n_episode)
            #else:
                #print('------------------------------------------')
               # sleep(2)
        if done and self.eval:
            if self.getgifflag:
                self.getgif(self.image_sequence, self.inventory_sequence)
            # if self.n_step < 50:
            #     self.getgif(self.image_sequence, self.inventory_sequence)
            #print(self.image_sequence)
            self.n_total_step += self.n_step

            if truncated:
                print(f'Ep {self.n_episode}: Timeout ({self.n_step} steps)!\t\tTotal steps: {self.n_total_step}.')
            else:
                self.done_ = True
                #if self.writer is not None:
                    #self.writer.add_scalar('Time steps', self.n_step, self.n_episode)
                #print(f'Ep {self.n_episode}: Goal Reached within {self.n_step} steps!\t\tTotal steps: {self.n_total_step}.')
            
            if not truncated:
                self.done_ = True
            
        if basic:
            return state, reward, sat, truncated 
            
        return state_feats, reward, done, info

    def reset(self,basic=False):
        self.n_step = 0
        self.n_episode += 1
        self.done_ = False
        init_state = self.scenario.init()
        og_map, flat_map, invetory = init_state.get_ogmap_map_invetory()
        #print("flat map is:")
        #print(flat_map)
        #print("--------------------")
        if self.eval:
            self.image_sequence=[]
            self.inventory_sequence=[]
            self.image_sequence.append(flat_map)
            self.inventory_sequence.append(invetory)

        #print(self.image_sequence)
        #print(os.getcwd()) #/Users/wind/Documents/Masters/RL/CSC2542 PROJECT

        if self.config.world.procgen_ood:
            self.sample_another_scenario()  # sample again
        
        self.state_before = init_state
        #if isDebug:
           # print(f'Map:\n{self.scenario}')
        if basic:
            return init_state

        init_state_feats = init_state.features()
        return init_state_feats


        
    def sample_another_scenario(self):
        self.scenario = self.world.sample_scenario_with_goal(self.cookbook.index[self.goal])

    def render(self, mode='human'):
        print(f'Ep {self.n_episode}, step: {self.n_step}\nstate:{self.state_before}')
        print('------------------------------------------') 
        #imgs = np.random.randint(0, 255, (100, 50, 50, 3), dtype=np.uint8)
        #imgs = [Image.fromarray(img) for img in imgs]
        print(os.getcwd())
        # duration is the number of milliseconds between frames; this is 40 frames per second
        #imgs[0].save("array.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)

    def getgif(self, image_sequence, inventory_sequence):
        images = []
        for map_array, inventory_array in zip(image_sequence, inventory_sequence):
            # Convert map_array values to corresponding colors
            height, width = map_array.shape
            border_size = 50
            map_image = Image.new('RGB', (width * 50 + 2*border_size, height * 50 + 2*border_size), color=(255, 255, 255))
            draw = ImageDraw.Draw(map_image)
            cell_size = 50  # Adjust the cell size as needed

            for y in range(height):
                for x in range(width):
                    cell_value = map_array[y, x]
                    color = self.color_map.get(cell_value, (255, 255, 255))  # Default to white if value not found
                    top_left = (x * cell_size + border_size, y * cell_size + border_size)
                    bottom_right = ((x + 1) * cell_size + border_size - 1, (y + 1) * cell_size + border_size - 1)
                    draw.rectangle([top_left, bottom_right], fill=color)

            # Create an image for inventory text
            #inventory_text="Inventory is: " + ', '.join(f"{int(count)} x {idx}" for count, idx in zip(inventory_array[2:],self.cookbook.index.ordered_contents[1:]))
            inv_images=[]
            font = ImageFont.truetype("Minecraft.ttf", 30)

            inventory_image=Image.new('RGB', (map_image.width, 60), color=(255, 255, 255))
            draw=ImageDraw.Draw(inventory_image)
            draw.text((20, 20), f"step:{self.n_step}, total reward:{round(self.accum_reward,5)}", font=font, fill="black")
            inv_images.append(inventory_image)

            inventory_image=Image.new('RGB', (map_image.width, 60), color=(255, 255, 255))
            draw=ImageDraw.Draw(inventory_image)
            draw.text((20, 20), "Inventory is:", font=font, fill="black")
            inv_images.append(inventory_image)

            for count, idx in zip(inventory_array[2:],self.cookbook.index.ordered_contents[1:]): 
                inventory_image = Image.new('RGB', (map_image.width, 60), color=(255, 255, 255))
                draw = ImageDraw.Draw(inventory_image)
                #font = ImageFont.load_default()
                inventory_text=f"{idx}: {int(count)}"
                draw.text((20, 20), inventory_text, font=font, fill="black")
                inv_images.append(inventory_image)

            # Combine map and inventory images with some whitespace in between
            combined_image = Image.new('RGB', (map_image.width, map_image.height + inventory_image.height + len(inv_images)*60+10),
                                    color=(255, 255, 255))
            combined_image.paste(map_image, (0, 0))
            height=map_image.height
            for i in inv_images:
                combined_image.paste(i, (0, height))
                height+=60

            # Append combined image to the list
            images.append(combined_image)
            # Save the list of images as a GIF file
        images[0].save(f'experiments/vase/{self.alg_name}/episode{self.n_episode}.gif', save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)

        #imgs = [Image.fromarray(np.array(img,dtype=np.uint8)) for img in self.image_sequence]
