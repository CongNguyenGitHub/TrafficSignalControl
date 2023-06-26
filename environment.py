import sumo_rl
import numpy as np
import gymnasium as gym
import random 
from generator import TrafficGenerator

class SUMOEnvironment():
    def __init__(self,num_seconds=500,change_frequency=20,num_cars=500,max_num_seconds_generate_cars=500):
        self.num_seconds=num_seconds
        self.change_frequency=change_frequency
        self.max_num_seconds_generate_cars=max_num_seconds_generate_cars
        self.num_cars=num_cars
        self.current_screen=None 
        self.done=False
        self.num_episodes=0
        self.env=self.create_map()
        self.action_space=self.env.action_space
    def create_map(self):
        t=TrafficGenerator(self.max_num_seconds_generate_cars,self.num_cars)
        t.generate_routefile()
        return gym.make('sumo-rl-v0',
                net_file='big-intersection.net.xml',
                route_file='routes.rou.xml',
                render_mode='rgb_array',
                num_seconds=self.num_seconds
                )
    def reset(self):
        if ((self.num_episodes+1)%self.change_frequency==0):
            self.env=self.create_map()
        self.num_episodes+=1        
        self.env.reset()
        self.current_screen=None 
        return self.get_state()
    def close(self):
        self.env.close()
    def render(self):
        return self.env.render()
    def step(self,action):
        _,reward,_,self.done,_=self.env.step(action)
        return self.get_state(),reward,self.done
    def just_starting(self):
        return self.current_screen is None 
    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen=self.get_processed_screen()
            black_screen=np.zeros_like(self.current_screen)
            return black_screen 
        else :
            s1=self.current_screen
            s2=self.get_processed_screen()
            self.current_screen=s2
            return s2-s1
    def get_processed_screen(self):
        screen=self.env.render()
        width,height,_=screen.shape
        left=(width-128)//2
        top=(height-128)//2
        right=(width+128)//2
        bottom=(height+128)//2
        crop_screen=screen[left+30:right+30,top:bottom,:]
        return np.array(crop_screen)
