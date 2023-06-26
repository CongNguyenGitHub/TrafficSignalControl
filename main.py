import gymnasium as gym
import sumo_rl
from environment import SUMOEnvironment
from generator import TrafficGenerator
from agent import  QAgent, PGAgent

env=SUMOEnvironment()
agent=QAgent()
agent.learn(env,100)



