from environment import SUMOEnvironment
from agent import  QAgent, PGAgent

env=SUMOEnvironment()
agent=PGAgent()
agent.learn(env,100)
env.close()


