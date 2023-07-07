from environment import SUMOEnvironment
from agent import  QAgent, PGAgent
import numpy as np
import matplotlib.pyplot as plt

print("-------------------------------Deep Policy Gradient-------------------------------")
env=SUMOEnvironment()
agent=PGAgent()
total_rewards_1,total_waiting_1,total_queued_1=np.array(agent.learn(env,2))

print("----------------------------Deep Value-function Based----------------------------")
env=SUMOEnvironment()
agent=QAgent()
total_rewards_2,total_waiting_2,total_queued_2=np.array(agent.learn(env,2))

plt.xlabel('Training episodes')
plt.ylabel('Average Reward per Simulation')
plt.plot(total_rewards_1[:,0],total_rewards_1[:,1],color="red",label="Deep Policy Gradient")
plt.plot(total_rewards_2[:,0],total_rewards_2[:,1],color="green",label="Deep Value-function Based")
plt.legend()
plt.savefig("Reward_Figure.png")
plt.close()

plt.xlabel('Training episodes')
plt.ylabel('Average Cumulative Delay per Simulation (s)')
plt.plot(total_waiting_1[:,0],total_waiting_1[:,1],color="red",label="Deep Policy Gradient")
plt.plot(total_waiting_2[:,0],total_waiting_2[:,1],color="green",label="Deep Value-function Based")
plt.legend()
plt.savefig("Waiting_Figure.png")
plt.close()

plt.xlabel('Training episodes')
plt.ylabel('Average vehicles halting in the intersection per Simulation')
plt.plot(total_queued_1[:,0],total_queued_1[:,1],color="red",label="Deep Policy Gradient")
plt.plot(total_queued_2[:,0],total_queued_2[:,1],color="green",label="Deep Value-function Based")
plt.legend()
plt.savefig("Queue_Figure.png")
plt.close()


