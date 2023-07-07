import numpy as np
from network import PGNetwork, VNetwork, QNetwork
import torch.optim 
import torch
from collections import deque
from environment import SUMOEnvironment
import random
class Agent():
    def __init__(self,alpha,gamma,num_actions):
        self.alpha=alpha
        self.gamma=gamma
        self.num_actions=num_actions
    def select_action(self,state):
        pass
    def learn(self,max_num_episodes):
        pass
class PGAgent(Agent):
    def __init__(self,alpha=1e-5,gamma=0.99,num_actions=4):
        super().__init__(alpha,gamma,num_actions)
        self.policy_net=PGNetwork(num_actions)
        self.value_net=VNetwork()
        
    def select_action(self,state):#Choose action base on current policy
        state=torch.tensor(state,dtype=torch.float32)
        action_probabilities=self.policy_net(state.unsqueeze(0).permute(0,3,1,2))
        distribution=torch.distributions.Categorical(action_probabilities)
        action=distribution.sample()[0].detach().item() 
        return action 
    
    def learn(self,env,max_num_episodes):
        maxlen=5
        #store all information in training
        total_rewards=[]
        total_waiting=[]
        total_queued=[]
        f=open("PG_Output.txt","w") 
        policy_optimizer=torch.optim.Adam(self.policy_net.parameters(),lr=self.alpha)
        value_optimizer=torch.optim.Adam(self.value_net.parameters(),lr=self.alpha)
        
        for episode in range(max_num_episodes):
            state=env.reset()
            done=False

            reward_per_step=[]
            waiting_per_step=[]
            queued_per_step=[]
            
            states=[]
            actions=[]
            rewards=[]
            while not done :
                action=self.select_action(state)
                next_state,reward,done=env.step(action) 
                #Store experience of current episode
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                #Update current state
                state=next_state
                
                #Get information current state
                reward_per_step.append(reward)
                waiting_per_step.append(env.get_total_accumulated_waiting_time())
                queued_per_step.append(env.get_total_queued())
            #Using formulation R(t)=r+gamma*r(t+1)
            returns=[]
            discounted_reward=0
            for reward in reversed(rewards):
                discounted_reward=reward+self.gamma*discounted_reward
                returns.append(discounted_reward)
            returns.reverse()
            
            #Convert to numpy
            returns=np.array(returns)
            states=np.array(states)
            actions=np.array(actions)
            rewards=np.array(rewards)
            
            #Convert to tensor
            returns=torch.tensor(returns,dtype=torch.float32).unsqueeze(-1)
            states=torch.tensor(states,dtype=torch.float32).permute(0,3,1,2)
            actions=torch.tensor(actions,dtype=torch.int64).unsqueeze(-1)
            rewards=torch.tensor(rewards,dtype=torch.float32).unsqueeze(-1)
            
            #Estimate state value
            values=self.value_net(states)
            
            #Probility of action a follow current policy in state s
            actions_prob=torch.gather(input=self.policy_net(states),dim=1,index=actions)
            #Calculate the policy loss
            policy_loss=(-torch.log(actions_prob)*(returns-values)).sum()
            
            #Minimize the policy loss
            policy_optimizer.zero_grad()   
            policy_loss.backward(retain_graph=True)
            policy_optimizer.step()
            
            #Calculate the value loss 
            value_loss=torch.nn.functional.mse_loss(returns,values)
            
            #Minimize the value loss
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
            
            #Caculate average
            average_reward=np.mean(reward_per_step)
            average_waiting=np.mean(waiting_per_step)
            average_queued=np.mean(queued_per_step)
            
            #Insert it
            total_rewards.append((episode,average_reward))
            total_waiting.append((episode,average_waiting))
            total_queued.append((episode,average_queued))

            f.write(f'Episode {episode+1} || Average reward: {average_reward} || Average waiting: {average_waiting} || Average queued: {average_queued}\n')
        f.close()
        return total_rewards,total_waiting,total_queued
class QAgent(Agent):
    def __init__(self,alpha=1e-5,gamma=0.99,num_actions=4,max_epsilon=1,min_epsilon=0.01,epsilon_decay=1e-4,batch_size=16,target_update_frequency=10):          
        super().__init__(alpha,gamma,num_actions)
        self.q_net=QNetwork(num_actions)
        self.target_net=QNetwork(num_actions)
        self.memory=deque(maxlen=20000)
        self.max_epsilon=max_epsilon
        self.min_epsilon=min_epsilon
        #decay epsilon over steps
        self.epsilon_decay=epsilon_decay
        self.target_update_frequency=target_update_frequency
        self.batch_size=batch_size
    
    def select_action(self,state):#select best action base on q_value(state)
        state=torch.tensor(state,dtype=torch.float32).unsqueeze(0).permute(0,3,1,2)
        best_action=torch.argmax(self.q_net(state),dim=1)[0].detach().item()
        return best_action

    def learn(self,env,max_num_episodes):
        total_rewards=[]
        total_waiting=[]
        total_queued=[]
        f=open("QL_Output.txt","w")

        self.target_net.load_state_dict(self.q_net.state_dict())
        optimizer=torch.optim.Adam(self.q_net.parameters(),lr=self.alpha)
        for episode in range(max_num_episodes):
            state=env.reset()
            
            reward_per_step=[]
            waiting_per_step=[]
            queued_per_step=[]

            done=False
            while not done:
                #Choose action base on epsilon greedy
                random_number=np.random.uniform(0,1)
                if random_number<=self.max_epsilon:
                    action=env.action_space.sample()
                else:
                    action=self.select_action(state)

                next_state,reward,done=env.step(action)
                #Store experience into memory
                experience=(state,action,reward,done,next_state)
                self.memory.append(experience)
                #Update state to next state
                state=next_state

                reward_per_step.append(reward)
                waiting_per_step.append(env.get_total_accumulated_waiting_time())
                queued_per_step.append(env.get_total_queued())


                if len(self.memory)>self.batch_size:
                    #Sample batch from experience
                    experiences=random.sample(self.memory,self.batch_size)
                    states=np.array([ex[0] for ex in experiences])
                    actions=np.array([ex[1] for ex in experiences])
                    rewards=np.array([ex[2] for ex in experiences])
                    dones=np.array([ex[3] for ex in experiences])
                    next_states=np.array([ex[4] for ex in experiences])
                    
                    #Convert data to tensor
                    states=torch.tensor(states,dtype=torch.float32).permute(0,3,1,2)
                    actions=torch.tensor(actions,dtype=torch.int64).unsqueeze(-1)
                    rewards=torch.tensor(rewards,dtype=torch.float32).unsqueeze(-1)
                    dones=torch.tensor(dones,dtype=torch.float32).unsqueeze(-1)
                    next_states=torch.tensor(next_states,dtype=torch.float32).permute(0,3,1,2)
                    
                    #Caculate max q_value(s')
                    target_q_values=self.target_net(next_states)
                    max_target_q_values=target_q_values.max(dim=1,keepdim=True)[0]
                    #Using formulation q_value(s,a)=r+gamma*max q_value(s')
                    targets=rewards+self.gamma*(1-dones)*max_target_q_values
                    
                    #Predict q_value(s,a)
                    q_values=self.q_net(states)
                
                    action_q_values=torch.gather(input=q_values,dim=1,index=actions)
                    #Caculate the loss
                    loss=torch.nn.functional.mse_loss(action_q_values,targets)
                    
                    #Minimize the loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                #Decay epsilon over steps
                if self.max_epsilon > self.min_epsilon:
                    self.max_epsilon-=self.epsilon_decay
                else: self.max_epsilon=self.min_epsolon
            #Update target_net after specific num_episodes
            if (episode+1)%self.target_update_frequency==0:
                self.target_net.load_state_dict(self.q_net.state_dict())
            #Caculate average
            average_reward=np.mean(reward_per_step)
            average_waiting=np.mean(waiting_per_step)
            average_queued=np.mean(queued_per_step)

            #Insert it
            total_rewards.append((episode,average_reward))
            total_waiting.append((episode,average_waiting))
            total_queued.append((episode,average_queued))

            f.write(f'Episode {episode+1} || Average reward: {average_reward} || Average waiting: {average_waiting} || Average queued: {average_queued}\n')
        f.close()
        return total_rewards,total_waiting,total_queued
