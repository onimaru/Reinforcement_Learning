import random
from random import randint
import pandas as pd
import numpy as np
import gym
import matplotlib.pyplot as plt
from time import sleep

env = gym.make('FrozenLake-v0')

def policy(Q,state,epsilon):
    if np.random.rand() < epsilon:
        action = np.random.choice([0,1,2,3])
    else:
        action = np.argmax(Q[state,:])
    return action

def reset_memory():
    return np.random.rand(env.observation_space.n, env.action_space.n)*0.0001
    
def memory(Q,next_state,state,action,reward,alpha,gamma):
    return (1-alpha)*Q[state,action] + alpha*(reward +gamma*np.max(Q[next_state,:])) 
    

Q = reset_memory()
R = []
episodes=10000
epsilon=0.1
alpha=0.9
gamma=0.9
for i in range(episodes):
    state = env.reset()
    r = 0
    while True:
        maze = env.render()
        print('Episodes: {} | Total reward: {}'.format(i,sum(R)))
        action = policy(Q,state,epsilon)
        next_state,reward,done,_ = env.step(action)
        Q[state,action] = memory(Q,next_state,state,action,reward,alpha,gamma)
        state = next_state
        r+=reward
        if done:
            maze = env.render()
            print('Episodes: {} | Total reward: {}'.format(i,sum(R)))
            print('Episode end!')
            break
        sleep(0.2)
    R.append(r)

print('Episodes: {} | Total reward: {}'.format(i,sum(R)))
print('Training end!')