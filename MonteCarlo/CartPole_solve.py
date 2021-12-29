import numpy as np
import gym
import sys

env = gym.make('CartPole-v0')
    
def softmax(state,theta):
    a = np.matmul(state,theta)
    return np.exp(a)/np.sum(np.exp(a))

def policy(softmax_probs):
    if np.random.rand() < softmax_probs[0]:
        return 0
    else: return 1

def grads(action,softmax_probs):
    s = softmax_probs
    if action == 0:
        return np.array([s[1],-s[1]])[None,:]
    else: 
        return np.array([-s[0],s[0]])[None,:]

def get_episode(theta):
    state = env.reset()
    episode = []
    while True:
        #env.render()    # uncomment this to enable render mode
        s = softmax(state,theta)
        action = policy(s)
        next_state, reward, done, _ = env.step(action)
        
        episode.append((state,reward,action,s))
        state = next_state
        if done: break
    return episode

def cp_play(n_episodes,alpha,y):
    R = []
    episode_length = []
    theta = np.random.rand(4,2)
    for i in range(n_episodes):
        episode = get_episode(theta)
        states = [item[0] for item in episode]
        rewards = [item[1] for item in episode]
        actions = [item[2] for item in episode]
        softs = [item[3] for item in episode]
        R.append(sum(rewards))
        episode_length.append(len(episode))
        grad = [grads(i,s) for i,s in zip(actions,softs)]
        
        for t in range(len(grad)):
            theta += alpha*np.array(np.dot(states[t][None,:].T,grad[t])*sum([r*(y**r) for i,r in enumerate(rewards[t:])]))
        
        l=100
        if len(R)>=l:
            for j in range(l,len(R)):
                if np.mean(R[j-l:j])>=195:
                    print('Solved in:')
                    env.close()
                    sys.exit('Episode {} with average reward in last 100 steps: {}.'.format(j-l, np.mean(R[j-l:j])))
        
        print('Episodes: {}, Average reward: {}'.format(i,sum(rewards)))
    return R

cp_play(1000,0.005,0.99)

env.close()
sys.exit('Not solved in 1000 steps')