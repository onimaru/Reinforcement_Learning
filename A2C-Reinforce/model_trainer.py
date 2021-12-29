# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

class AgentTrainer:
    def __init__(self,agent,env,n_actions,state_dim,gamma,entropy_coef,λ,loss_func,optimizer_input,max_session_size):
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.λ = λ
        self.max_session_size = max_session_size
        self.loss_func = loss_func
        self.agent = agent
        self.create_optimizer(optimizer_input)

    def create_optimizer(self,optimizer):
        self.optimizer= optimizer(self.agent.parameters(),1e-3)
        
    @torch.no_grad()
    def predict_probs(self,states):
        states = torch.Tensor(states)
        if self.agent.name == 'a2c':
            logit,_ = self.agent(torch.Tensor(states))
        else:
            logit = self.agent(torch.Tensor(states))
        probs = torch.softmax(logit,dim=1).detach().numpy()
        return probs
    
    def generate_session(self,env):
        states, actions, rewards, next_states = [], [], [], []
        s = env.reset()
        for t in range(self.max_session_size):
            action_probs = self.predict_probs(np.array([s]))[0]
            a = np.random.choice(range(self.n_actions),p=action_probs)
            new_s, r, done, info = env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(new_s)

            s = new_s
            if done:
                break

        return states, actions, rewards, next_states
    
    def get_cumulative_rewards(self,rewards):
        G = []
        for t in range(len(rewards)):
            G.append(sum([r*(self.gamma**j) for j,r in enumerate(rewards[t:])]))
        return G

    def to_one_hot(self,y_tensor, ndims):
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
        y_one_hot = torch.zeros(
            y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
        return y_one_hot
    
    def train(self,states, actions, rewards, next_states):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int32)
        cumulative_returns = np.array(self.get_cumulative_rewards(rewards))
        cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)
        if self.agent.name == 'a2c':
            logits,value_s = self.agent(states)
            value_s = value_s.view(-1)
        else:
            logits = self.agent(states)
        probs = nn.functional.softmax(logits, -1)
        log_probs = nn.functional.log_softmax(logits, -1)
        log_probs_for_actions = torch.sum(log_probs * self.to_one_hot(actions, self.n_actions), dim=1)
        entropy =  -torch.sum(probs * log_probs,dim=1)[0]
        
        if self.agent.name == 'a2c':
            loss_actor = -torch.mean(log_probs_for_actions * (cumulative_returns-value_s)) - self.entropy_coef * entropy
            loss_critic = self.λ*self.loss_func(value_s,cumulative_returns)
            loss = loss_actor + loss_critic
        else:
            loss =  -torch.mean(log_probs_for_actions * (cumulative_returns)) - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.agent.name == 'a2c':
            return np.sum(rewards),loss_actor.detach().numpy().tolist(), loss_critic.detach().numpy().tolist()
        else:
            return np.sum(rewards),loss.detach().numpy().tolist(), loss.detach().numpy().tolist()
