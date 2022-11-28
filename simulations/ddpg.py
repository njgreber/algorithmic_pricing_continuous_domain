"""
Deep deterministic policy gradient 
9. March 2022
Author: Nicolas Greber
"""

import argparse
from itertools import count

import os, sys, random
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from params import *
from env import Bertrand
from env import SinghVives

if environment == "SinghVives":
    env = SinghVives()
elif environment == "Bertrand":
    env = Bertrand()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=capacity):
        # initialize the storage as list
        self.storage = []
        # define max_size of storage (defined above, hyper-parameter)
        self.max_size = max_size
        # if storage full, overwrite first the 0-th observation
        self.ptr = 0

    def push(self, data):
        """
        data : (s_t, s_t+1, a_t, R_t, done) of one episode
        pushes (s_t, s_t+1, a_t, R_t, done) to storage of earlier functions. if
        the storage is full all observations are overwritten one after the
        other (starting with the earliest). 
        """
        # if the storage is full
        if len(self.storage) == self.max_size:
            # overwrite observation "ptr", which starts at 0
            self.storage[int(self.ptr)] = data
            # and increases to storage length, then 0 again
            self.ptr = (self.ptr + 1) % self.max_size
        # if storage is not full yet (only in the beginning)
        else:
            # data is appended
            self.storage.append(data)

    def sample(self, batch_size):
        """
        takes samples of the storage with length batch_size
        the abbreviations stand for state_t, state_t, action_t, reward_t and
        done_t. 
        """
        # random integers from 0 to lenght of storage of lenght batch_size
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        s_t, s_tp1, a_t, r_t, d_t = [], [], [], [], []
        
        # for every random integer from above stores the corresponding
        # observation from storage in batch
        for i in ind:
            S_t, S_tp1, A_t, R_t, D_t = self.storage[i]
            
            # append transition parameters to batches
            s_t.append(np.array(S_t, copy=False))
            s_tp1.append(np.array(S_tp1, copy=False))
            a_t.append(np.array(A_t, copy=False))
            r_t.append(np.array(R_t, copy=False))
            d_t.append(np.array(D_t, copy=False))

        return np.array(s_t), np.array(s_tp1), np.array(a_t), np.array(r_t).reshape(-1, 1), np.array(d_t).reshape(-1, 1)



class Actor(nn.Module):
    """
    network design of the actor
    """
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        
        nodes1 = hidden_layers_actor[0]
        nodes2 = hidden_layers_actor[1]

        self.l1 = nn.Linear(state_dim, nodes1)
        self.l2 = nn.Linear(nodes1, nodes2)
        self.l3 = nn.Linear(nodes2, action_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x

class Critic(nn.Module):
    """
    network design of the critic
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        nodes1 = hidden_layers_critic[0]
        nodes2 = hidden_layers_critic[1]

        self.l1 = nn.Linear(state_dim + action_dim, nodes1)
        self.l2 = nn.Linear(nodes1 , nodes2)
        self.l3 = nn.Linear(nodes2, action_dim)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class DDPG(object):
    def __init__(self, state_dim, action_dim, exploration_strategy):
        # initilialize
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=lr_critic) 
        self.replay_buffer = Replay_buffer()

        self.epsilon = epsilon
        self.depsilon = depsilon
        self.min_epsilon = min_epsilon
        self.tau = None

    def select_action(self, state, step, start_size):
        state = torch.FloatTensor(state).to(device)
        if step <= start_size:
            action = self.actor(state).cpu().data.numpy().flatten()
            action[0] = np.array(random.uniform(-1, 1))
        # select action with greedy strategy
        elif exploration_strategy == "greedy":
            action = self.actor(state).cpu().data.numpy().flatten()
            epsilon_greedy = np.exp(beta * (step - start_size))
            if random.uniform(0, 1) < max(epsilon_greedy, min_prob_expl):
                action[0] = np.array(random.uniform(-1, 1))
        # select action with Gaussian noise
        elif exploration_strategy == "noisy":
            action = self.actor(state).cpu().data.numpy().flatten()
            action += np.random.normal(0, max(self.epsilon,self.min_epsilon))
        # select action without added exploration
        elif exploration_strategy == "none":
            action = self.actor(state).cpu().data.numpy().flatten()
        return action


    def decay_epsilon(self):
        # used only if noisy exploration is used
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.depsilon
        return self.epsilon

    def update(self, step):

        # Sample replay buffer
        s_t, s_tp1, a_t, r_t, d_t = self.replay_buffer.sample(batch_size)
        # and save tensors separately
        state = torch.FloatTensor(s_t).to(device)
        next_state = torch.FloatTensor(s_tp1).to(device)
        action = torch.FloatTensor(a_t).to(device)
        done = torch.FloatTensor(1-d_t).to(device)
        reward = torch.FloatTensor(r_t).to(device)

        # Compute the target Q value
        next_q_values = self.critic_target(next_state,
                                      self.actor_target(next_state))
        target_Q = reward + (done * delta * next_q_values).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)


        # Optimize the critic
        self.critic_optimizer.zero_grad()

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        critic_loss.backward()

        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        actor_loss.backward() 
        self.actor_optimizer.step()

        # Update the frozen target models with "soft update" -> the
        # equation with tau (add number)
        if self.tau == None:
            for param, target_param in zip(self.critic.parameters(), 
                                           self.critic_target.parameters()):
                target_param.data.copy_(param.data)

            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(param.data)
            self.tau = tau
        else:
            for param, target_param in zip(self.critic.parameters(), 
                                           self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) *
                                        target_param.data)

            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) *
                                        target_param.data)
        return critic_loss





