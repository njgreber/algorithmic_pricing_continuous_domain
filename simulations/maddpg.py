"""
Multi-agent deep deterministic policy gradient 
9. March 2022
Author: Nicolas Greber
"""

import argparse
from itertools import count

import os, sys, random
import numpy as np
import logging

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import sys


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
        # if the storage is full, delete old and append new 
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

        return np.array(s_t), np.array(s_tp1), np.array(a_t), np.array(r_t), np.array(d_t).reshape(-1, 1)



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
    def __init__(self, state_dim, action_dim, n_agents):
        super(Critic, self).__init__()
        nodes1 = hidden_layers_critic[0]
        nodes2 = hidden_layers_critic[1]

        self.l1 = nn.Linear(state_dim + action_dim * n_agents, nodes1)
        self.l2 = nn.Linear(nodes1 , nodes2)
        self.l3 = nn.Linear(nodes2, action_dim)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class ACTOR(object):
    def __init__(self, state_dim, action_dim, actor_nr, epsilon):
        # initialize actors in maddpg
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim, number_firms).to(device)
        self.critic_target = Critic(state_dim, action_dim, number_firms).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=lr_critic)
        self.epsilon = epsilon

        self.actor_nr = actor_nr


class MADDPG(object):
    # maddpg, with all actors in this class. (because they need to be able to
    # access each others policies
    def __init__(self, state_dim, action_dim, n_agents):
        self.replay_buffer = Replay_buffer() # ok
        self.tau = None

        self.epsilon = epsilon
        self.depsilon = depsilon
        self.min_epsilon = min_epsilon
        self.n_agents = n_agents

        self.agents = []

        for agent_nr in range(self.n_agents):
            self.agents.append(ACTOR(state_dim, action_dim,
                                     agent_nr, self.epsilon))

    def select_action(self, state, step, start_size):
        actions = []
        # start by randomly choosing actions
        if step <= start_size:
            if number_firms == 2:
                actions = [np.random.uniform(-1,1), np.random.uniform(-1,1)]
            elif number_firms == 3:
                actions = [np.random.uniform(-1,1), np.random.uniform(-1,1),
                                np.random.uniform(-1,1)]
        # greedy exploration strategy
        elif exploration_strategy == "greedy":
            for agent in self.agents:
                state = torch.FloatTensor(state).to(device)
                action = agent.actor(state).cpu().data.numpy().flatten()
                epsilon_greedy = np.exp(beta * (step - start_size))
                if random.uniform(0, 1) < epsilon_greedy:
                    action[0] = np.array(random.uniform(-1, 1))
                actions.append(action[0])
        # noisy exploration strategy
        elif exploration_strategy == "noisy":
            for agent in self.agents:
                state = torch.FloatTensor(state).to(device)
                action = agent.actor(state).cpu().data.numpy().flatten()
                action += np.random.normal(0, max(agent.epsilon,self.min_epsilon))
                actions.append(action[0])
        # no exploration strategy
        elif exploration_strategy == "none":
            for agent in self.agents:
                state = torch.FloatTensor(state).to(device)
                action = agent.actor(state).cpu().data.numpy().flatten()
                actions.append(action[0])
        return actions

    def decay_epsilon(self):
        # only used for noisy exploration
        if self.epsilon >= self.min_epsilon:
            self.epsilon -= self.depsilon
        return self.epsilon


    def update(self):

        msbe = [] # mean-squared bellman error

        for agent_i in range(self.n_agents): 

            # Sample replay buffer
            s_t, s_tp1, a_t, r_t, d_t = self.replay_buffer.sample(batch_size)
            # and save tensors separately
            state = torch.FloatTensor(s_t).to(device)
            next_state = torch.FloatTensor(s_tp1).to(device)
            actions = torch.FloatTensor(a_t).to(device)
            done = torch.FloatTensor(1-d_t).to(device)
            reward = torch.FloatTensor(r_t).to(device)

            agents_own_actions = actions[:,agent_i].unsqueeze(1)
            agents_own_reward = reward[:,agent_i].unsqueeze(1)

            if number_firms == 2:
                new_actions = torch.cat((self.agents[0].actor_target(next_state),
                                            self.agents[1].actor_target(next_state)),
                                            dim = 1)
                new_mu_actions = torch.cat((self.agents[0].actor(state),
                                            self.agents[1].actor(state)),
                                            dim = 1)
            if number_firms == 3:
                new_actions = torch.cat((self.agents[0].actor_target(next_state),
                                            self.agents[1].actor_target(next_state),
                                            self.agents[2].actor_target(next_state)),
                                            dim = 1)
                new_mu_actions = torch.cat((self.agents[0].actor(state),
                                            self.agents[1].actor(state),
                                            self.agents[2].actor(state)),
                                            dim = 1)

            new_actions = torch.FloatTensor(new_actions).to(device)
            new_mu_actions = torch.FloatTensor(new_mu_actions).to(device)
            # Compute the target Q value
            
            # ok (also called critic value_)
            next_q_values = self.agents[agent_i].critic_target(next_state, new_actions)
            target_Q = agents_own_reward + (done * delta * next_q_values).detach()

            # Get current Q estimate (also called critic_value)
            current_Q = self.agents[agent_i].critic(state, actions)

            self.agents[agent_i].critic_optimizer.zero_grad() # was self.critic_optim.zero_grad() before.

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            critic_loss.backward(retain_graph = True)
            msbe.append(critic_loss)
            
            self.agents[agent_i].critic_optimizer.step()

            self.agents[agent_i].actor_optimizer.zero_grad()

            # Compute actor loss
            actor_loss = -self.agents[agent_i].critic(state, new_mu_actions).mean()

            # Optimize the actor
            actor_loss.backward()
            self.agents[agent_i].actor_optimizer.step() # ok

            # Update the frozen target models with "soft update" -> the
            # equation with tau (add number)
            if self.tau == None:
                for param, target_param in zip(self.agents[agent_i].critic.parameters(), 
                                               self.agents[agent_i].critic_target.parameters()):
                    target_param.data.copy_(param.data)

                for param, target_param in zip(self.agents[agent_i].actor.parameters(),
                                               self.agents[agent_i].actor_target.parameters()):
                    target_param.data.copy_(param.data)
                self.tau = tau
            else:
                for param, target_param in zip(self.agents[agent_i].critic.parameters(), 
                                               self.agents[agent_i].critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) *
                                            target_param.data)

                for param, target_param in zip(self.agents[agent_i].actor.parameters(),
                                               self.agents[agent_i].actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) *
                                            target_param.data)

        return msbe

