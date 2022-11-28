"""
Main file: run simulations
3. June 2022
@Author Nicolas Greber
"""
import os, sys, random
import numpy as np
import logging
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from params import *
from ddpg import DDPG, state_dim, action_dim, env
from maddpg import MADDPG, state_dim, action_dim, env

# simulation preparation
if not os.path.exists(directory):
      os.makedirs(directory)

directory = directory + '/' + run + '/'
if not os.path.exists(directory):
      os.makedirs(directory)

if seed:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

# create copy of current params.py
with open('params.py','r') as paramlist, open('{}/params.txt'.\
    format(directory),'a') as copy:
    for line in paramlist:
        copy.write(line)

####################################################################
# MAIN LOOP FOR THE SIMULATION
# loops through simulation runs per experiment
for current_run in range(simulation_runs):
    # get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(directory))[2]
    run_num = len(current_num_files)
    # create new log file for each run 
    log_f_name = directory + run + '_' + str(run_num) + ".csv"
    # prepare log
    log_f = open(log_f_name,"w+")
    if number_firms == 2:
        log_f.write('timestep,reward1,reward2,prices1,prices2,epsilon,loss1,loss2\n')
    if number_firms == 3:
        log_f.write('timestep,reward1,reward2,reward3,prices1,prices2,prices3,loss1,loss2,loss3\n')

    # EITHER DDPG OR MADDPG. Two separate loops.
    ####################################################################
    # DDPG
    if algorithm == "ddpg":
        # initialize agents, variables, step counter
        agent0 = DDPG(state_dim, action_dim, exploration_strategy)
        agent1 = DDPG(state_dim, action_dim, exploration_strategy)
        critic_loss0 = 0
        critic_loss1 = 0
        if number_firms == 3:
            agent2 = DDPG(state_dim, action_dim, exploration_strategy)
            critic_loss2 = 0
        step = 0
        k = 0
        if number_firms == 2:
            state = np.array([0,0])
        if number_firms == 3:
            state = np.array([0,0,0]) 
        # start looping though periods
        for t in range(max_timesteps):
            # select actions for all actors
            action0 = agent0.select_action(state, step, start_learning)
            action1 = agent1.select_action(state, step, start_learning)
            if number_firms == 3:
                action2 = agent2.select_action(state, step, start_learning)
            # put them together (gym.env thinks it interacts with one agent)
            if number_firms == 2:
                action = [action0[0], action1[0]]
            elif number_firms == 3:
                action = [action0[0], action1[0], action2[0]]
            # next step in gym.environment with profits for both agents
            next_state, prices, done, info, delta_profits = env.step(action)
            # add the new step to the replay buffer
            agent0.replay_buffer.push((state, next_state, action0, delta_profits[0],
                                       float(done)))
            agent1.replay_buffer.push((state, next_state, action1, delta_profits[1],
                                       float(done)))
            if number_firms == 3:
                agent2.replay_buffer.push((state, next_state, action2, \
                                       delta_profits[2], \
                                       float(done)))
            # update the agents
            if step >= start_learning :
                critic_loss0 = agent0.update(k)
                critic_loss1 = agent1.update(k)
                if number_firms == 3:
                    critic_loss2 = agent2.update(k)
                k += 1
            if step >= start_learning and exploration_strategy == "noisy" and \
                    step % 10000 == 0: 
                agent0.decay_epsilon()
                agent1.decay_epsilon()
                if number_firms == 3:
                    agent2.decay_epsilon()
            # save transition in csv
            if number_firms == 2:
                log_f.write("{},{},{},{},{},{},{},{}\n".format(step, \
                        round(delta_profits[0], 3), \
                        round(delta_profits[1], 3), \
                        round(prices[0],3), round(prices[1],3), \
                        round(agent0.epsilon,3), critic_loss0, \
                        critic_loss1))
            if number_firms == 3:
                log_f.write("{},{},{},{},{},{},{},{},{},{}\n".maddpgat(step, \
                    round(delta_profits[0], 3), round(delta_profits[1], 3), \
                    round(delta_profits[2], 3), round(prices[0],3), \
                    round(prices[1],3), round(prices[2],3), \
                    critic_loss0, critic_loss1, critic_loss2))
            log_f.flush()
            step += 1
            state = next_state
        log_f.close()
        env.close()

    ####################################################################
    # MADDPG
    elif algorithm == "maddpg":
        # prepare agents and variables
        agents = MADDPG(state_dim, action_dim, number_firms)
        step = 0
        k = 0
        msbe = [0] * number_firms
        state = np.array([0]*number_firms)
        # start looping through periods
        for t in range(max_timesteps):
            # select actions for all actors
            actions = agents.select_action(state, step, start_learning)
            next_state, prices, done, info, delta_profits = env.step(actions)
            # add the new step to the replay buffer
            agents.replay_buffer.push((state, next_state, actions, delta_profits, float(done)))
            # update the agents
            if step >= start_learning:
                msbe = agents.update()
            if step >= start_learning and exploration_strategy == "noisy" and \
                    step % 10000 == 0: 
                agents.decay_epsilon()
            # log transitions in csv
            if number_firms == 2:
                log_f.write("{},{},{},{},{},{},{},{}\n".format(step, \
                        round(delta_profits[0], 3), \
                        round(delta_profits[1], 3), \
                        round(prices[0],3), round(prices[1],3), \
                        round(agents.epsilon,3), msbe[0], msbe[1]))
            if number_firms == 3:
                log_f.write("{},{},{},{},{},{},{},{},{},{}\n".format(step, \
                    round(delta_profits[0], 3), round(delta_profits[1], 3), \
                    round(delta_profits[2], 3), round(prices[0],3), \
                    round(prices[1],3), round(prices[2],3), \
                    msbe[0], msbe[1], msbe[2]))
            log_f.flush()
            step += 1
            state = next_state
        log_f.close()
        env.close()



