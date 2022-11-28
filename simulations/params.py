"""
Parameters for simulation runs
@ Author: Nicolas Greber
Created 31.07.2022
"""

#######################################################################
# General simulation settings

directory = "results"               # where results are stored
algorithm = "maddpg"                # which algo ("ddpg" or "maddpg")
topic = "base_case"                 # topic of the simulation (choose freely)
run = algorithm + '_' + topic       # name of folder 
seed = True                         # if True, simulation follows seed
random_seed = 1                     # seed
simulation_runs = 3                 # number of runs per simulations
environment = "Bertrand"            # "Bertrand" or "SinghVives"
max_timesteps =  3000000            # how many periods per simulation run

#######################################################################
# Economic model parameters (for "Bertrand")

number_firms = 2                    # number of firms (2 or 3)
horizontal_diff = 0.25              # horizontal differentiation
qualities = [2] * number_firms      # list with qualities of firms
outside_option = 0                  # quality of outside option
costs = [1] * number_firms          # list with costs 
price_range_frame = 0.2             # action set extends nash price and
                                    # monopoly price by so many percent-
                                    # age points upwards and downwards.
                                    # independent of the environment. 
                                    
# If one of number_firms, horizontal-diff, qualities, outside_option 
# or costs changes, the following parameters need adjustment. For the
# robustness tests, these parameters are included below.

nash_price = 1.47293                # nash price
monopoly_price = 1.92498            # monopoly price
monopoly_profit = 0.33749           # collusion profit per firm
nash_profit = 0.22293               # nash profit

#######################################################################
# Algorithm hyper-parameters

delta = 0.99                        # discount factor
lr_actor = 0.0001                   # learning rate actor
lr_critic = 0.001                   # learning rate critic
capacity = 100000                   # replay buffer size
start_learning = 100000             # period when agent starts learning
batch_size = 100                    # mini batch size
tau = 0.005                         # target smoothing coefficient
hidden_layers_actor = [16, 16]      # nodes in hidden layers (2 layers
                                    # hard-coded) 
hidden_layers_critic = [16, 16]     # nodes in hidden layers (2 layers
                                    # hard-coded)
exploration_strategy = "greedy"     # exploration type ("greedy" or "noisy")
beta = -0.000004                    # epsilon-greedy exploration beta
min_prob_expl = 0.0                 # min_prob_expl



#######################################################################

# Robustness test parameters:
#######################################################################
# 1. gaussian exploration:          # set exploration_strategy = "noisy"
gaussian_start_decrease= 500000     # when variance linearly decreases
epsilon = 0.2                       # starting variance
depsilon = 0.001                    # increment by which variance decreases
min_epsilon = 0.0                   # minimum variance

#######################################################################
# 2. linear demand:                 # set environment = "SinghVives"
                                    # and uncomment following four lines
#nash_price = 0.33333
#monopoly_price = 0.5
#monopoly_profit = 0.16667
#nash_profit = 0.14815

gamma = 0.5                         # differentiation in Singh Vives

#######################################################################
# 3. number of agents:              # set number_firms = 3 and uncomment
#nash_price = 1.37016
#monopoly_price = 2.0 
#monopoly_profit = 0.25
#nash_profit = 0.12016 
#######################################################################
# 4. larger action set              # set discount_factor to 0.99
# price_range_frame = 0.2           # action set extends nash price and
                                    # monopoly price by so many percent-
                                    # age points upwards and downwards.
                                    # independent of the environment. 


#######################################################################
# 5. discount factor                # set discount_factor to 0.99

#######################################################################

