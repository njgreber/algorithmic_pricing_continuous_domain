# Algorithmic pricing in the continuous domain
This repository includes the code for my master's thesis in Economics at the University of St. Gallen. I analyzed the behavior of two reinforcement learning pricing algorithms with continuous action spaces in game-theoretic environments. This code investigates whether the algorithms DDPG and MADDPG, which operate in the continuous action space, learn to collude. The code for the simulations is written in Python.

If you are interested in a copy of my master thesis, please contact me: nicolas.greber "at" gmail.com

## Prerequisites
* Make sure you are using Python 3.9 or higher
* Initialize a virtual environment
* Install packages from `requirements.txt`

## Code architecture

### General
All files containing code are in the folder called simulations. The code is constructed in a modular form so that additional algorithms or environments can be easily added. If the only goal is to recreate the simulation results, using the two files `params.py` and `main.py` is sufficient.

### Simulation settings
The file `params.py` includes all general simulation settings, economic model parameters and algorithm hyper-parameters, as well as standard settings for the robustness tests conducted in the thesis. Read the thesis for further explanation of the parameters listed here. 

#### General simulation settings

The parameters defined in this section are for general settings, not specific to the economic model or the algorithms. 

```
directory = "results"               where the results are stored
algorithm = "ddpg"                  choose algorithm ("ddpg" or "maddpg")
topic = "linear_demand"             topic of the simulation
run = algorithm + '_' + topic       name of results (automatic)
seed = True                         if True, simulation follows a seed
random_seed = 3                     seed for replication
simulation_runs = 3                 number of runs per simulations
environment = "Bertrand"            game-theoretic environment ("Bertrand"
                                    or "SinghVives")
max_timesteps =  3000000            how many periods per simulation run
```

#### Economic model parameters (for Bertrand Model with logit demand)

The parameters defined in this part are used for the Bertrand duopoly model described in chapter.

```
number_firms = 2                    number of firms (2 or 3)
horizontal_diff = 0.25              horizontal differentiation
qualities = [2] * number_firms      list with qualities of firms
outside_option = 0                  quality of outside option
costs = [1] * number_firms          list with marginal costs 
price_range_frame = 0.2             defines the size of the action set of the algorithms. 
                                    Given the difference between monopoly price and Nash 
                                    price is standardized to 1, this parameter adds so 
                                    much on both ends for the action set. Hence, maximum 
                                    price would be 
                                    Nash price + (monopoly price - Nash price) * 1.2.
                                    
```

If one of `number_firms`, `horizontal_diff`, `qualities`, `outside_option` or  `costs` changes, the Nash price, monopoly price, collusion profit per firm and Nash profit in `params.py` need adjustment as well. Calculation for these parameters are conducted separately.

#### Algorithm hyper-parameters 

This list includes all hyper-parameters of the algorithms used. Because the parameters are similar for both DDPG and MADDPG, they are shared for both algorithms. 

```
delta = 0.95                        discount factor
lr_actor = 0.0001                   learning rate of the actor network
lr_critic = 0.001                   learning rate of the critic network
capacity = 100000                   replay buffer size
start_learning = 100000             period when agent starts learning
batch_size = 100                    mini batch size
tau = 0.005                         target smoothing coefficient
hidden_layers_actor = [16, 16]      nodes in hidden layers (2 layers hard-coded) 
hidden_layers_critic = [16, 16]     nodes in hidden layers (2 layer hard-coded)
exploration_strategy = "greedy"     exploration type ("greedy" or "noisy")
beta = -0.000004                    epsilon-greedy exploration beta
```

#### Robustness test parameters:

For two of the robustness tests, further parameters are needed.

```
gaussian_start_decrease= 500000     when variance linearly decreases
epsilon = 0.2                       starting variance
depsilon = 0.001                    increment by which variance decreases
min_epsilon = 0.05                  minimum variance
gamma = 0.5                         differentiation in Singh Vives
```

### Running the simulation
To conduct the simulations, run `main.py`. 

### Further files
* `ddpg.py` and `maddpg.py` hold the code for deep deterministic policy gradient and multi-agent deep deterministic policy gradient respectively.
* `env.py`contains the code for two game-theoretic environments. The standard environment is the simultanous Bertrand pricing model with logit demand. The second environment is from a paper called "Price and Quantity Competition in a Differentiated Duopoly " by Singh and Vives (1984). For a detailed description of the environments please consult the thesis. 
* `sim_analysis.py` plots simulation results and creates tables. Change `path` and `simulation` to get plots. A large part of this file is hard-coded. This means that if the simulation of robustness tests is analyzed, details in the code (e.g. Nash price, monopoly price) have to be changed. 

## Authors
* Nicolas Greber



