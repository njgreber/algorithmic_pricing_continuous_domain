"""
Logit demand in economic model
2. June 2022
@Author: Nicolas Greber
"""

import random
import numpy as np
from numpy import exp
from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete
from params import *

# calculate max and min price (upper and lower bound)
price_range = (monopoly_price - nash_price)
lower_bound = nash_price - price_range_frame * price_range
upper_bound = monopoly_price + price_range_frame * price_range
price_range_pl_frame = upper_bound - lower_bound

# profit function for bertrand environment
def profit_bertrand(prices : list, costs : list, qualities : list, a0 : float, horiz_diff : float, n : int) -> list:
    # calculate the quantities: numerator
    numerator = [exp((qualities[i] - prices[i]) / horiz_diff) for i in range(n)]
    # and denumerator of the logit demand for each firm as list
    denumerator = np.sum(numerator) + exp(a0 / horiz_diff)
    # calculate profits (q * (p-c))
    profits = [max((numerator[i] / denumerator), 0) * (prices[i] - costs[i]) for i in range(n)]
    # number in [0,1] that tells the firms how high the profits are compared to
    # nash and monopoly profit, 0 being nash, 1 being monopoly
    delta_profits = [(profits[i] - nash_profit) / (monopoly_profit -
                                                   nash_profit) for i in range(n)]
    return delta_profits

# profit function for singh-vives environment
def profit_singhvives(prices : list, gamma : float) -> list:
    # calculating demand
    demand0 = (1 - gamma - prices[0] + gamma * prices[1]) / (1 - gamma**2)
    demand1 = (1 - gamma - prices[1] + gamma * prices[0]) / (1 - gamma**2)
    # calculate profits
    profit0 = demand0 * prices[0]
    profit1 = demand1 * prices[1]
    # delta profits
    delta_profits = [(profit0 - nash_profit) / (monopoly_profit - nash_profit),\
                     (profit1 - nash_profit) / (monopoly_profit - nash_profit)]
    return delta_profits

# Bertrand environment built on gym 
class Bertrand(Env):
    def __init__(self):
        self.state = [random.uniform(lower_bound, upper_bound), random.uniform(lower_bound, upper_bound)]
        self.action_space = Box(low= -1, high= 1,
                                shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=lower_bound, high=upper_bound, shape
                                     = (number_firms,), dtype=np.float32)           
    def step(self, actions):
        # apply action (new prices)
        prices = [None] * len(actions)
        for i in range(len(actions)):
            prices[i] = (price_range_pl_frame * (actions[i] / 2 + 0.5) +
                          lower_bound)
        state = prices
        delta_profits  = profit_bertrand(prices, costs, \
            qualities, outside_option, horizontal_diff, number_firms)
        done = False
        info = {}
        # apply action on state
        return state, prices, done, info, delta_profits

# SinghVives built on gym
class SinghVives(Env):
    def __init__(self):
        self.state = [random.uniform(lower_bound, upper_bound), random.uniform(lower_bound, upper_bound)]
        self.action_space = Box(low= -1, high= 1,
                                shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=lower_bound, high=upper_bound, shape
                                     = (number_firms,), dtype=np.float32)           

    def step(self, actions):
        # apply action (new prices)
        prices = [None] * len(actions)
        for i in range(len(actions)):
            prices[i] = (price_range_pl_frame * (actions[i] / 2 + 0.5) +
                          lower_bound)
        state = prices
        delta_profits  = profit_singhvives(prices, gamma)
        done = False
        info = {}
        # apply action on state
        return state, prices, done, info, delta_profits


















