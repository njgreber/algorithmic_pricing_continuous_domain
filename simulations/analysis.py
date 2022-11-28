import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from params import *
import numpy as np

# if you ran the simulation, you can use this file to make plots.
# a lot of the plotting is hard-coded to get the plots from the thesis. 
# just change the following variable to the "run" name of your simulation,
# e.g. maddpg_base_case or ddpg_3firms, and run the file. The plots will
# be saved in the same file as the simulation csv files.



simulation = "maddpg_base_case"
number_firms = 2
# SIMULATION RUNS WITH PERIOD ON X-AXIS
#########################################################
# plot the progression of profits of all three runs
fig, axs = plt.subplots(3, sharex=True, sharey=True)
mean_len = 100
for i in [1,2,3]:
    df = pd.read_csv('~/Desktop/master_thesis/simulations/results/{}/{}_{}.csv'.format(simulation,simulation,i))
    print(df['epsilon'][-1:])
    axs[i-1].plot(df['reward1'][:3000000].rolling(mean_len).mean(), label = \
                  'Agent 1', alpha = 0.8, color = 'b')
    axs[i-1].plot(df['reward2'][:3000000].rolling(mean_len).mean(), label = \
                  'Agent 2', alpha = 0.8, color = 'g')
    if number_firms == 3:

        axs[i-1].plot(df['reward3'][:3000000].rolling(mean_len).mean(), label = \
                      'Agent 3', alpha = 0.8, color = 'r')
    axs[i-1].axhline(y = 1 , color = 'c', linestyle = ':', label = \
                     'Collusion', alpha = 1.0)
    axs[i-1].axhline(y = 0 , color = 'k', linestyle = ':', label = \
                     'Nash', alpha = 1.0)
    if i in [1,3]:
        axs[i-1].set(ylabel='Run {}'.format(i))
    else:
        axs[i-1].set(ylabel='Average profit gain\nRun {}'.format(i))
    if i== 2:
        axs[i-1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.ylim([-1., 2])
    if i==3:
        axs[i-1].set(xlabel='Periods')
fig.subplots_adjust(right=0.75)   
fig.set_size_inches(7,4)
plt.show()
fig.savefig("results/{}/{}_profits".format(simulation,simulation), dpi = 300)
# plot the progression of prices of all three runs
fig, axs = plt.subplots(3, sharex=True, sharey=True)
for i in [1,2,3]:
    df = pd.read_csv('~/Desktop/master_thesis/simulations/results/{}/{}_{}.csv'.format(simulation,simulation,i))
    axs[i-1].plot(df['prices1'][:3000000].rolling(mean_len).mean(), label = \
                  'Agent 1', alpha = 0.8, color = 'b')
    axs[i-1].plot(df['prices2'][:3000000].rolling(mean_len).mean(), label = \
                  'Agent 2', alpha = 0.8, color = 'g')
    if number_firms == 3:
        axs[i-1].plot(df['prices3'][:3000000].rolling(mean_len).mean(), label = \
                      'Agent 3', alpha = 0.8, color = 'r')
    axs[i-1].axhline(y = monopoly_price , color = 'c', linestyle = ':', label = \
                     'Collusion', alpha = 1.0)
    axs[i-1].axhline(y = nash_price , color = 'k', linestyle = ':', label = \
                     'Nash', alpha = 1.0)
    if i in [1,3]:
        axs[i-1].set(ylabel='Run {}'.format(i))
    else:
        axs[i-1].set(ylabel='Average prices\nRun {}'.format(i))
    if i== 2:
        axs[i-1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.ylim([nash_price-0.2, monopoly_price+0.2])
    if i==3:
        axs[i-1].set(xlabel='Periods')
fig.subplots_adjust(right=0.75)   
fig.set_size_inches(7,4)
plt.show()
fig.savefig("results/{}/{}_prices".format(simulation,simulation), dpi = 300)


# plot the progression of the log mean squared bellman-error of all three runs
fig, axs = plt.subplots(3, sharex=True, sharey=True)
for i in [1,2,3]:
    df = pd.read_csv('~/Desktop/master_thesis/simulations/results/{}/{}_{}.csv'.format(simulation,simulation,i))
    axs[i-1].plot(np.log(df['loss1'])[:3000000].rolling(mean_len).mean(), label = \
                  'Agent 1', alpha = 0.8, color = 'b')
    axs[i-1].plot(np.log(df['loss2'])[:3000000].rolling(mean_len).mean(), label = \
                  'Agent 2', alpha = 0.8, color = 'g')
    if number_firms == 3:

        axs[i-1].plot(np.log(df['loss3'])[:3000000].rolling(mean_len).mean(), label = \
                      'Agent 3', alpha = 0.8, color = 'r')
    if i in [1,3]:
        axs[i-1].set(ylabel='Run {}'.format(i))
    else:
        axs[i-1].set(ylabel='Average critic loss\nRun {}'.format(i))
    if i== 2:
        axs[i-1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    #plt.ylim([-0.5, 1.2])
    if i==3:
        axs[i-1].set(xlabel='Periods')
fig.subplots_adjust(right=0.75)   
fig.set_size_inches(7,4)
plt.show()
fig.savefig("results/{}/{}_losses".format(simulation,simulation), dpi = 300)

# TABLE
#########################################################
profit_mean = []
profit_std = []
price_mean = []
for dataframe in [1,2,3]:
    profit_mean_input = []
    profit_std_input = []
    price_mean_input = []
    for i in [1,2]:

        df = pd.read_csv('~/Desktop/master_thesis/simulations/results/{}/{}_{}.csv'.format(simulation,simulation,dataframe))
        profit_mean_input.append(round(np.mean(df["reward{}".format(i)][-1000:]),3))
        profit_std_input.append(round(np.std(df["reward{}".format(i)][-1000:]),3))
        price_mean_input.append(round(np.mean(df["prices{}".format(i)][-1000:]),3))
    profit_mean.append(profit_mean_input)
    profit_std.append(profit_std_input)
    price_mean.append(price_mean_input)
print(profit_mean)
output = {"Avg. profit": profit_mean, "Sd. profit": profit_std, "Avg. price (Nash: 1.472)":
          price_mean}
out_data = pd.DataFrame(output, index = ["run 1", "run 2", "run 3"])
out_data.to_csv("results/{}/{}_table.csv".format(simulation,simulation))
print(out_data)

# only prepared for base case in Bertrand setting
# SCATTERPLOT WITH PRICES OF ALL THREE RUNS
#########################################################
from scipy.optimize import fmin
from env import profit_bertrand

# first order derivate of firm 1, profit function, change e with np.exp.
# fsolve for 0 with different p2, then you have the line you need!
colors = ["tab:red", "tab:olive", "tab:purple"]

plt.figure(figsize=(6, 4), dpi=100)

m = 0.25
a = 2.0
p = list(np.arange(1.1, 2.1, 0.01))
max_x = []
for j in range(len(p)):
    max_x.append(fmin(lambda x: -profit_bertrand([x, p[j]],[1,1], [2,2], 0,
                                                0.25, 2)[0], 1))
for i in [1,2,3]:
    df=pd.read_csv('~/Desktop/master_thesis/simulations/results/{}/{}_{}.csv'.format(simulation,simulation,i))
    plt.scatter(df['prices1'][-100:],df['prices2'][-100:],
               label = "Run {}".format(i),alpha = 0.8, color = colors[i-1])
    plt.ylim([nash_price-0.25, monopoly_price+0.05])
    plt.xlim([nash_price-0.25, monopoly_price+0.05])
    plt.xlabel("Avg. Price Agent 1")
    plt.ylabel("Avg. Price Agent 2")
    
plt.plot(max_x, p, label = "BR Agent 1", color = "b")
plt.plot(p, max_x, label = "BR Agent 2", color = "g")
plt.hlines(monopoly_price,1,3, colors = "c", linestyle = ':',label = "Collusion")
plt.vlines(monopoly_price,1,3, colors = "c", linestyle = ':')
plt.hlines(nash_price,1,3, colors = "k", linestyle = ':',label = "Nash")
plt.vlines(nash_price,1,3, colors = "k", linestyle = ':')
plt.subplots_adjust(right=0.7)   

plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.savefig("results/{}/{}_scatter".format(simulation,simulation))
plt.show()
#fig.suptitle('MADDPG base case: Profit gain')





