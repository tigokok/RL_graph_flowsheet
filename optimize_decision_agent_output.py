import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from collections import deque
from itertools import accumulate
import copy
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from gym.spec_gym import SpecGym
from agents.discrete_agent import DiscreteAgent
from nodeFlowsheeter.flowsheetEncoder import FlowsheetEncoder

# ==================== Colors ====================
from matplotlib.colors import to_rgb
light_tone = to_rgb("#B8B3E9")
dark_tone = to_rgb("#5D51CD")
light_tone_alt = to_rgb("#F3DFC1")
dark_tone_alt = to_rgb("#DB9E43")


# ==================== Configurable Parameters ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Agent architecture parameters
in_channels = 4
out_channels = 1
edge_attr_dim = 5
hidden_dim = 32
hidden_channels = 8
num_actions = 2
depth = 2

# ==================== Initialization ====================
# Saving
save = False
save_name = 'fixed_comp'

# Make gym
comp_mode = 'random'
node_types = ['feed', 'output', 'empty', 'ideal_dstwu']

feed_flow = 100
components = ['cyclo-pentane', 'n-hexane', 'n-heptane', 'n-octane']
feed_composition = np.array([0.4, 0.3, 0.2, 0.1])


gym = SpecGym(comp_mode=comp_mode, feed_flow=feed_flow,
              components=components,
              feed_composition=feed_composition,
              node_types=node_types)


# Transfer learning
# If true you can pick up previous weights and architecture and train
# i.e. if you first train on light fixed mixture (see: spec_gym.py)
# and then use the weights to train on random, performance increases

arch_params = torch.load("checkpoints/decision_architecture_r8.pth")
agent = DiscreteAgent(**arch_params)
agent.load_state_dict(torch.load("checkpoints/decision_weights_r8.pth"))
agent.to(device)

agent.train()


# ==================== Generate flowsheet ====================

n_step = 0
max_steps = 12
done = False

flowsheet, info = gym.reset()

while not done and n_step < max_steps:

    # make make for empty nodes and get fingerprint for the graph
    mask = torch.tensor(flowsheet.get_empty_nodes_mask())
    fingerprint = FlowsheetEncoder(flowsheet, in_channels).detach()

    # Calculate Q values
    q_vals = agent.forward(fingerprint) 

    masked_q_vals = q_vals.clone()
    masked_q_vals[~mask.bool()] = -1e9

    # Best action is highest argmax(Q values)
    # Which corresponds to highest expected reward
    flat_index = masked_q_vals.argmax()
    a1, a2 = np.unravel_index(flat_index, q_vals.shape)

    a1 = int(a1)
    a2 = int(a2)


    flowsheet, reward, done, info = gym.step([a1, a2])
    n_step += 1


flowsheet.display_graph()
# ==================== Replace ideal colums by shortcuts ====================

from nodeFlowsheeter.simulation import Simulation
from nodeFlowsheeter.nodes import *
from scipy.optimize import minimize
import numpy as np

column_ids = []
initial_params = []

# Initialize columns and initial parameters
for node_id, node in flowsheet.nodes.items():
    if node.node_type == 'ideal_dstwu':
        dstwu = DstwuNode(node_id, node.lk, node.hk, node.recov_lk, node.recov_hk)
        flowsheet.add_unit([dstwu])

        column_ids.append(node_id)
        initial_params.extend([node.recov_lk, node.recov_hk]) 


# Bounds: for each column two parameters (lk_recov, hk_recov)
bounds = []
for _ in column_ids:
    bounds.append((0.75, 0.99))  # lk_recov bounds
    bounds.append((0.01, 0.1))   # hk_recov bounds

profit_log = []
params_log = []
tac_log = []
Q_log = []

# ==================== Optimize column ====================
def run_sim(params):
    for i, node_id in enumerate(column_ids):
        recov_lk = params[2*i]
        recov_hk = params[2*i + 1]

        flowsheet.nodes[node_id].recov_lk = recov_lk
        flowsheet.nodes[node_id].recov_hk = recov_hk


    sim = Simulation(flowsheet=flowsheet, components=components)
    sim.run_simulation()

    val, tac = flowsheet.estimate_flowsheet_value(gym.spec)
    

    # Log profit and params
    profit_log.append(val)
    tac_log.append(-tac)
    params_log.append(params.copy())

    qs = []
    for i in column_ids:
        qs.append(flowsheet.nodes[i].Q)
    Q_log.append(qs)

    return -val  # minimize negative profit = maximize profit

result = minimize(run_sim, initial_params, bounds=bounds, method='Nelder-Mead')



# Run flowsheet one last time with optimal params
for i, node_id in enumerate(column_ids):
    flowsheet.nodes[node_id].recov_lk = result.x[2*i]
    flowsheet.nodes[node_id].recov_hk = result.x[2*i + 1]

sim = Simulation(flowsheet=flowsheet, components=components)
sim.run_simulation()
final_val = flowsheet.estimate_flowsheet_value(gym.spec)

print(f"Optimal parameters: {result.x}")
print(f"Final profit after optimization: {final_val}")
flowsheet.print_streams()

s = 5

# Normalized profit
plt.figure(figsize=(10, 5))
plt.scatter(range(len(profit_log)), profit_log, s = s, color=dark_tone)
plt.xlabel('Iteration')
plt.ylabel('Normalized profit')
plt.yscale('log')
plt.ylim((profit_log[0], max(profit_log) * 1.001))
plt.title('Profit during optimization')
plt.grid(True)
plt.show()

# TAC
plt.figure(figsize=(10, 5))
plt.scatter(range(len(tac_log)), tac_log, s = s, color=dark_tone)
plt.xlabel('Iteration')
plt.ylabel('TAC (EUR)')
plt.title('Total annualized costs during optimization')
plt.grid(True)
plt.show()

# Duties for the columns
Q_log = np.array(Q_log)
x = np.linspace(0, np.size(Q_log, 0) - 1, np.size(Q_log, 0))

plt.figure(figsize=(10, 5))
plt.scatter(x, Q_log[:, 0], color=dark_tone, s = s, label='Column 1')
plt.scatter(x, Q_log[:, 1], color=dark_tone_alt, s = s, label='Column 2')
plt.scatter(x, Q_log[:, 2], color=light_tone, s = s, label='Column 3')
plt.xlabel('Iteration')
plt.ylabel('Q [kW]')
plt.title('Column duties')
plt.grid(True)
plt.legend(loc=1)
plt.show()

flowsheet.display_graph()