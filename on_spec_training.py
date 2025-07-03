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

# Training hyperparameters
training_stride = 10
train_amount = 5
learning_rate = 0.0005
gamma = 0.7
epochs = 8000
max_steps = 8

# Epsilon-greedy exploration parameters
epsilon_lower = 0.0
epsilon_upper = 0.90

# Replay buffer parameters
buffer_size = 250
batch_size = 50

# Running average window size for plotting
running_avg_window = 100


# ==================== Initialization ====================
# Saving
save = True
save_name = 'fixed_comp'

# Make gym
comp_mode = 'light'
node_types = ['feed', 'output', 'empty', 'ideal_dstwu']

feed_flow = 100
components = ['cyclo-pentane', 'n-hexane', 'n-heptane', 'n-octane']
feed_composition = np.array([0.4, 0.3, 0.2, 0.1])


gym = SpecGym(comp_mode='light', feed_flow=feed_flow,
              components=components,
              feed_composition=feed_composition,
              node_types=node_types)


# Transfer learning
# If true you can pick up previous weights and architecture and train
# i.e. if you first train on light fixed mixture (see: spec_gym.py)
# and then use the weights to train on random, performance increases
transfer_learning = False
if transfer_learning:
    arch_params = torch.load("checkpoints/agent_architecture_r8.pth")
    agent = DiscreteAgent(**arch_params)
    agent.load_state_dict(torch.load("checkpoints/agent_weights_r8.pth"))
    agent.to(device)

else:
    agent = DiscreteAgent(in_channels=in_channels, out_channels=out_channels,
                            edge_attr_dim=edge_attr_dim, hidden_dim=hidden_dim,
                            hidden_channels=hidden_channels,
                            num_actions=num_actions, lr=learning_rate,
                            depth=depth).to(device)

agent.train()


loss_fn = nn.HuberLoss()
reward_history = np.zeros(epochs)
loss_history = np.zeros(epochs)
best_reward = 0

# Epsilon decays from upper to lower over the first third of training, then stays constant
decay_steps = int(epochs * 0.75)
epsilon_decay = np.linspace(epsilon_upper, epsilon_lower, num=decay_steps)
epsilon = np.concatenate([epsilon_decay, np.full(epochs - decay_steps, epsilon_lower)])


replay_buffer = deque(maxlen=buffer_size)

loss = 0

# ==================== Training Loop ====================

for ep in tqdm(range(epochs)):
    '''
    Every episode reset the flowsheet and all necessary parameters
    '''

    flowsheet, info = gym.reset()

    total_reward = 0
    reward = [0, 0]
    done = False
    n_step = 0


    # Core flowsheeting loop. Place columns / outputs
    # until max_steps are done or there are no empty nodes left
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

        
        # epsilon greedy algorithm
        # if rand > ep -> pick agent output
        # else -> pick random decision
        if np.random.rand() > epsilon[ep]:
            a1 = int(a1)
            a2 = int(a2)

        else:
            a1 = int(torch.argmax(torch.rand(len(mask)) * mask))
            a2 = np.random.randint(0, num_actions)

        flowsheet, reward, done, info = gym.step([a1, a2])
        step_reward = reward

        total_reward += step_reward
        q_pred = q_vals[a1, a2]

        # Store necessary information from gym step in a buffer
        transition = {
            "fingerprint": fingerprint.detach(),
            "action": (int(a1), int(a2)),
            "reward": step_reward,
            "next_fp": FlowsheetEncoder(flowsheet, in_channels).detach() if not done else None,
            "done": done
        }
        replay_buffer.append(transition)


        n_step += 1

    # Reward history for now defined as the amount recovered
    try:
        reward_history[ep] = flowsheet.amount_on_spec(0.9)
    except:
        reward_history[ep] = 0

    # Training loop
    if ep % training_stride == 0 and len(replay_buffer) >= batch_size:
        for _ in range(train_amount):
            batch = random.sample(replay_buffer, k=batch_size)

            fps = [t["fingerprint"] for t in batch]
            actions = [t["action"] for t in batch]
            rewards = torch.tensor([t["reward"] for t in batch], dtype=torch.float32)
            dones = torch.tensor([t["done"] for t in batch], dtype=torch.bool)

            batch_graph = Batch.from_data_list(fps)
            q_vals = agent.forward(batch_graph)

            # Following code is a mess
            # it was difficult to train on batches as every training step
            # had a separate fingerprint for the graph which is a PyG data package
            # .. so normal indenting (doing it vectorized) didn't work out.

            # It works batch-wise now..
            # A future cleanup would be better here

            node_counts = [fp.num_nodes for fp in fps]
            node_offsets = [0] + list(accumulate(node_counts))

            q_preds = []
            for i, (a1, a2) in enumerate(actions):
                global_node_idx = node_offsets[i] + a1
                q_preds.append(q_vals[global_node_idx, a2])
            q_preds = torch.stack(q_preds)  

            q_targets = rewards.clone()

            not_done_indices = (~dones).nonzero(as_tuple=True)[0]
            if len(not_done_indices) > 0:
                next_fps = [batch[i]["next_fp"] for i in not_done_indices]
                next_batch_graph = Batch.from_data_list(next_fps)
                with torch.no_grad():
                    q_next = agent.forward(next_batch_graph)

                next_node_counts = [fp.num_nodes for fp in next_fps]
                next_node_offsets = [0] + list(accumulate(next_node_counts))

                q_max_next_list = []
                for idx in range(len(not_done_indices)):
                    nodes_in_graph = slice(next_node_offsets[idx], next_node_offsets[idx + 1])
                    q_next_graph = q_next[nodes_in_graph, :num_actions] 
                    q_max_next_list.append(q_next_graph.max()) 

                q_max_next = torch.stack(q_max_next_list)
                q_targets[not_done_indices] += gamma * q_max_next


            loss = agent.train_step(q_preds, q_targets)
        if ep % 100 == 0:
            print(f"Epoch {ep}, Buffer: {len(replay_buffer)}, Loss: {loss}")

    
    loss_history[ep] = loss

    if total_reward > best_reward:
        best_reward = total_reward
        winner_flowsheet = copy.deepcopy(flowsheet)


# ==================== Post Training ====================

# display latest flowsheet
flowsheet.display()
flowsheet.print_streams()

# display best flowsheet
winner_flowsheet.print_streams()
winner_flowsheet.display()

# Plot rewards running average
running_avg = np.convolve(reward_history, np.ones(running_avg_window) / running_avg_window, mode='valid')

plt.figure(figsize=(10, 5))
plt.scatter(range(len(reward_history)), reward_history, color=light_tone, label='Rewards', s=10)
plt.plot(range(running_avg_window - 1, len(reward_history)), running_avg, color=dark_tone, label=f'Running average (window={running_avg_window})')
plt.xlabel('Episode')
plt.ylabel('Product flow on spec [-]')
plt.title('Fraction of product on-spec during training')
plt.legend(loc=2)
plt.grid(True)
plt.show()

# Plot loss running average
running_avg = np.convolve(loss_history, np.ones(running_avg_window) / running_avg_window, mode='valid')

plt.figure(figsize=(10, 5))
plt.scatter(range(len(loss_history)), loss_history, color=light_tone, label='Loss', s=10)
plt.plot(range(running_avg_window - 1, len(loss_history)), running_avg, color=dark_tone, label=f'Running average (window={running_avg_window})')
plt.xlabel('Episode')
plt.ylabel('Huber loss')
plt.title('Loss of fraction of product on-spec during training')
plt.legend(loc=2)
plt.grid(True)
plt.show()



architecture_params = {
    "in_channels": in_channels,
    "out_channels": out_channels,
    "edge_attr_dim": edge_attr_dim,
    "hidden_dim": hidden_dim,
    "hidden_channels": hidden_channels,
    "num_actions": num_actions,
    "lr": learning_rate,
    "depth": depth
}

if save and save_name:
    torch.save(agent.state_dict(), f"checkpoints/agent_weights_{save_name}.pth")
    torch.save(architecture_params, f"checkpoints/agent_architecture_{save_name}.pth")
