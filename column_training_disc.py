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
from gym.column_gym import ColumnGym
from agents.discrete_agent import DiscreteAgent
from agents.column_discrete import ColumnDiscreteAgent
from nodeFlowsheeter.flowsheetEncoder import FlowsheetEncoder
import torch.nn.functional as F

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
training_stride = 1
train_amount = 1
learning_rate = 0.0005
gamma = 0.85
epochs = 4_000
max_steps = 8

# Epsilon-greedy exploration parameters
epsilon_lower = 0
epsilon_upper = 0.9

# Replay buffer parameters
buffer_size = 500
batch_size = 32

# Running average window size for plotting
running_avg_window = 100




# ==================== Initialization ====================
# Saving
save = False
save_name = 'fixed_comp'

# Make gym
comp_mode = 'random'
node_types = ['feed', 'output', 'empty', 'dstwu']

feed_flow = 100
components = ['cyclo-pentane', 'n-hexane', 'n-heptane', 'n-octane']
feed_composition = np.array([0.4, 0.3, 0.2, 0.1])


gym = ColumnGym(comp_mode=comp_mode, feed_flow=feed_flow,
              components=components,
              feed_composition=feed_composition,
              node_types=node_types)


# Column agent
key_pairs = 6
lk_modes = 3
hk_modes = 3
out_dim = key_pairs * lk_modes * hk_modes
column_agent = ColumnDiscreteAgent(in_dim=4, out_dim=out_dim, hidden_dims=64).to(device)

# Set to train mode
column_agent.train()


# ------ LOAD DECISON AGENT -------
force_good_decision = True

arch_params = torch.load("checkpoints/decision_architecture_r8.pth")
decision_agent = DiscreteAgent(**arch_params)
decision_agent.load_state_dict(torch.load("checkpoints/decision_weights_r8.pth"))
decision_agent.to(device)
decision_agent.eval()

loss_fn = nn.HuberLoss()
spec_history = np.zeros(epochs)
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

        if not force_good_decision:
            # Make decision using decision agent
            decision_q_vals = decision_agent.forward(fingerprint) 
            decision_masked_q_vals = decision_q_vals.clone()
            decision_masked_q_vals[~mask.bool()] = -1e9
            flat_index = decision_masked_q_vals.argmax()
            node_id, action_id = np.unravel_index(flat_index, decision_q_vals.shape)

        else:
            valid_indices = torch.nonzero(mask, as_tuple=False).flatten()
            idx = torch.randint(0, len(valid_indices), (1,))
            node_id = valid_indices[idx].item()
     
            incoming_stream = gym.sim.get_incoming_streams(node_id)[0]
            in_comp = incoming_stream['composition']

            if max(in_comp) > gym.spec: action_id = 0
            else: action_id = 1
        
        # Parameter agent
        # Only train if placing a column
        if action_id == 1:  # -> column
            incoming_stream = gym.sim.get_incoming_streams(node_id)[0]
            in_comp = incoming_stream['composition']
            in_comp = torch.tensor(in_comp, dtype=torch.float32).to(device)

            # Get action
            pair, recov_lk_id, recov_hk_id = column_agent.forward(in_comp)

            # convert to lk, hk, recov_lk, recov_hk
            

        flowsheet, reward, done, info = gym.step([node_id, action_id, lk, hk, recov_lk, recov_hk])
        step_reward = reward

        total_reward += step_reward

        # Store necessary information from gym step in a buffer
        # Only do so if a column was placed!
        if action_id == 1:
            transition = {
                "in_comp": in_comp,
                "action": (pair_idx, recov_lk, recov_hk),
                "reward": step_reward,
                "done": done
            }
            replay_buffer.append(transition)


        n_step += 1

    reward_history[ep] = total_reward
    # Reward history for now defined as the amount recovered
    try:
        spec_history[ep] = flowsheet.amount_on_spec(0.9)
    except:
        spec_history[ep] = 0

    # Training loop
    if ep % training_stride == 0 and len(replay_buffer) >= batch_size:
        for _ in range(train_amount):
            batch = random.sample(replay_buffer, k=batch_size)

            in_comps = torch.stack([t["in_comp"] for t in batch]).to(device)  # shape: (B, 4)
            actions = torch.stack([torch.tensor(t["action"], dtype=torch.float32) for t in batch]).to(device)
            rewards = torch.tensor([t["reward"] for t in batch], dtype=torch.float32, device=device)
            dones = torch.tensor([t["done"] for t in batch], dtype=torch.float32, device=device)

            # --- Critic Update ---
            with torch.no_grad():
                # Sample next action from actor
                next_actions, log_probs = column_actor.sample(in_comps)  # log_probs: (B,)

                target_q1 = target_column_critic_1(in_comps, next_actions)
                target_q2 = target_column_critic_2(in_comps, next_actions)
                target_q = torch.min(target_q1, target_q2) - alpha * log_probs  # SAC objective

                q_target = rewards + gamma * (1 - dones) * target_q  # shape: (B,)

            # Current Q estimates
            q1 = column_critic_1(in_comps, actions)
            q2 = column_critic_2(in_comps, actions)

            # Critic updates
            critic1_loss = column_critic_1.train_step(in_comps, actions, q_target)
            critic2_loss = column_critic_2.train_step(in_comps, actions, q_target)

            # Actor update
            actor_loss = column_actor.train_step(in_comps, critics=[column_critic_1, column_critic_2])

            # --- Target network update ---
            with torch.no_grad():
                for param, target_param in zip(column_critic_1.parameters(), target_column_critic_1.parameters()):
                    target_param.data.mul_(1 - tau)
                    target_param.data.add_(tau * param.data)
                for param, target_param in zip(column_critic_2.parameters(), target_column_critic_2.parameters()):
                    target_param.data.mul_(1 - tau)
                    target_param.data.add_(tau * param.data)

        if ep % 100 == 0:
            print(f"Epoch {ep}, Buffer: {len(replay_buffer)}, Actor Loss: {actor_loss:.4f}, Critic1 Loss: {critic1_loss:.4f}, Critic2 Loss: {critic2_loss:.4f}")


    loss_history[ep] = loss

    if total_reward > best_reward:
        best_reward = total_reward
        winner_flowsheet = copy.deepcopy(flowsheet)


# ==================== Post Training ====================

# display latest flowsheet
flowsheet.display()
flowsheet.print_streams()

# display best flowsheet

#winner_flowsheet.print_streams()
#winner_flowsheet.display()

# Plot spec running average
running_avg = np.convolve(spec_history, np.ones(running_avg_window) / running_avg_window, mode='valid')

plt.figure(figsize=(10, 5))
plt.scatter(range(len(spec_history)), spec_history, color=light_tone, label='Rewards', s=10)
plt.plot(range(running_avg_window - 1, len(spec_history)), running_avg, color=dark_tone, label=f'Running average (window={running_avg_window})')
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

# Plot reward running average
running_avg = np.convolve(reward_history, np.ones(running_avg_window) / running_avg_window, mode='valid')

plt.figure(figsize=(10, 5))
plt.scatter(range(len(reward_history)), reward_history, color=light_tone, label='Reward', s=10)
plt.plot(range(running_avg_window - 1, len(reward_history)), running_avg, color=dark_tone, label=f'Running average (window={running_avg_window})')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Total reward during training')
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


'''
if save and save_name:
    torch.save(agent.state_dict(), f"checkpoints/agent_weights_{save_name}.pth")
    torch.save(architecture_params, f"checkpoints/agent_architecture_{save_name}.pth")
'''