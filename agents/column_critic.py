import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class ColumnCritic(nn.Module):
    def __init__(self, input_dim=4, action_dim=3, num_classes=6,hidden_dim=128, lr=5e-4):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(input_dim + action_dim-1 + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Q-value
        )
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def forward(self, in_comp, action):
        """
        in_comp: tensor of shape (B, 4)
        action:  tensor of shape (B, 3) → [pair_index (scalar), recov_lk, recov_hk]
        """
        # Convert the first column (pair_index) to long for one-hot encoding
        pair_idx = action[:, 0].long()  # (B,)
        pair_onehot = F.one_hot(pair_idx, num_classes=6).float()  # (B, 6)

        # Keep the recovery values as float
        recovs = action[:, 1:]  # (B, 2)

        # Concatenate one-hot + recoveries
        a_concat = torch.cat([pair_onehot, recovs], dim=-1)  # (B, 8)
        x = torch.cat([in_comp, a_concat], dim=-1)           # (B, 12)

        q_val = self.q_net(x)  # (B, 1)
        return q_val.squeeze(-1)  # (B,)


    def train_step(self, in_comps, actions, q_targets, max_norm=1.0):
        q_preds = self.forward(in_comps, actions)
        loss = self.criterion(q_preds, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        self.optimizer.step()
        return loss.item()
