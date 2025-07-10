import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.nn.init as init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ColumnActor(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, lr = 5e-4, alpha = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8)  # [6 logits for (lk, hk), 2 for recoveries]
        )

        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -torch.log(torch.tensor(1.0 / 6.0))  # for 6 discrete pairs


        # Initialize weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                init.zeros_(m.bias)

        # valid (lk, hk) pairs
        self.pair_indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        self.index_to_pair_map = {i: pair for i, pair in enumerate(self.pair_indices)}
        self.pair_to_index_map = {pair: i for i, pair in enumerate(self.pair_indices)}

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.alpha = alpha

    def forward(self, in_comp):
        out = self.net(in_comp)  # shape: (B, 8)
        pair_logits = out[..., :6]          # logits over 6 discrete (lk, hk) pairs
        recovs_raw = out[..., 6:]           # raw values for recoveries

        # Sample or pick max from categorical (use Gumbel-Softmax for differentiability)
        pair_dist = F.softmax(pair_logits, dim=-1)
        pair_index = torch.argmax(pair_dist, dim=-1)  # (B,)
        
        # Recoveries are sigmoid-constrained between 0–1
        recov_lk = 0.75 + 0.24 * torch.sigmoid(recovs_raw[..., 0])  # → [0.75, 0.99]
        recov_hk = 0.01 + 0.09 * torch.sigmoid(recovs_raw[..., 1])  # → [0.01, 0.10]

        
        recov_vals = torch.stack([recov_lk, recov_hk], dim=-1)  # (B, 2)
        
        return pair_index, recov_vals  # used during act + training
    
    def sample(self, in_comp):
        out = self.net(in_comp)  # (B, 8)
        pair_logits = out[..., :6]
        recovs_raw = out[..., 6:]

        # Create categorical distribution over discrete actions
        pair_dist = torch.distributions.Categorical(logits=pair_logits)

        # Sample discrete pair index
        pair_index = pair_dist.sample()  # (B,)

        # Calculate log prob of sampled pair
        pair_log_prob = pair_dist.log_prob(pair_index)  # (B,)

        # For recoveries, model continuous outputs with e.g. a Gaussian
        # Here, just sigmoid raw outputs as deterministic values (you could model uncertainty better)
        # Recoveries are sigmoid-constrained between 0–1
        recov_lk = 0.80 + 0.19 * torch.sigmoid(recovs_raw[..., 0])  # → [0.85, 0.99]
        recov_hk = 0.01 + 0.09 * torch.sigmoid(recovs_raw[..., 1])  # → [0.01, 0.10]

        
        recov_vals = torch.stack([recov_lk, recov_hk], dim=-1)  # (B, 2)

        # Total log prob is log prob of discrete pair plus (optionally) log prob of continuous
        # If treating continuous outputs deterministically, omit continuous log prob

        # Return combined action tensor: (pair_index, recov_lk, recov_hk)
        actions = torch.cat([pair_index.unsqueeze(-1).float(), recov_vals], dim=-1)  # (B,3)

        return actions, pair_log_prob


    def index_to_pair(self, idx):
        return self.index_to_pair_map[int(idx)]

    def pair_to_index(self, lk, hk):
        return self.pair

    def train_step(self, in_comps, critics, max_norm=1.0):
        # critics is a list or tuple of critic instances
        new_actions, log_probs = self.sample(in_comps)
        q1_pi = critics[0](in_comps, new_actions)
        q2_pi = critics[1](in_comps, new_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_probs - q_pi).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        self.optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()


        return actor_loss.item()