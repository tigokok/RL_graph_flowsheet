import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import NNConv
  
class ColumnDiscreteAgent(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, lr):
        super().__init__()

        self.q_head = self.make_mlp(in_dim=in_dim, out_dim=out_dim.output_dim, hidden_dim=hidden_dim,
                                        depth=depth, final_activation=None)

        self.optimizer = Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.HuberLoss()

    def make_mlp(self, in_dim, out_dim, hidden_dims, final_activation=torch.sigmoid()):
        layers = []

        layers.append(nn.Linear(in_dim, hidden_dims[0]))
        layers.append(nn.ReLU())

        if len(hidden_dims) > 1:
            for i in range(len(hidden_dims)):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                layers.append(nn.ReLU())
                
        layers.append(nn.Linear(hidden_dims[-1], out_dim))
        if final_activation: 
            layers.append(final_activation)

        return nn.Sequential(*layers)

    def forward(self, x):
        q_vals = self.q_head(x)  
        q_vals = q_vals.view(-1, self.num_actions)
        return q_vals

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def train_step(self, q_preds, q_targets, max_norm=1.0):
        loss = self.criterion(q_preds, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        self.optimizer.step()
        return loss.item()
