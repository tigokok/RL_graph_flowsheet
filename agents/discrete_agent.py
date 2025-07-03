import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import NNConv
    
class SimpleGNN(nn.Module):
    def __init__(self, in_channels, out_channels, edge_attr_dim, hidden_channels):
        super().__init__()
        edge_nn = nn.Sequential(
            nn.Linear(edge_attr_dim, in_channels * hidden_channels),
            nn.ReLU()
        )
        self.conv = NNConv(in_channels, hidden_channels, edge_nn)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv(x, edge_index, edge_attr)
        x = self.norm(x)
        x = F.relu(x)
        x = self.lin(x)
        return x


class DiscreteAgent(nn.Module):
    def __init__(self, in_channels, out_channels, edge_attr_dim, hidden_dim, hidden_channels, 
                 num_actions, lr, depth = 2):
        super().__init__()
        self.num_actions = num_actions
        self.output_dim = self.num_actions

        self.gnn = SimpleGNN(in_channels, out_channels, edge_attr_dim, hidden_channels)
        self.q_head = self.make_mlp(in_dim=out_channels, out_dim=self.output_dim, hidden_dim=hidden_dim,
                                        depth=depth, final_activation=None)

        self.optimizer = Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.HuberLoss()

    def make_mlp(self, in_dim, out_dim, hidden_dim, depth, final_activation=None):
        layers = []
        if depth == 1:
            layers.append(nn.Linear(in_dim, out_dim))

        elif depth == 2:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, out_dim))

        else:
        
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(depth - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, out_dim))

        if final_activation:
            layers.append(final_activation)

        return nn.Sequential(*layers)

    def forward(self, data):
        x = self.gnn(data) 
        q_vals = self.q_head(x)  
        q_vals = q_vals.view(-1, self.num_actions)
        return q_vals

    def predict(self, data):
        self.eval()
        with torch.no_grad():
            return self.forward(data)

    def train_step(self, q_preds, q_targets, max_norm=1.0):
        loss = self.criterion(q_preds, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        self.optimizer.step()
        return loss.item()
