"""
hetero_gnn_model.py
Multi-Task Spatio-Temporal Heterogeneous Graph Neural Network (PyTorch Geometric).
Works with FP16 mixed precision on GPU (we trained on a Tesla V100).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

class MultiTaskHVACHeteroGNN(nn.Module):
    def __init__(self, hidden_channels=128, out_channels=64, num_layers=3, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.lin_dict = nn.ModuleDict({
            "resident": Linear(8, hidden_channels),
            "location": Linear(8, hidden_channels),
            "device": Linear(8, hidden_channels),
            "part": Linear(8, hidden_channels),
        })

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ("resident", "located_in", "location"): SAGEConv((-1, -1), hidden_channels),
                ("location", "has_resident", "resident"): SAGEConv((-1, -1), hidden_channels),
                ("resident", "owns", "device"): SAGEConv((-1, -1), hidden_channels),
                ("device", "owned_by", "resident"): SAGEConv((-1, -1), hidden_channels),
                ("device", "requires_part", "part"): SAGEConv((-1, -1), hidden_channels),
                ("part", "used_in", "device"): SAGEConv((-1, -1), hidden_channels),
            }, aggr="mean")
            self.convs.append(conv)

        # Model 1: Spatio-Temporal Service/Appliance Demand per Location
        self.head_demand = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )

        # Model 2A & 2B: Device Power Regressor & Failure Risk Classifier
        self.head_power = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )
        self.head_failure = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )

        # Model 3: Customer Retention / "Sleeper" Churn Classifier
        self.head_sleeper = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )

        # Model 4: Spare-Parts Inventory Forecasting Regressor
        self.head_inventory = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        h_dict = {
            node_type: F.leaky_relu(self.lin_dict[node_type](x))
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            h_next = conv(h_dict, edge_index_dict)
            h_dict = {
                key: F.dropout(F.leaky_relu(h_next[key]) + h_dict[key], p=self.dropout, training=self.training)
                for key in h_dict.keys()
            }

        return {
            "demand_pred": self.head_demand(h_dict["location"]),
            "power_pred": self.head_power(h_dict["device"]),
            "failure_logits": self.head_failure(h_dict["device"]),
            "sleeper_logits": self.head_sleeper(h_dict["resident"]),
            "inventory_pred": self.head_inventory(h_dict["part"]),
            "embeddings": h_dict
        }

    def predict_cross_sell_links(self, h_dict, resident_indices, device_indices):
        r_emb = h_dict["resident"][resident_indices]
        d_emb = h_dict["device"][device_indices]
        scores = (r_emb * d_emb).sum(dim=-1)
        return torch.sigmoid(scores)

