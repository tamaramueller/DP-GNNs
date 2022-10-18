from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch
from torch_geometric.nn import global_mean_pool
import sys


class SyntheticDatasetModel(torch.nn.Module):
    def __init__(self, hidden_channels, dataset, model_type:str):
        super(SyntheticDatasetModel, self).__init__()
        self.model_type = model_type
        self.hidden_channels = hidden_channels
        
        if self.model_type == "GCN":
            self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels*2)
            self.conv3 = GCNConv(hidden_channels*2, hidden_channels*4)
            self.conv4 = GCNConv(hidden_channels*4, hidden_channels*8)
        elif self.model_type == "GraphSAGE":
            self.conv1 = SAGEConv(dataset.num_node_features, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels*2)
            self.conv3 = SAGEConv(hidden_channels*2, hidden_channels*4)
            self.conv4 = SAGEConv(hidden_channels*4, hidden_channels*8)
        elif self.model_type == "GAT":
            self.conv1 = GATConv(dataset.num_node_features, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels*2)
            self.conv3 = GATConv(hidden_channels*2, hidden_channels*4)
            self.conv4 = GATConv(hidden_channels*4, hidden_channels*8)
        else:
            print("ERROR: model must be GAT or GCN or GraphSAGE")
            sys.exit(-1)

        self.lin = torch.nn.Sequential(torch.nn.Linear(hidden_channels*8, 256), torch.nn.ReLU(), torch.nn.Linear(256, dataset.num_tasks))

    def forward(self, x, edge_index, batch):
        # 1. graph convolutions
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()        
        x = self.conv4(x, edge_index)
        x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)

        # 3. Apply a final classifier
        x = self.lin(x)
        
        return x