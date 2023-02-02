from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch
import torch_geometric
from torch_geometric.nn import global_mean_pool


class MolbaceModel(torch.nn.Module):
    def __init__(self, hidden_channels, dataset, model_type:str):
        super(MolbaceModel, self).__init__()
        torch.manual_seed(12345)
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

        self.lin = torch.nn.Sequential(torch.nn.Linear(hidden_channels*4, 512), torch.nn.ReLU(), torch.nn.Linear(512, dataset.num_tasks))
        self.norm = torch_geometric.nn.InstanceNorm(1, affine=True)

    def forward(self, x, edge_index, batch):
        x = self.norm(x, batch)
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = self.norm(x, batch)
        x = x.relu()
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.norm(x, batch)
        x = x.relu()
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.norm(x, batch)
        x = x.relu()        
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = self.conv4(x, edge_index)
        # x = self.norm(x, batch)
        # x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin(x)
        
        return x