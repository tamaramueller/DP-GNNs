from torch_geometric.data import InMemoryDataset
from sklearn.model_selection import train_test_split
import torch


class MyOwnDataset(InMemoryDataset):
    def __init__(self, data_list, path):
        super(MyOwnDataset, self).__init__(path)

        self.num_graphs = len(data_list)
        self.num_tasks = 1

        X_train, X_test = train_test_split(range(self.num_graphs), test_size=0.4, random_state=1)
        X_test, X_val = train_test_split(X_test, test_size=0.25, random_state=1)

        self.train_mask = torch.zeros(self.num_graphs, dtype=torch.bool)
        self.test_mask = torch.zeros(self.num_graphs, dtype=torch.bool)
        self.val_mask = torch.zeros(self.num_graphs, dtype=torch.bool)
        self.train_mask[X_train] = True
        self.test_mask[X_test] = True
        self.val_mask[X_val] = True

        self.data, self.slices = self.collate(data_list)
