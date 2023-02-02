import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, DataLoader, InMemoryDataset


class MyFingerprintDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(MyFingerprintDataset, self).__init__('../dataset')

        self.num_graphs = len(data_list)
        self.num_tasks = 4

        X_train, X_test = train_test_split(range(self.num_graphs), test_size=0.4)
        X_test, X_val = train_test_split(X_test, test_size=0.25)

        self.train_mask = torch.zeros(self.num_graphs, dtype=torch.bool)
        self.test_mask = torch.zeros(self.num_graphs, dtype=torch.bool)
        self.val_mask = torch.zeros(self.num_graphs, dtype=torch.bool)
        self.train_mask[X_train] = True
        self.test_mask[X_test] = True
        self.val_mask[X_val] = True

        self.data, self.slices = self.collate(data_list)


def clean_fingerprint_data(dataset):

    value_map = {0:0, 2:1, 3:1, 4:2, 5:3, 6:1, 9:0, 10:1, 11:3}

    new_dataset_list = []
    for graph in dataset:
        if (graph.x.numel() != 0 and int(graph.y) in [0,2,3,4,5,6,9,10,11]):
            graph.y = torch.tensor([value_map[int(graph.y)]])
            new_dataset_list.append(graph)
            
    return new_dataset_list