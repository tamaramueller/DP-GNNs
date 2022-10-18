import torch_geometric
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import datasets.synthetic_dataset as synthetic_dataset


def generate_dataset(nr_graphs:int, 
                     nr_classes:int, 
                     connectivity:list, 
                     nodes_per_graph:int, 
                     nr_node_features:int, 
                     means:list, 
                     std_devs:list):
    """
    @params:
        nr_graphs: int: number of graphs in dataset
        nr_classes: int: number of classes in the dataset
        connectivity: list: specify which probability of connectivity each class should have
            e.g. for a binary classification task the connectivity list could look like this: [0.1, 0.6] to produce 
                 very dense graphs for one class and very sparse ones for the other
        nodes_per_graph: int: specifies the number of nodes a graph should contain
        nr_node_features: int: specifies the nunmber of node features per node
        means: list: list of mean values for Gaussian distributions from which node features of the different classes will be sampled
        std_devs: list: list of standard deviations for Gaussian distribution from which node featuers of the different classes will be sampled
    """

    nr_graphs_per_class = int(nr_graphs/nr_classes)
    my_data = []

    for c in tqdm(range(nr_classes)):
        for i in tqdm(range(nr_graphs_per_class)):
            edges = torch_geometric.utils.erdos_renyi_graph(nodes_per_graph, connectivity[c])
            node_features = torch.normal(means[c], std_devs[c], size=(nodes_per_graph, nr_node_features))

            graph = Data(x=node_features, edge_index=edges, y=torch.tensor([[c]]))
            my_data.append(graph)

    return my_data


def get_synthetic_dataloaders(num_graphs:int,
                              nr_classes:int, 
                              connectivity_list:list, 
                              nodes_per_graph:int, 
                              nr_node_features:int, 
                              means:list, 
                              std_devs:list,
                              batch_size:int):
    """
    @params:
        nr_graphs: int: number of graphs in dataset
        nr_classes: int: number of classes in the dataset
        connectivity: list: specify which probability of connectivity each class should have
            e.g. for a binary classification task the connectivity list could look like this: [0.1, 0.6] to produce 
                 very dense graphs for one class and very sparse ones for the other
        nodes_per_graph: int: specifies the number of nodes a graph should contain
        nr_node_features: int: specifies the nunmber of node features per node
        means: list: list of mean values for Gaussian distributions from which node features of the different classes will be sampled
        std_devs: list: list of standard deviations for Gaussian distribution from which node featuers of the different classes will be sampled
        batch_size: int
    """

    mydata = generate_dataset(num_graphs, nr_classes, connectivity_list, nodes_per_graph, nr_node_features, means, std_devs)
    dataset = synthetic_dataset.MyOwnDataset(mydata)
    dataset.num_tasks =  nr_classes

    train_dataset = dataset[dataset.train_mask]
    test_dataset = dataset[dataset.test_mask]
    val_dataset = dataset[dataset.val_mask]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, dataset