import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import os
import argparse
import sys
import numpy as np
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
import utils.pyg_graph_classification as pyg_graph_classification
from utils.opacus_extractions import compute_dp_sgd_privacy
import models.fingerprints_models as models
import datasets.fingerprints_dataset as fingerprints_dataset


def main(train_loader, test_loader, model, optimizer, criterion, device="cuda", val_loader=None,  
            batch_size:int=64, epochs:int=10, clip:float=1, noise_multiplier:float=1,
            max_epsilon:float=1, DP:bool=False):

    max_roc_auc = 0

    if not DP:
        noise_multiplier = -1
        clip = -1

    for epoch in range(1, epochs+1):
        logging.info(f"Epoch: {epoch:03d}")
        if DP:

            if val_loader is not None:
                delta=1/(len(train_loader.dataset)+len(test_loader.dataset)+len(val_loader.dataset))
            else:
                delta=1/(len(train_loader.dataset)+len(test_loader.dataset))
            
            epsilon, alpha = compute_dp_sgd_privacy(
                    sample_rate=batch_size / len(train_loader.dataset),
                    noise_multiplier=noise_multiplier,
                    epochs=epoch + 1,
                    delta=delta
            )
            # stop training if privacy budget is exhausted
            if epsilon >= max_epsilon: 
                break
            train_loss = pyg_graph_classification.train_dp(model, train_loader, device, optimizer, criterion, clip_norm=clip, noise_multiplier=noise_multiplier)
        else:
            train_loss = pyg_graph_classification.train(model, train_loader, device, optimizer, criterion)
            
        logging.info(train_loss)

        with torch.no_grad():
            test_acc, test_roc_auc_val, test_f1, test_precision, test_recall, test_sensitivity, test_specificity = pyg_graph_classification.test_roc_auc(test_loader, model, device, nr_classes=4)
            train_acc, train_roc_auc, train_f1, train_precision, train_recall, train_sensitivity, train_specificity = pyg_graph_classification.test_roc_auc(train_loader, model, device, nr_classes=4)
            # val_acc, val_roc_auc, val_f1, val_precision, val_recall = pyg_graph_classification.test_roc_auc(val_loader, model, device, nr_classes=4)

        if test_roc_auc_val > max_roc_auc: 
            max_test_acc = test_acc
            max_test_f1 = test_f1
            max_test_roc_auc = test_roc_auc_val
            max_precision = test_precision
            max_recall = test_recall
            max_sensitivity = test_sensitivity
            max_specificity = test_specificity
        
        print(f"Test Acc: {test_acc}")
        print(f"Train Acc: {train_acc}")
        print(f"Test ROC AUC: {test_roc_auc_val}, test f1: {test_f1}")
        print(f"Train ROC AUC: {train_roc_auc}, train f1: {train_f1}")
        print(f"Test Recall: {test_recall}")
        print(f"Test Precision: {test_precision}")
        print(f"Test Sensitivity: {test_sensitivity}")
        print(f"Test Specificity: {test_specificity}")

    print("done")

    if DP:
        return model, max_test_acc, max_test_roc_auc, max_test_f1, max_precision, max_recall, max_sensitivity, max_specificity
    else:
        return model, test_acc, test_roc_auc_val, test_f1, test_precision, test_recall, test_sensitivity, test_specificity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", "-lr", default=0.2, type=float, help="set learning rate")
    parser.add_argument("--batch_size", "-bs", type=int, default=64, help="set batch size")
    parser.add_argument("--differentially_private", "-dp", type=bool, default=False)
    parser.add_argument("--noise_multiplier", "-noise", type=float, default=1, help="set noise multiplier")
    parser.add_argument("--clip", "-clip", type=float, default=1)
    parser.add_argument("--epochs", "-e", default=10, type=int, help="set number of epochs")
    parser.add_argument("--model_type", "-model", default="GCN", type=str, help="define the graph conv. e.g.: GAT, GCN, GraphSAGE")
    parser.add_argument("--max_epsilon", "-max_eps", default=10, type=float, help="define the maximum epsilon")
    parser.add_argument("-nr_runs", default=1, type=int, help="set number of runs that shall be executed")
    parser.add_argument("--logging", default=1, type=int)
    args = parser.parse_args()

    f1_scores = []
    test_accs = []
    test_roc_aucs = []
    test_precisions = []
    test_recalls = []
    test_sensitivities = []
    test_specificities = []

    if args.logging ==1:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.info(f"Started graph classificaiton on Molbace dataset...")

    for i in range(args.nr_runs):
        if args.logging == 1:
            logger.info("++++++++++++++++++")
            logger.info(f"RUN: {i+1}")
            logger.info("++++++++++++++++++")

        seed =  torch.randint(0, 100, [1])
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        hidden_channels = 256
        if args.logging==1: logger.info(f"Using DP? {args.differentially_private}") 

        dataset = TUDataset(root='datasets/TUDataset', name='Fingerprint', use_node_attr=True)
        new_dataset_list = fingerprints_dataset.clean_fingerprint_data(dataset)
        new_dataset = fingerprints_dataset.MyFingerprintDataset(new_dataset_list)

        model = models.FingerprintsModel(hidden_channels=hidden_channels, dataset=new_dataset, model_type=args.model_type).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

        train_loader = DataLoader(new_dataset[new_dataset.train_mask], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(new_dataset[new_dataset.val_mask], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(new_dataset[new_dataset.test_mask], batch_size=args.batch_size, shuffle=False)

        if args.differentially_private:
            criterion = torch.nn.CrossEntropyLoss(reduction="none")
        else:
            criterion = torch.nn.CrossEntropyLoss()

        trained_model, test_acc, test_roc_auc, test_f1, test_precision, test_recall, test_sensitivity, test_specificity = main(train_loader, test_loader, model, optimizer, criterion, val_loader=val_loader,
                device=device, batch_size=args.batch_size, epochs=args.epochs, noise_multiplier=args.noise_multiplier, 
                clip=args.clip, max_epsilon=args.max_epsilon, DP=args.differentially_private)

        f1_scores.append(test_f1)
        test_accs.append(test_acc)
        test_roc_aucs.append(test_roc_auc)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_sensitivities.append(test_sensitivity)
        test_specificities.append(test_specificity)

    if args.logging==1:
        logger.info("#######################")
        logger.info(f"Results of {args.nr_runs} runs with {args.model_type} epochs {args.epochs}, lr {args.learning_rate}, DP {args.differentially_private}, batch size: {args.batch_size}, noise {args.noise_multiplier}, clip {args.clip}, max epsilon {args.max_epsilon}")
        logger.info(f"Test Accs: {test_accs}; mean: {np.array(test_accs).mean()}, std: {np.array(test_accs).std()}")
        logger.info(f"Test ROC AUCs: {test_roc_aucs}; mean: {np.array(test_roc_aucs).mean()}, std: {np.array(test_roc_aucs).std()}")
        logger.info(f"Test F1s: {f1_scores}; mean: {np.array(f1_scores).mean()}, std: {np.array(f1_scores).std()}")
        logger.info(f"Test Precisions: {test_precisions}; mean: {np.array(test_precisions).mean()}, std: {np.array(test_precisions).std()}")
        logger.info(f"Test Recalls: {test_recalls}; mean: {np.array(test_recalls).mean()}, std: {np.array(test_recalls).std()}")
        logger.info(f"Test Sensitivities: {test_sensitivities}, mean: {np.array(test_sensitivities).mean()}, std: {np.array(test_sensitivities).std()}")
        logger.info(f"Test Specificities: {test_specificities}, mean: {np.array(test_specificities).mean()}, std: {np.array(test_specificities).std()}")
        logger.info("#######################")
