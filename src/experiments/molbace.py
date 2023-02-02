from sklearn.model_selection import train_test_split
import numpy as np
from torch_geometric.data import DataLoader
import torch
import sys
import os
import argparse
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from tqdm import tqdm
import wandb
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
import utils.pyg_graph_classification as pyg_graph_classification
from utils.opacus_extractions import compute_dp_sgd_privacy
import models.molbace_models as models


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch.x.float(), batch.edge_index, batch.batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def run_main(train_loader, test_loader, model, optimizer, criterion, device="cuda", val_loader=None, scheduler=None, 
            writer=None, batch_size:int=64, epochs:int=10, clip:float=1, noise_multiplier:float=1,
            max_epsilon:float=1, DP:bool=False):
            
    max_test_acc = 0
    max_sensitivity=0
    max_specificity=0
    max_roc_auc = 0
    max_f1 = 0

    logger.info(f"Using DP? {DP}")
    if not DP:
        noise_multiplier = -1
        clip = -1

    for epoch in range(1, epochs+1):
        logger.info(f"Epoch: {epoch:03d}")

        if DP:
            # calculate epsilon and alpha
            epsilon, alpha = compute_dp_sgd_privacy(
                    sample_rate=batch_size / len(train_loader.dataset),
                    noise_multiplier=noise_multiplier,
                    epochs=epoch,
                    delta=1/(len(train_loader.dataset)+len(test_loader.dataset))
            )
            if epsilon >= max_epsilon: 
                break
            train_loss = pyg_graph_classification.train_dp(model, train_loader, device, optimizer, criterion, clip_norm=clip, noise_multiplier=noise_multiplier)
        else:
            train_loss = pyg_graph_classification.train(model, train_loader, device, optimizer, criterion, scheduler=scheduler, writer=writer, epoch=epoch)
            
        logger.info(f"train loss: {train_loss}")

        with torch.no_grad():
            test_acc, test_sensitivity, test_specificity, test_precision, test_f1, true_y_test = pyg_graph_classification.test_binary_classification(test_loader, model, device)
            train_acc, train_sensitivity, train_specificity, train_precision, train_f1, true_y_train = pyg_graph_classification.test_binary_classification(train_loader, model, device)
            if val_loader is not None: val_acc, val_sensitivity, val_specificity, val_precision, val_f1, true_y_val = pyg_graph_classification.test_binary_classification(val_loader, model, device)
            test_perf = eval(model, device, test_loader, evaluator)
            train_perf = eval(model, device, train_loader, evaluator)

        if test_perf['rocauc']>max_roc_auc: 
            max_test_acc = test_acc
            max_sensitivity=test_sensitivity
            max_specificity=test_specificity
            max_f1 = test_f1
            max_roc_auc = test_perf['rocauc']

        logger.info(f"Test Acc: {test_acc}")
        logger.info(f"Train Acc: {train_acc}")
        logger.info(f"Test ROC AUC: {test_perf['rocauc']}")
        if val_loader is not None: logger.info(f"Val Acc: {val_acc}")
        logger.info(f"Test sensitiviy: {test_sensitivity}, specificity: {test_specificity}, precision: {test_precision}, f1: {test_f1}")
        logger.info(f"Train sensitivity: {train_sensitivity}, specificity: {train_specificity}, precision: {train_precision}, f1: {train_f1}")

    logger.info("done.")

    if DP:
        return model, max_test_acc, max_sensitivity, max_specificity, max_f1
    else:
        return model, test_acc, test_sensitivity, test_specificity, test_f1


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", "-lr", default=0.01, type=float, help="set learning rate")
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--batch_size", "-bs", type=int, default=24, help="set batch size")
    parser.add_argument("--differentially_private", "-dp", type=bool, default=False)
    parser.add_argument("--noise_multiplier", "-noise", type=float, default=0.5, help="set noise multiplier")
    parser.add_argument("--clip", "-clip", type=float, default=5)
    parser.add_argument("--epochs", "-e", default=50, type=int, help="set number of epochs")
    parser.add_argument("--model_type", "-model", default="GCN", type=str, help="define the graph conv. e.g.: GAT, GCN, GraphSAGE")
    parser.add_argument("--max_epsilon", "-max_eps", default=20, type=int, help="define the maximum epsilon")
    parser.add_argument("-nr_runs", default=1, type=int, help="set number of runs that shall be executed")
    parser.add_argument("--logging", default=1, type=int)
    args = parser.parse_args()

    f1_scores = []
    test_accs = []
    test_sensitivities = []
    test_specificities = []
    test_roc_aucs = []

    if args.logging==1: 
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.info(f"Started graph classificaiton on Molbace dataset...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dp_string = "DP_" if args.differentially_private else ""

    for i in range(args.nr_runs):
        if args.logging==1:
            logger.info("++++++++++++++++++")
            logger.info(f"RUN: {i+1}")
            logger.info("++++++++++++++++++")

        seed = torch.randint(0, 100, [1])
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # load data and loaders
        d_name = "ogbg-molbace"
        dataset = PygGraphPropPredDataset(name = d_name) 
        split_idx = dataset.get_idx_split() 
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)
        evaluator = Evaluator(d_name)

        if not args.differentially_private:
            noise_multiplier = -1
            clip = -1

        model = models.MolbaceModel(hidden_channels=args.hidden_channels, dataset=dataset, model_type=args.model_type).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

        if args.differentially_private:
            criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.learning_rate, max_lr=args.learning_rate*10, step_size_up=200, step_size_down=200, gamma=0.99)
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
            scheduler = None

        trained_model, test_acc, test_sensitivity, test_specificity, test_f1 = run_main(train_loader, test_loader, model, optimizer, criterion, 
                device=device, batch_size=args.batch_size, epochs=args.epochs, noise_multiplier=args.noise_multiplier, scheduler=scheduler,
                clip=args.clip, max_epsilon=args.max_epsilon, DP=args.differentially_private)
        with torch.no_grad():
            test_perf = eval(trained_model, device, test_loader, evaluator)
            test_acc, test_sensitivity, test_specificity, test_precision, test_f1, true_y_test = pyg_graph_classification.test_binary_classification(test_loader, model, device)

        logger.info("===================")
        logger.info(test_perf)
        logger.info(test_acc, test_sensitivity, test_specificity, test_f1)
        logger.info("====================")

        f1_scores.append(test_f1)
        test_accs.append(test_acc)
        test_sensitivities.append(test_sensitivity)
        test_specificities.append(test_specificity)
        test_roc_aucs.append(test_perf['rocauc'])

    logger.info("#######################")
    logger.info(f"Results of {args.nr_runs} runs with {args.model_type} lr {args.learning_rate}, DP {args.differentially_private}, batch size: {args.batch_size}, noise {args.noise_multiplier}, clip {args.clip}, max epsilon {args.max_epsilon}")
    logger.info(f"Test Accs: {test_accs}; mean: {np.array(test_accs).mean()}, std: {np.array(test_accs).std()}")
    logger.info(f"Test Sensitivities: {test_sensitivities}; mean: {np.array(test_sensitivities).mean()}, std: {np.array(test_sensitivities).std()}")
    logger.info(f"Test F1s: {f1_scores}; mean: {np.array(f1_scores).mean()}, std: {np.array(f1_scores).std()}")
    logger.info(f"Test Specificities: {test_specificities}; mean: {np.array(test_specificities).mean()}, std: {np.array(test_specificities).std()}")
    logger.info(f"Test ROC AUCs: {test_roc_aucs}; mean: {np.array(test_roc_aucs).mean()}, std: {np.array(test_roc_aucs).std()}")
    logger.info("#######################")
