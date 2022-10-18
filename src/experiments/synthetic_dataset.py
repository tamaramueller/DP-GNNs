import os
import logging
import argparse
import torch

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
import utils.utils as utils
import utils.config as config
import utils.generate_synthetic_dataset as generate_synthetic_dataset
import models.synthetic_model as synthetic_model
import utils.train_utils as train_utils


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", "-lr", default=0.01, type=float, help="set learning rate")
    parser.add_argument("--batch_size", "-bs", type=int, default=64, help="set batch size")
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--dp", "-dp", type=int, default=0)
    parser.add_argument("--clip", "-clip", type=float, default=10)
    parser.add_argument("--epochs", "-e", default=10, type=int, help="set number of epochs")
    parser.add_argument("--model_type", "-model", default="GCN", type=str, help="define the graph conv. e.g.: GAT, GCN, GraphSAGE")
    parser.add_argument("--max_epsilon", "-max_eps", default=5, type=int, help="define the maximum epsilon")
    parser.add_argument("--hidden_channels", default=64, type=int, help="determine the number of hidden channels in the model")
    parser.add_argument("--nr_graphs", type=int, default=1000)
    parser.add_argument("--nodes_per_graph", default=20, type=int)
    parser.add_argument("--nr_node_features", default=10, type=int)
    parser.add_argument("--logging", default=1, type=int, help="set to 1 to log some infos, set to 0 to run without logging")
    args = parser.parse_args()

    # setting device and making execution deterministic
    cpu_generator, gpu_generator = utils.make_deterministic()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DP=True if args.dp==1 else False

    if args.logging==1: 
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        # logging.basicConfig(filename=os.path.join(config.log_path, "synthetic.log"), encoding='utf-8', level=logging.INFO)#, file_mode="w")
        logger.info(f"Started graph classificaiton on synthetic dataset...")
        logger.info(f"Using {device}")
        if DP:
            logger.info(f"Training will be differentially private with a max. privacy budget epsilon = {args.max_epsilon}.")

    train_loader, val_loader, test_loader, dataset = generate_synthetic_dataset.get_synthetic_dataloaders(args.nr_graphs, config.nr_classes, config.connectivity_list, args.nodes_per_graph, args.nr_node_features, config.means, config.std_devs, args.batch_size)
    model = synthetic_model.SyntheticDatasetModel(hidden_channels=args.hidden_channels, dataset=dataset, model_type=args.model_type).to(device)
    if args.logging==1: logger.info("dataloaders and model are ready.")

    # generate functorch model and set up DP environment
    fmodel, params, buffers, optimizer, criterion, train_loader = train_utils.set_up_train_environment(DP, model, int(dataset.train_mask.sum().item()), args.epochs, train_loader, args.clip, args.learning_rate, args.batch_size, args.max_epsilon)

    if args.logging==1: logger.info("Training started...")
    for epoch in range(1, args.epochs):
        if args.logging==1: logger.info(f"Epoch {epoch}")
        if DP:
            train_loss, train_acc, params = train_utils.train_dp(fmodel, params, buffers, train_loader, device, optimizer, criterion)
        else:
            train_loss, train_acc = train_utils.train(model, train_loader, optimizer, criterion, device)
        if args.logging==1: logger.info(f"        Train Loss: {train_loss}, train acc: {train_acc}")
        if DP:
            new_state_dict = {k:v for k,v in zip(model.state_dict().keys(), params)}
            model.load_state_dict(new_state_dict)

        val_acc, val_loss = train_utils.test(model, val_loader, criterion, device)
        test_acc, test_loss = train_utils.test(model, test_loader, criterion, device)

        if args.logging==1: logger.info(f'         Test Acc: {test_acc:.4f}, Val Acc: {val_acc:.4f}')

    if args.logging==1: logger.info("done")
