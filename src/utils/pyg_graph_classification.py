import torch
from torch_geometric.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import utils
import sklearn.metrics
from sklearn import metrics


def train(model, train_loader, device, optimizer, criterion, scheduler=None, epoch=None):
    model.train()
    epoch_losses = []

    for step, data in enumerate(tqdm(train_loader, desc="Iteration")):        
        optimizer.zero_grad()

        data = data.to(device)
        out = model(data.x.float(), data.edge_index, data.batch)
        is_labeled = data.y == data.y
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            loss = criterion(out.float(), data.y.long())
        else:
            loss = criterion(out.float()[is_labeled], data.y.float()[is_labeled].to(device))
        epoch_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if scheduler is not None: 
            scheduler.step()

    return np.mean(np.array(epoch_losses))


def train_dp(model, train_loader, device, optimizer, criterion, clip_norm=1e8, noise_multiplier=0., writer=None, scheduler=None, epoch=None):
    model.train()
    epoch_losses = []

    for param in model.parameters():
        if param.requires_grad:
            param.accumulated_gradients = []

    for step, data in enumerate(tqdm(train_loader, desc="Iteration")):        
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x.float(), data.edge_index, data.batch)
        is_labeled = data.y == data.y

        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            loss = criterion(out.float(), data.y.long())
        else:
            loss = criterion(out.float()[is_labeled], data.y.float()[is_labeled].to(device))

        for item in loss:
            item.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            with torch.no_grad():
                for param in model.parameters():
                    if hasattr(param, "accumulated_gradients") and param.grad is not None:
                        param.accumulated_gradients.append(param.grad.clone().detach())
                        param.grad.zero_()
        with torch.no_grad():
            for param in model.parameters():
                if hasattr(param, "accumulated_gradients") and param.grad is not None:
                    aggregated_gradient = torch.stack(param.accumulated_gradients, dim=0).sum(dim=0)
                    noise = torch.randn_like(param.grad)*(noise_multiplier)
                    param.grad.data = (aggregated_gradient+noise)/len(loss)
                    param.accumulated_gradients.clear()

        optimizer.step()
        epoch_losses.append(torch.mean(loss))
    
        if scheduler is not None:
            scheduler.step()
            if writer is not None: writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch+step)

    return torch.mean(torch.tensor(epoch_losses))


def test_my_metrics(loader, model, device):
    model.eval()
    epoch_predictions = []
    epoch_labels = []
    correct = 0

    for data in loader:  
        out = model(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device))  
        pred = out.argmax(dim=1) 
        is_labeled = (data.y == data.y).squeeze(-1)

        correct += int((pred[is_labeled] == data.y[is_labeled].squeeze(-1).to(device)).sum()) 
        epoch_predictions.append(pred.cpu())
        epoch_labels.append(data.y[is_labeled].squeeze())

    roc_auc = roc_auc_score(torch.cat(epoch_labels), torch.cat(epoch_predictions))

    return roc_auc


def test_roc_auc(loader, model, device, nr_classes=2):
    model.eval()
    epoch_predictions_indicators = []
    epoch_predictions = []
    epoch_labels = []
    epoch_label_indicators = []
    correct = 0

    for data in loader:  
        out = model(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device))  
        if nr_classes ==2:
            pred = torch.round(torch.sigmoid(out))
        else:
            pred = out.argmax(dim=1) 

        is_labeled = (data.y == data.y).squeeze(-1)

        correct += int((pred[is_labeled] == data.y[is_labeled].squeeze(-1).to(device)).sum()) 

        predictions_indicators = [utils.convert_label_to_indicator_array(int(i), nr_classes) for i in pred.cpu()[is_labeled]]
        epoch_predictions_indicators.append(np.array(predictions_indicators).squeeze())
        epoch_predictions.append(pred.squeeze())
        indicator_labels_batch_nodes = [utils.convert_label_to_indicator_array(int(i), nr_classes) for i in data.y[is_labeled]]
        epoch_label_indicators.append(np.array(indicator_labels_batch_nodes).squeeze())
        epoch_labels.append(data.y[is_labeled].squeeze())

    flatten_epoch_labels = torch.tensor([item.tolist() for subl in epoch_label_indicators for item in subl])
    flatten_epoch_preds = torch.tensor([item.tolist() for subl in epoch_predictions_indicators for item in subl])
    
    roc_auc = sklearn.metrics.roc_auc_score(np.array(flatten_epoch_labels), np.array(flatten_epoch_preds), average='micro')
    f1_score = sklearn.metrics.f1_score(np.array(flatten_epoch_labels), np.array(flatten_epoch_preds), average='micro')
    accuracy_score = sklearn.metrics.accuracy_score(np.array(flatten_epoch_labels), np.array(flatten_epoch_preds))

    classification_report = sklearn.metrics.classification_report(np.array(flatten_epoch_labels), np.array(flatten_epoch_preds), output_dict=True)
    weighted_avg = classification_report['weighted avg']
    precision = weighted_avg['precision']
    recall = weighted_avg['recall']

    if nr_classes > 2:
        sensitivities = []
        specificities = []
        multi_label_confusion_matrix = sklearn.metrics.multilabel_confusion_matrix(np.array(flatten_epoch_labels), np.array(flatten_epoch_preds))
        for i in range(multi_label_confusion_matrix.shape[0]):
            TP = multi_label_confusion_matrix[i][1][1]
            TN = multi_label_confusion_matrix[i][0][0]
            FP = multi_label_confusion_matrix[i][0][1]
            FN = multi_label_confusion_matrix[i][1][0]

            sensitivities.append(TP / float(TP + FN))
            specificities.append(TN / float(TN+FP))


    return accuracy_score, roc_auc, f1_score, precision, recall, np.array(sensitivities).mean(), np.array(specificities).mean()


def test(loader, model, device):
    model.eval()
    correct = 0

    for data in loader:  
        out = model(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device))  
        pred = out.argmax(dim=1) 
        correct += int((pred == data.y.to(device)).sum()) 
    accuracy = correct / len(loader.dataset)

    return accuracy


def test_binary_classification(loader, model, device):
    model.eval()
    correct=0
    outs = []
    correct_ys = []

    for data in loader:
        out = model(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device))
        outs.append(out.cpu())
        correct_ys.append(data.y)

    outputs = torch.cat(outs)
    true_ys = torch.cat(correct_ys)

    y_pred_tag = torch.round(torch.sigmoid(outputs))
    correct_results_sum = (y_pred_tag == true_ys).sum().float()
    acc = correct_results_sum/len(true_ys)

    confusion_matrix = metrics.confusion_matrix(true_ys, y_pred_tag)

    TP = confusion_matrix[1][1]
    TN = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]

    sensitivity = (TP / float(TP + FN))
    specificity = (TN / float(TN+FP))
    precision = (TN / float(TN + FP))
    f1 = 2*((precision*sensitivity)/ (precision + sensitivity))

    return acc, sensitivity, specificity, precision, f1, true_ys        


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    binary_pred = []
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        is_labeled = (batch.y == batch.y).squeeze(-1)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch.x.float(), batch.edge_index, batch.batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
            binary_pred.append(torch.round(torch.sigmoid(pred).cpu()))

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    evaluator_output = evaluator.eval(input_dict)

    return evaluator_output


def prepare_ogb_data(dataset_name:str, batch_size:int):

    dataset = PygGraphPropPredDataset(name = dataset_name) 
    split_idx = dataset.get_idx_split() 

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)

    evaluator = Evaluator(dataset_name)

    return train_loader, val_loader, test_loader, dataset, evaluator

