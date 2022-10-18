import numpy as np
import torch
from tqdm import tqdm
from functorch import jacrev
import torch_geometric
from functorch import make_functional_with_buffers
from opacus.accountants.utils import get_noise_multiplier
from opacus.optimizers import DPOptimizer
from opacus.utils.batch_memory_manager import wrap_data_loader


def train(model, train_loader, optimizer, criterion, device):
    model.train()

    correct = 0
    epoch_loss = 0
    for data in tqdm(train_loader):
        optimizer.zero_grad() 
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y.squeeze())  
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred = out.cpu().argmax(dim=1)
        correct += int((pred == data.y.squeeze().cpu()).sum())  

    return epoch_loss/len(train_loader), correct/len(train_loader.dataset)


def compute_loss(params, buffers, data_x, data_edge_index, data_batch, targets, fmodel, loss_fn):
    predictions = fmodel(params, buffers, data_x, data_edge_index, data_batch)
    loss = loss_fn(predictions.squeeze(), targets.squeeze())
    return loss


# functorch implementation of per sample gradients needed for DP
compute_per_sample_grads = jacrev(compute_loss)


def train_dp(fmodel, params, buffers, train_loader, device, optimizer, criterion, scheduler=None):
    epoch_losses = []
    correct = 0

    for step, data in enumerate(tqdm(train_loader, desc="Iteration")):        
        optimizer.zero_grad(True)
        data = data.to(device)
        out = fmodel(params, buffers, data.x.float(), data.edge_index, data.batch)
        pred = out.cpu().argmax(dim=1)
        correct += int((pred == data.y.squeeze().cpu()).sum())  

        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            loss = criterion(out.squeeze(), data.y.squeeze())
        else:
            loss = criterion(out.squeeze(), data.y.squeeze().float().to(device))

        per_sample_grads = compute_per_sample_grads(
            params,
            buffers,
            data.x,
            data.edge_index,
            data.batch,
            data.y,
            fmodel,
            criterion,
        )

        for param, grad_sample in zip(params, per_sample_grads):
            param.grad_sample = grad_sample
            param.grad = (grad_sample.mean(0))

        optimizer.step()
        epoch_losses.append(torch.mean(loss.detach().cpu()))

        if scheduler is not None:
            scheduler.step()

    acc = correct/len(train_loader.dataset)
    return np.mean(epoch_losses), acc, params


def test(model, test_loader, criterion, device):
    model.eval()
    epoch_losses = []

    correct = 0
    for data in tqdm(test_loader):
        data = data.to(device)
        y = data.y.squeeze()
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.cpu().argmax(dim=1)
        correct += int((pred == data.y.squeeze().cpu()).sum())  
        loss = criterion(out.squeeze(), data.y.squeeze())  
    
    epoch_losses.append(torch.mean(loss.detach().cpu()))
   
    return correct/len(test_loader.dataset) , np.mean(epoch_losses)#, f1_score, roc_auc, specificity, sensitivity


def set_up_train_environment(dp:bool, 
                             model:torch.nn.Module, 
                             nr_train_samples:int, 
                             epochs:int, 
                             train_loader:torch_geometric.loader.DataLoader, 
                             clip:float, 
                             learning_rate:float, 
                             batch_size:int, 
                             max_epsilon:float=None):

    fmodel, params, buffers = make_functional_with_buffers(model)

    if dp:
        optimizer = torch.optim.SGD(params, lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        NOISE = get_noise_multiplier(target_epsilon=max_epsilon, target_delta=1/nr_train_samples, sample_rate=1/len(train_loader), epochs=epochs)
        optimizer = DPOptimizer(
                            optimizer,
                            noise_multiplier=NOISE,
                            max_grad_norm=clip,
                            expected_batch_size=batch_size,
                            loss_reduction="mean",
                        )
        train_loader = wrap_data_loader(data_loader=train_loader, max_batch_size=batch_size, optimizer=optimizer)
        torch.set_grad_enabled(False)
        return fmodel, params, buffers, optimizer, criterion, train_loader
    else:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
        torch.set_grad_enabled(True)

        return model, params, buffers, optimizer, criterion, train_loader