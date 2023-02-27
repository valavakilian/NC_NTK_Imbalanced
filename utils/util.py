import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import copy
import argparse


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def client_update(client_model, optimizer, train_loader, epoch=5):
    """Train a client_model on the train_loder data."""
    client_model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(params['device']), target.to(params['device'])
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()


def average_models(global_model, client_models):
    """Average models across all clients."""
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k] for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)


def evaluate_model(model, data_loader):
    """Compute loss and accuracy of a single model on a data_loader."""
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(params['device']), target.to(params['device'])
            output = model(data)
            loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)

    return loss, acc


def evaluate_many_models(models, data_loader):
    """Compute average loss and accuracy of multiple models on a data_loader."""
    num_nodes = len(models)
    losses = np.zeros(num_nodes)
    accuracies = np.zeros(num_nodes)
    for i in range(num_nodes):
        losses[i], accuracies[i] = evaluate_model(models[i], data_loader)
    return losses, accuracies


class Net_eNTK(nn.Module):
    def __init__(self):
        super(Net_eNTK, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def compute_eNTK(model, X, subsample_size=100000, seed = 123):
    """"compute eNTK"""
    model.eval()
    params = list(model.parameters())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random_index = torch.randperm(11169345)[:subsample_size]
    grads = None
    for i in tqdm(range(X.size()[0])):
        model.zero_grad()
        model.forward(X[i : i+1])[0].backward()

        grad = []
        for param in params:
            if param.requires_grad:
                grad.append(param.grad.flatten())
        grad = torch.cat(grad)
        # print(len(grad))
        grad = grad[random_index]

        if grads is None:
            grads = torch.zeros((X.size()[0], grad.size()[0]), dtype=torch.half)
        grads[i, :] = grad

    return grads


def eNTK_loader(model,data, targets,params):
    grads_data = compute_eNTK(model, data.to(params['device']),seed=params['seed'])
    grads_data = grads_data.float().to(params['device'])
    targets = targets.to(params['device'])
    # gradient
    targets_onehot = F.one_hot(targets, num_classes=10).to(params['device']) 
    del data
    torch.cuda.empty_cache()
    return grads_data.cpu(),targets_onehot.cpu(),targets.cpu()

def eNTK_trainer(model, model_c,train_loader,params,optimizer,criterion):
    """Train a client_model on the train_loder data."""
    model.train()
    model_c.train()
    batch_idx = 0
    for data, targets in train_loader:
        grads_data,targets_onehot,targets = eNTK_loader(model,data, targets,params)
        # eval on train

        optimizer.zero_grad()

        logits_class_train = model_c(grads_data.to(params['device']))
        loss = criterion(logits_class_train,targets.to(params['device']))

        loss.backward()
        optimizer.step()

        _, targets_pred_train = logits_class_train.max(1)

        train_acc = targets_pred_train.eq(targets.to(params['device'])).sum() / (1.0 * logits_class_train.shape[0])

        print('batch %d: train accuracy=%0.5g' % (batch_idx, train_acc.item()))
        batch_idx+=1
    return train_acc

def imbalance(R,maj_classes= [0, 1, 2, 3, 4],min_classes= [5, 6, 7, 8, 9]):
    # number of samples
    maj_classes = [0, 1, 2, 3, 4]
    min_classes = [5, 6, 7, 8, 9]
    classes = maj_classes + min_classes
    N = 5000
    K = 10
    
    
    n_c_train_target = {}
    for c in range(K):
        if c in maj_classes:
            n_c_train_target[c] = N
        else:
            n_c_train_target[c] = int(N // R)
    

    print("n_c_target: " + str(n_c_train_target))
    N_train_total = sum(n_c_train_target.values())
    print("N_train_total: " + str(N_train_total))

    return n_c_train_target,classes
