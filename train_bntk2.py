import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import copy
import argparse
import os
from utils.util import *
from pretrain_code.generate_cifar import IMBALANCECIFAR10
from loss.loss_imbalance import *
import torchvision.models as models
from model.classifier import NTK_classify
import pickle


params = {
        "lr" : 0.1,
        "batch_size": 128,
        "gamma" : 0.5,
        "R": 10,
        "local_steps": 100 ,
        "seed": 1,
        "imb_type": 'step',
        "save_path":'',
        "pretrain_path": 'NC_CDT_resnet_CIFAR_R10/cdt_gamma_0.0_version_0model.pth',
        "data_path" :'./data',
        "workers": 4,
        'dataset_name' : 'CIFAR10',
        "train_sampler" : None,
        "loss_name": 'CDT',
        'device': 'cuda:0',
        'num_rounds':100,
        'round':5,
        "lr_decay":0.5,
        "epochs":10
    }
class features:
            pass


def hook(self, input, output):
    features.value = input[0].clone()
torch.manual_seed(params["seed"])
torch.cuda.manual_seed(params["seed"])

R = params['R']
delta_list = [R, R, R, R, R, 1, 1, 1, 1, 1]
n_c_train_target,classes = imbalance(R)
K = len(classes)
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
if params['dataset_name'] == 'CIFAR10':
    im_size      = 32
padded_im_size      = 32
C                   = 10
if params['dataset_name'] == 'CIFAR10':
    input_ch        = 3

################## DATA LOADER 
data_path = './data'

train_dataset = IMBALANCECIFAR10(data_path, imb_type=params['imb_type'],
                                rand_number=params['seed'], train=True, download=False,
                                transform=transform_train, n_c_train_target=n_c_train_target, classes=classes)
val_dataset = datasets.CIFAR10(data_path, train=False, download=False, transform=transform_val)

n_c_test_target = [0 for _ in range(0,K)]
for val_data in val_dataset:
    c = int(val_data[1])
    n_c_test_target[c] += 1

print("^"*100)
print("n_c_test_target: " + str(n_c_test_target))
print("^"*100)

cls_num_list = train_dataset.get_cls_num_list()
cls_priors = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
print('\nTotal number of samples: ', sum(cls_num_list))
print('cls num list:')
print(cls_num_list)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=params['batch_size'], shuffle=(params["train_sampler"] is None),
    num_workers=params['workers'], pin_memory=True, sampler=params["train_sampler"] )

analysis_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=params['batch_size'], shuffle=(params["train_sampler"] is None),
    num_workers=params['workers'], pin_memory=True, sampler=params["train_sampler"])

test_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=params['batch_size'], shuffle=False,
    num_workers=params['workers'], pin_memory=True)

if params['loss_name'] == 'CrossEntropyLoss':
    criterion = nn.CrossEntropyLoss().to(params['device'])
    criterion_summed = nn.CrossEntropyLoss(reduction='sum').to(params['device'])
elif params['loss_name'] == "CDT":
    criterion = CDTLoss(delta_list, gamma=params['gamma'], weight=None, reduction=None).to(params['device'])
    criterion_summed = CDTLoss(delta_list, gamma=params['gamma'], weight=None, reduction="sum").to(params['device'])
elif params['loss_name'] == "LDT":
    criterion = LDTLoss(delta_list,gamma=params['gamma'], weight=None, reduction=None).to(params['device'])
    criterion_summed = LDTLoss(delta_list, gamma=params['gamma'], weight=None, reduction="sum").to(params['device'])
else:
    print('the loss is not supported')

model = models.resnet18(pretrained=False, num_classes=C)
model.conv1 = nn.Conv2d(input_ch, model.conv1.weight.shape[0], 3, 1, 1, bias=False) # Small dataset filter size used by He et al. (2015)
model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
model.fc = nn.Linear(in_features=512, out_features=10, bias=False)
model2 = torch.load(params['pretrain_path'])

model.load_state_dict(model2.state_dict(),strict =True)
### random reinitialized the classifier
model.fc = nn.Linear(512, 1).to(params['device'])
model = model.to(params['device'])

# Init and load model ckpt
print('load model')
######################################
epoch_list = [1,   3,   5,   7,   9,
                11,  20,  30,  40,  60,
                80, 101, 120, 140, 160,
                180, 201, 220, 235, 245, 250, 260,
                275, 280, 290, 299, 305, 310, 315, 
                320, 325, 330, 335, 340, 345, 349, 350]
epochs_lr_decay     = [116, 232]
######################################

# Init linear models
model_c = NTK_classify(100000, 10).to(params['device'])

optimizer = optim.SGD(model_c.parameters(),
                        lr=params['lr'])

lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=epochs_lr_decay,
                                                gamma=params['lr_decay'])

# Test
## batch sampling 
for data, targets in test_loader:
    grad_eval, target_eval_onehot, target_eval  = eNTK_loader(model, data, targets ,params)
    break

class graphs:
    def __init__(self):
        self.W = []

f = open("NTK_retrain_CDT_R_" + str(R)+ "_NTK_Prints.txt", "w")
f.write("Create File!\n")
f.flush()

save_graph = graphs()
root_path = "NTK_retrain_CDT_R_" + str(R) + "/cdt_gamma_" + str(0.5) + "_version_" + str(0)
# root_path = "./here"
save_path = root_path

for round_idx in range(params['epochs']):
    torch.cuda.empty_cache()

    train_acc = eNTK_trainer(model, model_c,train_loader,params,optimizer,criterion)

    lr_scheduler.step()

    if round_idx in epoch_list:
        # eval on test
        model_c.eval()
        with torch.no_grad():
            logits_class_test = model_c(grad_eval.to(params['device']))
            _, targets_pred_test = logits_class_test.max(1)
            test_acc = targets_pred_test.eq(target_eval.cuda()).sum() / (1.0 * logits_class_test.shape[0])
            # test_acc = targets_pred_test.eq(target_eval).sum() / (1.0 * logits_class_test.shape[0])
            print('Round %d: train accuracy=%0.5g test accuracy=%0.5g' % (round_idx, train_acc.item(), test_acc.item()))
        

        save_graph.W.append(model_c.weight.cpu().numpy())

        this_epoch_W = model_c.weight.cpu().numpy()
        f.write("------------------------------\n")
        f.write("Epoch " + str(round_idx) + " -> \n")
        f.write("train_acc: " + str(train_acc) + " -> \n")
        f.flush()

path_train = save_path + 'graphs_train_save_{}_{}_R{}.pkl'.format("NTK_CDT_0.5", "CIFAR10", int(R))
f_train = open(path_train, "wb")
pickle.dump(save_graph, f_train)
f_train.close()