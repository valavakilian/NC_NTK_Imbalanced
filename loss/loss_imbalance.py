import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class CDTLoss(nn.Module):

    def __init__(self, delta_list, gamma=0.3, weight=None, reduction=None):
        super(CDTLoss, self).__init__()
        Delta_list = np.array(delta_list) ** gamma
        Delta_list = len(Delta_list) * Delta_list / sum(Delta_list)
        # Delta_list = Delta_list / np.min(Delta_list)
        print("Delta_list" + str(Delta_list))
        self.Delta_list = torch.cuda.FloatTensor(Delta_list)
        # self.Delta_list = torch.FloatTensor(Delta_list)
        self.weight = weight
        self.reduction = reduction

    def forward(self, x, target):
        # print("-"*20)
        # print("self.Delta_list: " + str(self.Delta_list))
        if self.reduction == "sum":
            output = x * self.Delta_list
            return F.cross_entropy(output, target, weight=self.weight, reduction='sum')
        else:
            output = x * self.Delta_list
            # print("x: " + str(x))
            # print("output: " + str(output))
            return F.cross_entropy(output, target, weight=self.weight)


class LDTLoss(nn.Module):

    def __init__(self, Delta_list, gamma = 0.3,weight=None, reduction = None):
        super(LDTLoss, self).__init__()
        self.gamma = gamma
        self.Delta_list = np.array(Delta_list) ** self.gamma
        self.Delta_list = len(self.Delta_list) * self.Delta_list / sum(self.Delta_list)
        self.Delta_list = torch.cuda.FloatTensor(self.Delta_list)
        self.weight = weight
        self.reduction = reduction

    def forward(self, x, target):
        if self.reduction == "sum":
            ldt_output = (x.T*self.Delta_list[target]).T
            return F.cross_entropy(ldt_output, target, weight=self.weight, reduction = 'sum')
        else:
            ldt_output = (x.T*self.Delta_list[target]).T
            return F.cross_entropy(ldt_output, target, weight=self.weight)
