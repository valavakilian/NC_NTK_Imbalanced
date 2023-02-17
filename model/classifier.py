import torch
import torch.nn as nn
import torch.nn.functional as F

class NTK_classify(nn.Module):
    def __init__(self,dim_rep =100000, classes = 10) :
        super(NTK_classify,self).__init__()
        self.weight = torch.nn.Parameter(torch.zeros(dim_rep,classes))
    def forward(self,x):
        return x@self.weight