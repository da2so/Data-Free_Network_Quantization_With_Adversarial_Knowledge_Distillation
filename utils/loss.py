import torch.nn.functional as F
import torch.nn as nn
import torch

def kd_loss(y, teacher_scores,T=1.0):
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) *(T**2)  / y.shape[0]
    return l_kl

def BCE_loss():
    return torch.nn.BCELoss()

def CE_loss():
    return torch.nn.CrossEntropyLoss()

def KLDiv_loss():
    return torch.nn.KLDivLoss()

def MSE_loss():
    return torch.nn.MSELoss()
class Entropy_Loss(nn.Module):
    def __init__(self):
        super(Entropy_Loss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

def E_loss():
    return Entropy_Loss()


