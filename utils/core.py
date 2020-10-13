from torchvision.datasets import CIFAR10,CIFAR100
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
import torch

from network.resnet import ResNet34, ResNet18
from trainer.utils import transformer

def load_student_n_testdataset(dataset_name,data_path,bs_size, lr_S):

    train_trans, test_trans=transformer(dataset_name)

    if dataset_name == 'cifar10': 
        net = ResNet18().cuda()
        data_test = CIFAR10(data_path,
                        train=False,
                        transform=test_trans,download=True)
    elif dataset_name == 'cifar100': 
        net = ResNet18(num_classes=100).cuda()
        data_test = CIFAR100(data_path,
                        train=False,
                        transform=test_trans,download=True)
    data_test_loader = DataLoader(data_test, batch_size=bs_size, num_workers=0)

    optimizer_S = torch.optim.SGD(net.parameters(), lr=lr_S, momentum=0.9, nesterov=True)
    return data_test_loader,optimizer_S, net,data_test

def load_teacher(dataset_name,teacher_path):
    if dataset_name =='cifar10':
        teacher=ResNet34()
    elif dataset_name =='cifar100':
        teacher=ResNet34(100)
    
    checkpoint = torch.load(teacher_path + 'teacher_'+dataset_name+'.pt')
    teacher.load_state_dict(checkpoint)  

    if torch.cuda.is_available():
        teacher = teacher.cuda()
    teacher.eval()

    return teacher


class hook_for_BNLoss():

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        
        T_mean = module.running_mean.data
        T_var = module.running_var.data
        self.G_kd_loss = self.Gaussian_kd(mean, var, T_mean, T_var)

    def Gaussian_kd(self, mean, var, T_mean, T_var):

        num = (mean-T_mean)**2 + var
        denom = 2*T_var
        std = torch.sqrt(var)
        T_std = torch.sqrt(T_var)

        return num/denom - torch.log(std/T_std) - 0.5

    def close(self):
        self.hook.remove()