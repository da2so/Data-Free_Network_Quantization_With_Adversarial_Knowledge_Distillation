import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self,opt):
        self.opt=opt
        nz=self.opt.latent_dim
        ngf=64
        nc=3
        img_size=self.opt.img_size
        super(Generator, self).__init__()

        self.init_size = img_size//8
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*8*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.Conv2d(ngf*8, ngf*4, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU()
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*4, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU()
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False)
        )

    def forward(self, z):
        out = self.l1(z.view(z.shape[0],-1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = nn.functional.interpolate(out, scale_factor=2)
        img = self.conv_blocks0(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img