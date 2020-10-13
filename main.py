import argparse

import torch
from tensorboardX import SummaryWriter

from dfqad import DFQAD
from trainer.teacher_train import Ttrainer

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10','cifar100'])
    parser.add_argument('--data', type=str, default='./trainer/dataset/')
    parser.add_argument('--teacher_dir', type=str, default='./trainer/models/')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--iter', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
    parser.add_argument('--lr_G', type=float, default=1e-3, help='learning rate for generator')
    parser.add_argument('--lr_S', type=float, default=0.1, help='learning rate for student')
    parser.add_argument('--alpha', type=float, default=0.01, help='alpha value')
    parser.add_argument('--latent_dim', type=int, default=512, help='dimensionality of the latent space')
    parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--saved_img_path', type=str, default='./outputs/saved_img/', help='save path for generated images')
    parser.add_argument('--saved_model_path', type=str, default='./outputs/saved_model/', help='save path for trained stduent')
    parser.add_argument('--do_warmup', type=str2bool, default=True, help= 'do warm-up??')
    parser.add_argument('--do_Ttrain', type=str2bool, default=True, help= 'do train teacher network??')

    opt = parser.parse_args()
    summary = SummaryWriter(f'logs/kdgan_{opt.dataset}')

    if opt.do_Ttrain == True :
        teacher=Ttrainer(dataset=opt.dataset,data_path=opt.data,model_path=opt.teacher_dir )
        teacher.build()

    kd_gan_obj=DFQAD(opt)
    kd_gan_obj.build(summary)

if __name__ == '__main__':
    main()