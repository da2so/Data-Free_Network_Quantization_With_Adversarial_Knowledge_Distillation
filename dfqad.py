import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F

from utils.core import load_teacher, load_student_n_testdataset , hook_for_BNLoss
from network.generator import Generator
from utils.loss import BCE_loss, CE_loss, E_loss, kd_loss
from utils.misc import denormalize, pack_images

class DFQAD():
    def __init__(self,opt):

        self.opt=opt
        #teacher load
        self.teacher=load_teacher(self.opt.dataset,self.opt.teacher_dir)
        
        # student load and test dataset load
        self.data_test_loader,self.optimizer_S, self.student, self.data_test= \
            load_student_n_testdataset(self.opt.dataset,opt.data,\
                self.opt.batch_size, self.opt.lr_S)
        
        #generator load
        self.generator = Generator(self.opt)
        if torch.cuda.is_available():
            self.generator = self.generator.cuda()
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr_G,betas=(0.5,0.999))

        #hooking for BNLoss
        self.loss_r_feature_layers = []

        for module in self.teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.loss_r_feature_layers.append(hook_for_BNLoss(module))

        #configure criterion
        self.criterion = torch.nn.BCELoss()
        self.CELoss=torch.nn.CrossEntropyLoss()
        self.KLDIVLoss = torch.nn.KLDivLoss()
        self.Entropy=E_loss()
        if torch.cuda.is_available():
            self.criterion = self.criterion.cuda()

        self.scheduler_S = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_S,T_max=self.opt.n_epochs, eta_min=0)
        self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G,T_max=self.opt.n_epochs, eta_min=0)


    def build(self,summary):
        print('-'*30+'Main start'+'-'*30)

        self.accr_best=0
        self.accr=0
        if self.opt.do_warmup== True:
            self.warm_up(summary)
        else:
            checkpoint = torch.load(self.opt.saved_model_path + 'warm_up_gan.pt')
            self.generator.load_state_dict(checkpoint)  
            if torch.cuda.is_available():
                self.generator = self.generator.cuda()

        
        for epoch in range(self.opt.n_epochs):
            for i in range(self.opt.iter):
                
                for _ in range(1):
                    self.student.eval()
                    self.generator.train()
                    
                    z = torch.randn(self.opt.batch_size, self.opt.latent_dim).cuda()
                    self.optimizer_G.zero_grad()


                    gen_imgs = self.generator(z)
                    o_T= self.teacher(gen_imgs)
                    o_S= self.student(gen_imgs)
                    pred = o_T.data.max(1)[1]
                    so_T = torch.nn.functional.softmax(o_T, dim = 1)
                    so_T_mean=so_T.mean(dim = 0)
                                    
                    l_ie = (so_T_mean * torch.log(so_T_mean)).sum() #IE loss
                    
                    l_oh= - (so_T * torch.log(so_T)).sum(dim=1).mean() #one-hot entropy               
                
                    l_bn = 0 #BN loss
                    for mod in self.loss_r_feature_layers:
                        l_bn+=mod.G_kd_loss.sum()  

                    l_s=l_ie +l_oh +l_bn
                    
                    l_kd_for_G =kd_loss(o_S, o_T)  #KD loss

                    g_loss= -l_kd_for_G + self.opt.alpha * l_s
                    g_loss.backward()     
                    self.optimizer_G.step()

                for _ in range(10):
                    self.student.train()
                    self.generator.eval()
                    self.optimizer_S.zero_grad()

                    z = torch.randn(self.opt.batch_size, self.opt.latent_dim).cuda()

                    gen_imgs = self.generator(z)
                    o_T= self.teacher(gen_imgs)
                    o_S= self.student(gen_imgs)

                    l_kd_for_S =kd_loss(o_S, o_T.detach())   #KD loss
                    s_loss=l_kd_for_S
                    s_loss.backward()    
                    self.optimizer_S.step() 


                if epoch % 10 == 0 and i==0:
                    print ("[Epoch %d/%d] [loss_logit: %f] [loss_oh: %f] [loss_ie: %f] [loss_BN: %f] [loss_kd: %f]" \
                % (epoch, self.opt.n_epochs,l_l.item(),l_oh.item(), l_ie.item(), l_bn.item(), l_kd_for_S.item()))
                if epoch % 10 != 0 and i==0:
                    print ("[Epoch %d/%d] [loss_kd: %f]" % (epoch,  self.opt.n_epochs, l_kd_for_S.item() ))

            summary.add_image( 'main/generated', pack_images( denormalize(gen_imgs.data,(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)).clamp(0,1).detach().cpu().numpy() ), global_step=epoch )
            summary.add_scalar('main/student_loss', l_kd_for_S.item(), epoch)
            summary.add_scalar('main/generator_loss', g_loss.item(), epoch)
            self.scheduler_S.step()
            self.scheduler_G.step()

            #save generated image per epoch
            self.test(summary,epoch)
            saved_img_path=os.path.join(self.opt.saved_img_path+'main/')
            if epoch >= self.opt.n_epochs-3:
                for m in range(np.shape(gen_imgs)[0]):
                    save_dir=saved_img_path+str(epoch)+'/'+ str(int(pred[m]))+'/'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    vutils.save_image(gen_imgs[m,:,:,:].data.clone(),save_dir+str(m)+'.png',normalize=True)
           
        torch.save(self.student.state_dict(),self.opt.saved_model_path + 'student.pt')
        torch.save(self.generator.state_dict(),self.opt.saved_model_path + 'gan.pt')
        summary.close()
        print('-'*30+'Main end'+'-'*30)


    def warm_up(self, summary, epochs=50):
        print('-'*30+'Warm up start'+'-'*30)
        self.generator.train()

        for epoch in range(epochs):
            for i in range(self.opt.iter):
                z = torch.randn(self.opt.batch_size, self.opt.latent_dim).cuda()

                self.optimizer_G.zero_grad()
                gen_imgs = self.generator(z)
                o_T= self.teacher(gen_imgs)
                pred = o_T.data.max(1)[1]
                so_T = torch.nn.functional.softmax(o_T, dim = 1)
                so_T_mean=so_T.mean(dim = 0)

                l_ie = (so_T_mean * torch.log(so_T_mean)).sum() #IE loss
                l_oh= - (so_T * torch.log(so_T)).sum(dim=1).mean() #one-hot entropy
               
                l_bn = 0 #BN loss
                for mod in self.loss_r_feature_layers:
                    l_bn+=mod.G_kd_loss.sum()  

                l_s=self.opt.alpha*(l_ie +l_oh +l_bn)

                l_s.backward()
                self.optimizer_G.step()

                if i == 1:
                    print ("[Epoch %d/%d]  [loss_oh: %f] [loss_ie: %f] [loss_BN: %f] " \
                % (epoch, epochs,l_oh.item(), l_ie.item(), l_bn.item()))
            self.scheduler_G.step()
            saved_img_path=os.path.join(self.opt.saved_img_path+'warm_up/')

            if epoch >= epochs-3:
                for m in range(np.shape(gen_imgs)[0]):
                    save_dir=saved_img_path+str(epoch)+'/'+ str(int(pred[m]))+'/'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    vutils.save_image(gen_imgs[m,:,:,:].data.clone(),save_dir+str(m)+'.png',normalize=True)
       
            summary.add_image( 'warmup/generated', pack_images( denormalize(gen_imgs.data,(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)).clamp(0,1).detach().cpu().numpy() ), global_step=epoch )
            summary.add_scalar('warmup_loss_sum', l_s.item(), epoch)
        if not os.path.exists(self.opt.saved_model_path):
            os.makedirs(self.opt.saved_model_path)  
        torch.save(self.generator.state_dict(),self.opt.saved_model_path + 'warm_up_gan.pt')
        print('-'*30+'Warm up end'+'-'*30)


    def test(self,summary,epoch):
        self.student.eval()
        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.data_test_loader):
                images = images.cuda()
                labels = labels.cuda()
                output = self.student(images)
                avg_loss += self.CELoss(output, labels).sum()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()

        avg_loss /= len(self.data_test)
        print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), float(total_correct) / len(self.data_test)))
        self.accr = round(float(total_correct) / len(self.data_test), 4)

        summary.add_scalar('main/test_acc', float(total_correct) / len(self.data_test), epoch)
        summary.add_scalar('main/test_loss', avg_loss.item(), epoch)    
        if self.accr > self.accr_best:
            torch.save(self.student.state_dict(),self.opt.saved_model_path + 'student.pt')
            self.accr_best = self.accr