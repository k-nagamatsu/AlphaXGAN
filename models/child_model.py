from models import generator, discriminator
from utils.utils import *
from utils.fid_score import calculate_fid_given_paths
from utils.inception_score import get_inception_score

import time,os
from imageio import imsave
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image

class Child_Model():
    def __init__(self,option):# 1.4.1 確認済
        self.option = option
        self.netG = generator.Generator_Net(option).to(option.device)
        self.netD = discriminator.Discriminator_Net(option).to(option.device)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        self.optimizerG = optim.Adam(self.netG.parameters(),lr=option.lrG,betas=(option.beta1,option.beta2))
        self.optimizerD = optim.Adam(self.netD.parameters(),lr=option.lrD,betas=(option.beta1,option.beta2))
        if option.use_dynamic_reset:
            self.loss_windowG = Window(option.dynamic_reset_window_size)
            self.loss_windowD = Window(option.dynamic_reset_window_size)
        if option.loss == 'BCE':
            self.criterion = nn.BCELoss()
        self.fixed_noise = torch.randn(self.option.fixed_noise_size,self.option.latent_dim,device=self.option.device)
        # fid stat
        if self.option.dataset == 'cifar10':
            self.fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
        elif self.option.dataset == 'stl10':
            self.fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
        else:
            raise NotImplementedError(f'no fid stat for {self.option.dataset}')
        assert os.path.exists(self.fid_stat)

    def train(self,phase,trainloader,stage,search_iter,design=None,controller_model=None):# 1.4.4 確認済
        assert (phase =='SDTG' or phase =='EDTG')
        assert ((controller_model and not design) if phase =='SDTG' else (design and not controller_model))
        if phase == 'SDTG':
            N_epoch = self.option.N_epoch_SDTG
            if self.option.use_controllerG: controller_model.ctrlG.net.eval()
            if self.option.use_controllerD: controller_model.ctrlD.net.eval()
        elif phase == 'EDTG':
            N_epoch = self.option.N_epoch_EDTG
            best_mean  = 0
            best_FID = np.inf
            best_mean_epoch = None
            best_FID_epoch  = None
            if self.option.loss == 'Hinge':
                step = 0
                gen_avg_param = copy_params(self.netG)

        flag_reset = False
        for epoch in range(N_epoch):
            start = time.time()
            self.netG.train()
            self.netD.train()
            lossG = 0
            lossD = 0
            for data in trainloader:
                if phase == 'SDTG':
                    with torch.no_grad():
                        design,_,_ = controller_model.sample(stage)
                
                if self.option.loss == 'BCE':
                    real_label = 1
                    fake_label = 0
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ## Train with all-real batch
                    self.netD.zero_grad()
                    real_imgs = data[0].to(self.option.device)
                    label = torch.full((self.option.train_batch_size,), real_label, device=self.option.device,dtype=torch.float)
                    output = self.netD(real_imgs,design).view(-1)
                    errD_real = self.criterion(output, label)
                    errD_real.backward()
                    D_x = output.mean().item()

                    ## Train with all-fake batch
                    noise = torch.randn(self.option.train_batch_size,self.option.latent_dim,device=self.option.device)
                    fake_imgs = self.netG(noise,design)
                    label.fill_(fake_label)
                    output = self.netD(fake_imgs.detach(),design).view(-1)
                    errD_fake = self.criterion(output, label)
                    errD_fake.backward()
                    errD = errD_real + errD_fake
                    self.optimizerD.step()

                    lossD += errD.item()
                    D_G_z1 = output.mean().item()

                    # (2) Update G network: maximize log(D(G(z)))
                    self.netG.zero_grad()
                    label.fill_(real_label)
                    output = self.netD(fake_imgs,design).view(-1)
                    errG = self.criterion(output, label)
                    errG.backward()
                    self.optimizerG.step()

                    lossG += errG.item()
                    D_G_z2 = output.mean().item()

                else: # self.option.loss == 'Hinge'

                    # -------------------
                    # Train Discriminator
                    # -------------------
                    self.optimizerD.zero_grad()
                    real_imgs = data[0].to(self.option.device)
                    real_validity = self.netD(real_imgs,design)
                    noise = torch.randn(real_imgs.shape[0],self.option.latent_dim,device=self.option.device)
                    fake_imgs = self.netG(noise,design)
                    fake_validity = self.netD(fake_imgs.detach(),design)

                    # cal loss
                    errD = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                           torch.mean(nn.ReLU(inplace=True)(1.0 + fake_validity))
                    errD.backward()
                    self.optimizerD.step()
                    lossD += errD.item()

                    # ---------------
                    # Train Generator
                    # ---------------
                    if step % 5 == 0:
                        self.optimizerG.zero_grad()
                        noise = torch.randn(2 * real_imgs.shape[0],self.option.latent_dim,device=self.option.device)
                        fake_imgs = self.netG(noise,design)
                        fake_validity = self.netD(fake_imgs,design)

                        # cal loss
                        errG = -torch.mean(fake_validity)
                        errG.backward()
                        self.optimizerG.step()
                        lossG += errG.item()
                        
                        # moving average weight
                        for p, avg_p in zip(self.netG.parameters(), gen_avg_param):
                            avg_p.mul_(0.999).add_(p.data,alpha=0.001) # add_(0.001, p.data)
                    else:
                        errG = None
                    step += 1

                if (phase == 'SDTG') and self.option.use_dynamic_reset:
                    if errG != None:self.loss_windowG.push(errG.item())
                    self.loss_windowD.push(errD.item())
                    if self.loss_windowG.is_full():
                        if min([self.loss_windowG.var,self.loss_windowD.var]) < self.option.dynamic_reset_threshold:
                            flag_reset = True
                            text = f'{stage},{search_iter},{epoch+1}\n'
                            write(f'../output/log/dynamic_reset.csv',text)
                            break

            elapsed_time = int(time.time() - start)+1
            text =  f'{search_iter},{epoch+1},{elapsed_time},{round(lossG,5)},{round(lossD,5)}'
            print(f'{phase},{text}')
            if self.option.use_dynamic_reset: text += f',{round(self.loss_windowD.var,5)},{round(self.loss_windowG.var,5)}'
            write(f'../output/log/{phase}.csv',text+'\n')

            if flag_reset:
                break

            if phase == 'EDTG' and ((epoch+1) % 5 == 0 or epoch == N_epoch - 1):
                if self.option.loss == 'Hinge':
                    backup_param = copy_params(self.netG)
                    load_params(self.netG, gen_avg_param)
                start_test_time = time.time()
                mean,std,fid_score = self.test(design,search_iter,epoch)
                mean,std,fid_score = round(mean,5),round(std,5),round(fid_score,5)
                elapsed_test_time = int(time.time() - start_test_time)+1
                text =  f'{search_iter},{epoch+1},{elapsed_test_time},{mean},{std},{fid_score}\n'
                write(f'../output/log/EDTG_epoch_score.csv',text)
                if (fid_score < best_FID) or (best_mean < mean):
                    if fid_score < best_FID: best_FID,best_FID_epoch = fid_score,epoch+1
                    if best_mean < mean: best_mean,best_std,best_mean_epoch = mean,std,epoch+1
                    torch.save(self.netG.state_dict(), f'../output/model/netG_epoch{epoch+1}')
                if self.option.loss == 'Hinge':
                    load_params(self.netG, backup_param)
                if best_mean_epoch + 40 < epoch:
                    break

        if (phase == 'EDTG'):
            return best_FID, best_mean, best_std, best_FID_epoch, best_mean_epoch
        else:
            return flag_reset

    def fight(self,design,testloader):# 1.4.1確認済
        self.netG.eval()
        self.netD.eval()
        with torch.no_grad():
            # 本物画像のうちDが正しく本物画像と見抜いた数
            for data in testloader:
                real_imgs = data[0].to(self.option.device)
                break
            real_validity = self.netD(real_imgs,design).view(-1)# 1に近づけたい
            TP = torch.where(real_validity >= 0.5,torch.ones_like(real_validity),torch.zeros_like(real_validity)).sum()

            # 贋作画像のうちDが正しく贋作画像と見抜いた数
            noise = torch.randn(self.option.test_batch_size,self.option.latent_dim,device=self.option.device)
            fake_imgs = self.netG(noise,design)# 0に近づけたい
            fake_validity = self.netD(fake_imgs,design).view(-1)
            TN = torch.where(fake_validity < 0.5,torch.ones_like(fake_validity),torch.zeros_like(fake_validity)).sum()
            TPTN = int(TP + TN)
        winner = 'Dis' if self.option.test_batch_size <= TPTN else 'Gen'
        return winner,TPTN

    def test(self,design,search_iter,epoch,calculate_FID=False):# 1.4.4確認済
        self.netG.eval()
        if calculate_FID:
            # get fid and inception score
            fid_buffer_dir = os.path.join(self.option.path_helper['sample_path'], 'fid_buffer')
            os.makedirs(fid_buffer_dir, exist_ok=True)

        with torch.no_grad():
            imgs = self.netG(self.fixed_noise,design)
        save_image(imgs,f'../output/img/{search_iter}_{epoch+1}.pdf',nrow=10,normalize=True)

        img_list = list()
        for iter_idx in range(self.option.num_eval_imgs // self.option.test_batch_size):
            with torch.no_grad():
                z = torch.randn(self.option.test_batch_size,self.option.latent_dim,device=self.option.device)
                imgs = self.netG(z,design)
                imgs = imgs.mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',torch.uint8).numpy()
            if calculate_FID:
                for img_idx, img in enumerate(imgs):
                    file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
                    imsave(file_name, img)
            img_list.extend(list(imgs))

        mean, std = get_inception_score(img_list)
        fid_score = calculate_fid_given_paths([fid_buffer_dir, self.fid_stat], inception_path=None) if calculate_FID else 1
        if calculate_FID: os.system('rm -r {}'.format(fid_buffer_dir))

        return mean,std,fid_score