from utils import utils
import time
import numpy as np
import math

import torch
import torch.nn as nn
import torchvision
from torch.nn.functional import grid_sample

from nets.discriminator import MultiscaleDiscriminator
from utils.nets_utils import get_norm_layer, weights_init

class lossCollector():
    def __init__(self, args,FM_D):
        super(lossCollector, self).__init__()
        self.args = args
        self.start_time = time.time()
        self.loss_dict = {}
        self.L1 = torch.nn.L1Loss()
        self.L2 = torch.nn.MSELoss()
        self.loss_VGG = VGGPerceptualLoss()
        self.loss_LAB = LabColorLoss()
        self.FM_D = FM_D

    def get_id_loss(self, a, b):
        return (1 - torch.cosine_similarity(a, b, dim=1)).mean()

    def get_lpips_loss(self, a, b):
        return self.lpips(a, b)
        
    def get_L1_loss(self, a, b,test=False):
        loss = self.L1(a, b)
        if test:
            self.loss_dict['L_test']=round(loss.item(),4)
        else:
            self.loss_dict['L_train']=round(loss.item(),4)
            return loss

    def get_L2_loss(self, a, b):
        return self.L2(a, b)

    # for step 2 loss
    def get_img_encoder_loss(self, sketch_ftmap_layers, image_ftmap_layers, test=False):
        loss = 0.0
        for sketch_tfmap, image_tfmap in zip(sketch_ftmap_layers, image_ftmap_layers):
            loss += self.get_L1_loss(sketch_tfmap,image_tfmap,test)

        return loss

    def get_Lab_loss(self,a,b):
        loss = self.loss_LAB(b,a)
        return loss

    def get_FM_loss(self,a,b):
        L_fm = 0
        n_layers_D = 4
        num_D = 2
        feat_weights = 4.0 / (n_layers_D + 1)
        D_weights = 1.0 / num_D
        for i in range(0, n_layers_D):
            L_fm += D_weights * feat_weights * self.get_L1_loss(a[i].detach(), b[i])
            L_fm += D_weights * feat_weights * self.get_L1_loss(a[i].detach(), b[i])
        return L_fm

    def get_VGG_loss(self,a,b):
        return self.loss_VGG(a,b)

    def get_geo_loss(self,a,b):
        return self.get_L1_loss(a,b)

    def get_hinge_loss(self, Di, label):
        L_adv = 0
        for di in Di:
            L_adv += utils.hinge_loss(di[0], label)
        return L_adv

    # for step 3 loss

    def get_recon_loss(self,a,b):
        L_recon = 0.0
        if self.args.W_recon_Lab:
            L_recon_lab = self.args.W_recon_Lab * self.get_Lab_loss(a,b)
            L_recon += L_recon_lab

        if self.args.W_recon_FM:
            L_recon_fm = self.args.W_recon_FM * self.get_FM_loss(a,b)
            L_recon += L_recon_fm

        if self.args.W_recon_VGG:
            L_recon_VGG = self.args.W_recon_VGG * self.get_VGG_loss(a,b)
            L_recon += L_recon_VGG

        # save in dict
        self.loss_dict["L_recon"] = round(L_recon.item(), 4)
        self.loss_dict["L_recon_lab"] = round(L_recon_lab.item(), 4)
        self.loss_dict["L_recon_fm"] = round(L_recon_fm.item(), 4)
        self.loss_dict["L_recon_VGG"] = round(L_recon_VGG.item(), 4)

        return L_recon

    def get_cycle_loss(self,a,b):
        L_swap_cycle = 0.0

        if self.args.W_cycle_Lab:
            L_swap_cycle_lab = self.args.W_cycle_Lab * self.get_Lab_loss(a,b)
            L_swap_cycle += L_swap_cycle_lab

        if self.args.W_cycle_FM:
            L_swap_cycle_fm = self.args.W_cycle_FM * self.get_FM_loss(a,b)
            L_swap_cycle += L_swap_cycle_fm

        if self.args.W_cycle_VGG:
            L_swap_cycle_VGG = self.args.W_cycle_VGG * self.get_VGG_loss(a,b)
            L_swap_cycle += L_swap_cycle_VGG

        # save in dict
        self.loss_dict["L_swap_cycle"] = round(L_swap_cycle.item(), 4)
        self.loss_dict["L_swap_cycle_lab"] = round(L_swap_cycle_lab.item(), 4)
        self.loss_dict["L_swap_cycle_fm"] = round(L_swap_cycle_fm.item(), 4)
        self.loss_dict["L_swap_cycle_VGG"] = round(L_swap_cycle_VGG.item(), 4)
        return L_swap_cycle

    def get_swap_loss(self,a,b):
        L_swap = 0.0
        if self.args.W_swap_geo:
            L_swap_geo = self.args.W_swap_geo * self.get_geo_loss(a,b)
            L_swap += L_swap_geo

        if self.args.W_swap_cycle:
            L_swap_cycle = self.args.W_swap_cycle * self.get_cycle_loss(a,b)
            L_swap += L_swap_cycle
        
        # save in dict
        self.loss_dict["L_swap"] = round(L_swap.item(), 4)
        self.loss_dict["L_swap_geo"] = round(L_swap_geo.item(), 4)
        self.loss_dict["L_swap_cycle"] = round(L_swap_cycle.item(), 4)
        return L_swap

    def get_adv_loss(self, d_geo_real, d_app_real, d_recon_geo_img, d_recon_app_img, d_mix_img):
        L_adv = 0.0
        if self.args.W_adv_geo:
            L_adv_geo = self.args.W_adv_geo * math.log10(d_geo_real).mean()
            L_adv += L_adv_geo

        if self.args.W_adv_app:
            L_adv_app = self.args.W_adv_app * math.log10(d_app_real).mean()
            L_adv += L_adv_app

        if self.args.W_adv_recon_geo:
            L_adv_recon_geo = self.args.W_adv_recon_geo * (1-math.log10(d_recon_geo_img)).mean()
            L_adv += L_adv_recon_geo

        if self.args.W_adv_recon_app:
            L_adv_recon_app = self.args.W_adv_recon_app * (1-math.log10(d_recon_app_img)).mean()
            L_adv += L_adv_recon_app
        
        if self.args.W_adv_mix:
            L_adv_mix = self.args.W_adv_mix * (1-math.log10(d_mix_img)).mean()
            L_adv += L_adv_mix

        # save in dict
        self.loss_dict["L_adv"] = round(L_adv.item(), 4)
        self.loss_dict["L_adv_geo"] = round(L_adv_geo.item(), 4)
        self.loss_dict["L_adv_app"] = round(L_adv_app.item(), 4)
        self.loss_dict["L_adv_recon_geo"] = round(L_adv_recon_geo.item(), 4)
        self.loss_dict["L_adv_recon_app"] = round(L_adv_recon_app.item(), 4)
        self.loss_dict["L_adv_mix"] = round(L_adv_mix.item(), 4)

        return L_adv

    def get_loss_G(self, geo_real, app_real, recon_geo_img, recon_app_img, d_geo_real, d_app_real, d_recon_geo_img, d_recon_app_img, d_mix_img):
        L_G = 0.0
        if self.args.W_recon:
            L_recon = self.get_cycle_loss(geo_real, recon_geo_img)
            L_G += self.args.W_recon * L_recon

        if self.args.W_swap:
            L_swap = self.get_swap_loss(app_real, recon_app_img)
            L_G += self.args.W_swap * L_swap

        if self.args.W_adv:
            L_adv = self.get_adv_loss(d_geo_real, d_app_real, d_recon_geo_img, d_recon_app_img, d_mix_img)
            L_G += self.args.W_adv * L_adv

        # save in dict
        self.loss_dict["L_G"] = round(L_G.item(), 4)
        self.loss_dict["L_recon"] = round(L_recon.item(), 4)
        self.loss_dict["L_swap"] = round(L_swap.item(), 4)
        self.loss_dict["L_adv"] = round(L_adv.item(), 4)

        return L_G

    # I1,I1',I2,I2'
    def get_loss_D(self, d_geo_real, d_app_real, d_recon_geo_img, d_recon_app_img, d_mix_img):
        L_D = 0.0
        if self.args.W_D_geo:
            L_D_geo = self.args.W_D_geo * math.log10(d_geo_real).mean()
            L_D += L_D_geo

        if self.args.W_D_app:
            L_D_app = self.args.W_D_app * math.log10(d_app_real).mean()
            L_D += L_D_app

        if self.args.W_D_recon_geo:
            L_D_recon_geo = self.args.W_D_recon_geo * (1-math.log10(d_recon_geo_img)).mean()
            L_D += L_D_recon_geo

        if self.args.W_D_recon_app:
            L_D_recon_app = self.args.W_D_recon_app * (1-math.log10(d_recon_app_img)).mean()
            L_D += L_D_recon_app
        
        if self.args.W_D_mix:
            L_D_mix = self.args.W_D_mix * (1-math.log10(d_mix_img)).mean()
            L_D += L_D_mix

        # save in dict
        self.loss_dict["L_D"] = round(L_D.item(), 4)
        self.loss_dict["L_D_geo"] = round(L_D_geo.item(), 4)
        self.loss_dict["L_D_app"] = round(L_D_app.item(), 4)
        self.loss_dict["L_D_recon_geo"] = round(L_D_recon_geo.item(), 4)
        self.loss_dict["L_D_recon_app"] = round(L_D_recon_app.item(), 4)
        self.loss_dict["L_D_mix"] = round(L_D_mix.item(), 4)
        
        return L_D


    def print_loss(self, global_step):
        seconds = int(time.time() - self.start_time)
        print("")
        print(f"[ {seconds//3600//24:02}d {(seconds//3600)%24:02}h {(seconds//60)%60:02}m {seconds%60:02}s ]")
        print(f'steps: {global_step:06} / {self.args.max_step}')
        print(f'lossD: {self.loss_dict["L_D"]} | lossG: {self.loss_dict["L_G"]}')
    
    def print_L1_loss(self, global_step):
        seconds = int(time.time() - self.start_time)
        print("")
        print(f"[ {seconds//3600//24:02}d {(seconds//3600)%24:02}h {(seconds//60)%60:02}m {seconds%60:02}s ]")
        print(f'steps: {global_step:06} / {self.args.max_step}')
        print(f'loss: {self.loss_dict["L_train"]}')
    
class LabColorLoss(nn.Module):
    def __init__(self):
        super(LabColorLoss, self).__init__()
        self.balance_Lab = True
        self.FloatTensor = torch.cuda.FloatTensor
        self.criterion = nn.L1Loss()
        self.M = torch.tensor([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
    # cal lab written by tzt
    def func(self, x):
        mask = (x > 0.008856).float()
        return x ** (1 / 3) *mask + (7.787 * x + 0.137931) * (1 - mask)

    def RGB2Lab(self, input):
        # the range of input is from 0 to 1
        input_x = 0.412453 * input[:, 0, :, :] + 0.357580 * input[:, 1, :, :] + 0.180423 * input[:, 2, :, :]
        input_y = 0.212671 * input[:, 0, :, :] + 0.715160 * input[:, 1, :, :] + 0.072169 * input[:, 2, :, :]
        input_z = 0.019334 * input[:, 0, :, :] + 0.119193 * input[:, 1, :, :] + 0.950227 * input[:, 2, :, :]
        # normalize
        # input_xyz = input_xyz / 255.0
        input_x = input_x / 0.950456 # X
        input_y = input_y / 1.0 # Y
        input_z = input_z / 1.088754 # Z

        fx = self.func(input_x)
        fy = self.func(input_y)
        fz = self.func(input_z)

        Y_mask = (input_y > 0.008856).float()
        input_l = (116.0 * fy - 16.0) * Y_mask + 903.3 * input_y * (1 - Y_mask) # L
        input_a = 500 * (fx - fy) # a
        input_b = 200 * (fy - fz) # b

        input_l = torch.unsqueeze(input_l, 1)
        input_a = torch.unsqueeze(input_a, 1)
        input_b = torch.unsqueeze(input_b, 1)
        return torch.cat([input_l, input_a, input_b],1)
    # cal lab written by liuqk
    def f(self, input):
        output = input * 1
        mask = input > 0.008856
        output[mask] = torch.pow(input[mask], 1 / 3)
        output[1 - mask] = 7.787 * input[1 - mask] + 0.137931
        return output

    def rgb2xyz(self, input):
        """Change RGB color to XYZ color
        Args:
            input: 4-D tensor, [B, C, H, W]
        """
        assert input.size(1) == 3

        M_tmp = self.M.to(input.device).unsqueeze(0)
        M_tmp = M_tmp.repeat(input.size(0), 1, 1)  # BxCxC

        output = torch.einsum('bnc,bchw->bnhw', M_tmp, input)  # BxCxHxW

        # output = output / 255.0

        M_tmp = M_tmp.sum(dim=2, keepdim=True)  # BxCx1
        M_tmp = M_tmp.unsqueeze(3)  # BxCx1x1

        return output / M_tmp

    def xyz2lab(self, input):
        assert input.size(1) == 3

        output = input * 1
        xyz_f = self.f(input)
        # compute l
        mask = input[:, 1, :, :] > 0.008856
        output[:, 0, :, :][mask] = 116 * xyz_f[:, 1, :, :][mask] - 16
        output[:, 0, :, :][1 - mask] = 903.3 * input[:, 1, :, :][1 - mask]

        # compute a
        output[:, 1, :, :] = 500 * (xyz_f[:, 0, :, :] - xyz_f[:, 1, :, :])

        # compute b
        output[:, 2, :, :] = 200 * (xyz_f[:, 1, :, :] - xyz_f[:, 2, :, :])

        return output

    def cal_weight(self, tag_image, mask):
        n,c,h,w = tag_image.size()
        raw = np.load('utils/ab_count.npy')
        weight = self.FloatTensor(raw)
        weight = torch.unsqueeze(torch.unsqueeze(weight, 0), 0)
        weight = weight.repeat(n, 1, 1, 1)
        weight[weight == 0] = 1
        weight = weight.max() / weight
        weight[weight > 10] = 10

        image_a = torch.unsqueeze(tag_image[:,1,:,:], 1)
        image_b = torch.unsqueeze(tag_image[:,2,:,:], 1)
        m = torch.cat([image_b, image_a], 1) + 128
        m[m < 0] = 0
        m[m > 255] = 255
        m = m.int().float()
        m = (m - 127.5) / 127.5
        m = m.permute([0, 2, 3, 1])

        weight_mask = grid_sample(weight, m, mode='nearest')
        weight_mask = weight_mask * mask
        weight_mask[weight_mask == 0] = 1

        return weight_mask

    def forward(self, fake, real, mask=None):
        # normalize to 0~1
        fake_RGB = (fake + 1) / 2.0
        real_RGB = (real + 1) / 2.0
        ## from RGB to Lab by tzt
        # fake_Lab = self.RGB2Lab(fake_RGB)
        # real_Lab = self.RGB2Lab(real_RGB)
        # from RGB to Lab by liuqk
        fake_xyz = self.rgb2xyz(fake_RGB)
        fake_Lab = self.xyz2lab(fake_xyz)
        real_xyz = self.rgb2xyz(real_RGB)
        real_Lab = self.xyz2lab(real_xyz)
        # cal loss
        if self.balance_Lab:
            weight_mask = self.cal_weight(real_Lab, mask)
            diff = torch.abs(fake_Lab[:,1:,:,:] - real_Lab[:,1:,:,:].detach())
            w_diff = weight_mask * diff
            lab_loss = torch.mean(w_diff)
        else:
            lab_loss = self.criterion(fake_Lab[:,1:,:,:], real_Lab[:,1:,:,:].detach())
        # if (lab_loss != lab_loss).sum() > 0:
        #     pdb.set_trace()
        return lab_loss

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss