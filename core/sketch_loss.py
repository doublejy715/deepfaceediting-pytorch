import torch
from utils import utils
import time

class lossCollector():
    def __init__(self, args):
        super(lossCollector, self).__init__()
        self.args = args
        self.start_time = time.time()
        self.loss_dict = {}
        self.L1 = torch.nn.L1Loss()
        self.L2 = torch.nn.MSELoss()

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

    def get_L1_loss_with_same_person(self, a, b, same_person):
        return torch.sum(0.5 * torch.mean(torch.abs(a - b, 2).reshape(self.args.batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)

    def get_L2_loss_with_same_person(self, a, b, same_person):
        return torch.sum(0.5 * torch.mean(torch.pow(a - b, 2).reshape(self.args.batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)

    def get_attr_loss(self, a, b):
        L_attr = 0
        for i in range(len(a)):
            L_attr += torch.mean(torch.pow((a[i] - b[i]), 2).reshape(self.args.batch_size, -1), dim=1).mean()
        L_attr /= 2.0

        return L_attr
        
    def get_hinge_loss(self, Di, label):
        L_adv = 0
        for di in Di:
            L_adv += utils.hinge_loss(di[0], label)
        return L_adv
        
    def get_loss_G(self, I_t, Y, I_t_attr, I_s_id, Y_attr, Y_id, d_adv, same_person):
        
        L_G = 0.0
        
        # adv loss
        if self.args.W_adv:
            L_adv = self.get_hinge_loss(d_adv, True)
            L_G += self.args.W_adv * L_adv
        
        # id loss
        if self.args.W_id:
            L_id = self.get_id_loss(I_s_id.detach(), Y_id)
            L_G += self.args.W_id * L_id

        # attr loss
        if self.args.W_attr:
            L_attr = self.get_attr_loss(I_t_attr, Y_attr)
            L_G += self.args.W_attr * L_attr

        # recon_loss
        if self.args.W_recon:
            L_recon = self.get_L2_loss_with_same_person(Y, I_t, same_person)
            L_G += self.args.W_recon * L_recon
        
        # save in dict
        self.loss_dict["L_G"] = round(L_G.item(), 4)
        self.loss_dict["L_attr"] = round(L_attr.item(), 4)
        self.loss_dict["L_id"] = round(L_id.item(), 4)
        self.loss_dict["L_adv"] = round(L_adv.item(), 4)
        self.loss_dict["L_recon"] = round(L_recon.item(), 4)

        return L_G

    def get_loss_D(self, d_true, d_fake):
        
        # get loss
        L_true = self.get_hinge_loss(d_true, True)
        L_fake = self.get_hinge_loss(d_fake, False)
        L_D = 0.5*(L_true.mean() + L_fake.mean())
        
        # save in dict
        self.loss_dict["L_true"] = round(L_true.mean().item(), 4)
        self.loss_dict["L_fake"] = round(L_fake.mean().item(), 4)
        self.loss_dict["L_D"] = round(L_D.item(), 4)

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
    