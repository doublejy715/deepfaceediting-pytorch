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


    # def get_loss_G(self, I_t, Y, I_t_attr, I_s_id, Y_attr, Y_id, d_adv, same_person):

    #     return L_G

    def get_img_encoder_loss(self, sketch_ftmap_layers, image_ftmap_layers, test=False):
        loss = 0.0
        for sketch_tfmap, image_tfmap in zip(sketch_ftmap_layers, image_ftmap_layers):
            loss += self.L1(sketch_tfmap,image_tfmap)

        if test:
            self.loss_dict['L_test']=round(loss.item(),4)
        else:
            self.loss_dict['L_train']=round(loss.item(),4)

        return loss
    
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
    