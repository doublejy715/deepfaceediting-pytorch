import abc
from submodel.lpips import LPIPS
from submodel.VGGloss import VGGPerceptualLoss
import torch
import torch.nn.functional as F
import time



class LossInterface(metaclass=abc.ABCMeta):
    def __init__(self, args):
        """
        When overrided, super call is required.
        """
        self.args = args
        self.start_time = time.time()
        self.loss_dict = {}

    def print_loss(self, global_step):
        """
        Print discriminator and generator loss and formatted elapsed time.
        """
        seconds = int(time.time() - self.start_time)
        print("")
        print(f"[ {seconds//3600//24:02}d {(seconds//3600)%24:02}h {(seconds//60)%60:02}m {seconds%60:02}s ]")
        print(f'steps: {global_step:06} / {self.args.max_step}')
        print(f'lossD: {self.loss_dict["L_D"]} | lossG: {self.loss_dict["L_G"]}')

    @abc.abstractmethod
    def get_loss_G(self):
        """
        Caculate generator loss.
        Once loss values are saved in self.loss_dict, they can be uploaded on the 
        dashboard via wandb or printed in self.print_loss. self.print_loss can be 
        overrided as needed.
        """
        pass

    @abc.abstractmethod
    def get_loss_D(self):
        """
        Caculate discriminator loss.
        Once loss values are saved in self.loss_dict, they can be uploaded on the 
        dashboard via wandb or printed in self.print_loss. self.print_loss can be 
        overrided as needed.
        """
        pass


class Loss:
    L1 = torch.nn.L1Loss().to("cuda")
    L2 = torch.nn.MSELoss().to("cuda")

    @classmethod
    def get_lpips_loss(cls, a, b):
        if not hasattr(cls, 'lpips'):
            cls.lpips = LPIPS().eval().to("cuda")
        return cls.lpips(a, b)

    @classmethod
    def get_L1_loss(cls, a, b):   
        return cls.L1(a, b)

    @classmethod
    def get_L2_loss(cls, a, b):
        return cls.L2(a, b)

    @classmethod 
    def get_geo_loss(cls,a,b):
        return cls.get_L1_loss(a,b)


    def hinge_loss(X, positive=True):
        if positive:
            return torch.relu(1-X).mean()
        else:
            return torch.relu(X+1).mean()

    @classmethod
    def get_hinge_loss(cls, Di, label):
        L_adv = 0
        for di in Di:
            L_adv += cls.hinge_loss(di[0], label)
        return L_adv

    @classmethod
    def get_VGG_loss(cls, a, b):
        if not hasattr(cls, 'vgg'):
            cls.vgg = VGGPerceptualLoss().eval().to("cuda")
        return cls.vgg(a,b)

    def get_BCE_loss(logits, target):
        assert target in [1, 0]
        targets = torch.full_like(logits, fill_value=target)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss

    @classmethod
    def get_cycle_loss(cls, args, a, b):
        L_swap_cycle = 0.0

        if args.W_cycle_Lab:
            L_swap_cycle_lab = args.W_cycle_Lab * cls.get_Lab_loss(a,b)
            L_swap_cycle += L_swap_cycle_lab

        if args.W_cycle_FM:
            L_swap_cycle_fm = args.W_cycle_FM * cls.get_FM_loss(a,b)
            L_swap_cycle += L_swap_cycle_fm

        if args.W_cycle_VGG:
            L_swap_cycle_VGG = args.W_cycle_VGG * cls.get_VGG_loss(a,b)
            L_swap_cycle += L_swap_cycle_VGG

        # save in dict
        # cls.loss_dict["L_swap_cycle"] = round(L_swap_cycle.item(), 4)
        # cls.loss_dict["L_swap_cycle_lab"] = round(L_swap_cycle_lab.item(), 4)
        # cls.loss_dict["L_swap_cycle_fm"] = round(L_swap_cycle_fm.item(), 4)
        # cls.loss_dict["L_swap_cycle_VGG"] = round(L_swap_cycle_VGG.item(), 4)
        return L_swap_cycle

    @classmethod
    def get_swap_loss(cls, args, a, b):
        L_swap = 0.0
        if args.W_swap_geo:
            L_swap_geo = args.W_swap_geo * cls.get_geo_loss(a, b)
            L_swap += L_swap_geo

        if args.W_swap_cycle:
            L_swap_cycle = args.W_swap_cycle * cls.get_cycle_loss(args, a, b)
            L_swap += L_swap_cycle
        
        # save in dict
        # cls.loss_dict["L_swap"] = round(L_swap.item(), 4)
        # cls.loss_dict["L_swap_geo"] = round(L_swap_geo.item(), 4)
        # cls.loss_dict["L_swap_cycle"] = round(L_swap_cycle.item(), 4)
        return L_swap

