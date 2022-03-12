import time
from lib.loss_interface import LossInterface
from lib.loss_interface import Loss

class DeepfaceeditingLoss(LossInterface):
    def __init__(self, args):
        super(DeepfaceeditingLoss,self).__init__(args)

    def get_loss_G(self, G_dict):
        L_G = 0.0
        if self.args.W_recon:
            L_recon = Loss.get_cycle_loss(self.args, G_dict["I_target"], G_dict["I_target_recon"])
            L_G += self.args.W_recon * L_recon

        if self.args.W_swap:
            L_swap = Loss.get_swap_loss(self.args, G_dict["I_source"], G_dict["I_source_recon"])
            L_G += self.args.W_swap * L_swap

        if self.args.W_adv:
            L_adv = Loss.get_hinge_loss(G_dict["g_result"], True)
            L_G += self.args.W_adv * L_adv

        # save in dict
        self.loss_dict["L_G"] = round(L_G.item(), 4)
        self.loss_dict["L_recon"] = round(L_recon.item(), 4)
        self.loss_dict["L_swap"] = round(L_swap.item(), 4)
        self.loss_dict["L_adv"] = round(L_adv.item(), 4)

        return L_G

    def get_loss_D(self, D_dict):
        # L_real = Loss.get_hinge_loss(D_dict["d_target"], True)
        # L_fake = Loss.get_hinge_loss(D_dict["d_result"], False)
        # L_D_real_t = Loss.get_BCE_loss(D_dict["d_target"], True)
        # L_D_real_s = Loss.get_BCE_loss(D_dict["d_source"], True)
        # L_D_fake = Loss.get_BCE_loss(D_dict["d_result"], False)
        # L_D_fake_t_recon = Loss.get_BCE_loss(D_dict["d_target_recon"], False)
        # L_D_fake_s_recon = Loss.get_BCE_loss(D_dict["d_source_recon"], False)
        L_D_real_t = Loss.get_hinge_loss(D_dict["d_target"], True)
        L_D_real_s = Loss.get_hinge_loss(D_dict["d_source"], True)
        L_D_fake = Loss.get_hinge_loss(D_dict["d_result"], False)
        # L_D_fake_t_recon = Loss.get_hinge_loss(D_dict["d_target_recon"], False)
        # L_D_fake_s_recon = Loss.get_hinge_loss(D_dict["d_source_recon"], False)
        
        # L_reg = Loss.get_r1_reg(L_D_real, D_dict["I_target"])
        L_D = (L_D_real_t + L_D_real_s + L_D_fake)*0.3333# + L_D_fake_t_recon + L_D_fake_s_recon# + L_reg

        # L_D = self.args.W_adv * (L_real + L_fake) * 0.5
        self.loss_dict["L_D"] = round(L_D.item(), 4)

        return L_D

        
    def print_loss(self, global_step):
        seconds = int(time.time() - self.start_time)

        print("")
        print(f"[ {seconds//3600//24:02}d {(seconds//3600)%24:02}h {(seconds//60)%60:02}m {seconds%60:02}s ]")
        print(f'steps: {global_step:06} / {self.args.max_step}')
        # print(f'lossG: {self.loss_dict["L_G"]}')
        print(f'lossD: {self.loss_dict["L_D"]} | lossG: {self.loss_dict["L_G"]}')
