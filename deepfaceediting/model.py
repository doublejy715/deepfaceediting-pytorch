import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from deepfaceediting.dataset import Dataset
from deepfaceediting.deepfaceediting_part import DFE_generator
from deepfaceediting.loss import DeepfaceeditingLoss
from deepfaceediting.checkpoint import load_checkpoint, save_checkpoint

from lib.utils import setup_ddp, update_net, save_image
from lib.model_interface import ModelInterface

from submodel.discriminator import StarGANv2Discriminator

class Deepfaceediting(ModelInterface):
    def __init__(self, args, gpu):
        super(Deepfaceediting, self).__init__(args, gpu)

    def set_dataset(self):
        self.dataset = Dataset(self.args.dataset, self.args.isMaster)

    def set_data_iterator(self):
        sampler = torch.utils.data.distributed.DistributedSampler(self.dataset) if self.args.use_mGPU else None
        self.dataloader = DataLoader(self.dataset, batch_size=self.args.batch_size, pin_memory=True, sampler=sampler, num_workers=8, drop_last=True)
        self.iterator = iter(self.dataloader)

    def load_next_batch(self):
        try:
            I_s, I_t = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            I_s, I_t = next(self.iterator)
            
        I_s, I_t = I_s.to(self.gpu), I_t.to(self.gpu)
        return I_s, I_t

    def initialize_models(self):
        self.LD_G = DFE_generator().cuda(self.gpu).train()
        self.LD_D = StarGANv2Discriminator().cuda(self.gpu).train()

    def set_multi_GPU(self):
        setup_ddp(self.gpu, self.args.gpu_num)

        self.LD_G = torch.nn.parallel.DistributedDataParallel(self.LD_G, device_ids=[self.gpu], broadcast_buffers=False, find_unused_parameters=True).module
        self.LD_D = torch.nn.parallel.DistributedDataParallel(self.LD_D, device_ids=[self.gpu], broadcast_buffers=False, find_unused_parameters=True).module
    
    def load_checkpoint(self):
        load_checkpoint(self.args, LD_G = self.LD_G, LD_D = self.LD_D, LD_opt_G = self.opt_G, LD_opt_D = self.opt_D)

    def set_optimizers(self):
        self.opt_G = optim.Adam(self.LD_G.parameters(), lr=self.args.lr_G, betas=(self.args.beta1, 0.999))
        self.opt_D = optim.Adam(self.LD_D.parameters(), lr=self.args.lr_D, betas=(self.args.beta1, 0.999))

    def set_loss_collector(self):
        self._loss_collector = DeepfaceeditingLoss(self.args)

    @property
    def loss_collector(self):
        return self._loss_collector

    def train_step(self):
        I_source, I_target = self.load_next_batch()

        ###########
        #  train  #
        ###########
        # generate mixed img
        I_result, I_target_feature = self.LD_G(I_source, I_target)
        I_target_sketch = self.LD_G.get_sketch_from_image(I_target)

        # recon geometry image
        I_target_adain_params = self.LD_G.Style_E(I_target)
        I_target_recon = self.LD_G.Local_G(I_target_feature, I_target_adain_params)

        # recon appear image
        I_source_feature = self.LD_G.Image_E(I_source)
        I_result_adain_params = self.LD_G.Style_E(I_result)
        I_source_recon = self.LD_G.Local_G(I_source_feature, I_result_adain_params)

        # D
        g_target = self.LD_D(I_target)
        g_source = self.LD_D(I_source)
        g_target_recon = self.LD_D(I_target_recon)
        g_source_recon = self.LD_D(I_source_recon)
        g_result = self.LD_D(I_result)

        G_dict = {
            "I_target": I_target,
            "I_source": I_source,
            "I_target_recon": I_target_recon,
            "I_source_recon": I_source_recon,
            "g_result": g_result,
        }

        # get G loss
        loss_G = self.loss_collector.get_loss_G(G_dict)
        update_net(self.opt_G,loss_G)

        d_target = self.LD_D(I_target)
        d_source = self.LD_D(I_source)
        d_target_recon = self.LD_D(I_target_recon.detach())
        d_source_recon = self.LD_D(I_source_recon.detach())
        d_result = self.LD_D(I_result.detach())

        D_dict = {
            "d_target": d_target,
            "d_result": d_result,
            "d_source": d_source,
            "d_target_recon" : d_target_recon,
            "d_source_recon" : d_source_recon
        }

        # get D loss
        loss_D = self.loss_collector.get_loss_D(D_dict)
        update_net(self.opt_D,loss_D)

        return [I_source, I_source_recon, I_target, (I_target_sketch-0.5)*2, I_target_recon, I_result]

    def save_image(self, results, step):
        save_image(self.args, step, 'img', results)

    def save_checkpoint(self, args, step):
        save_checkpoint(args, step, LD_G = self.LD_G, LD_D = self.LD_D, LD_opt_G = self.opt_G, LD_opt_D = self.opt_D)

