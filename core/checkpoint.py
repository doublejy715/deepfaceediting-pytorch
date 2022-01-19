import torch
import os

parts_key = ['bg','eye1','eye2','nose','mouth']

class ckptIO():
    def __init__(self, args):
        super(ckptIO, self).__init__()
        self.args = args

    def test_load_ckpt(self,model):
        ckpt_path = f'{self.args.save_root}/{self.args.ckpt_id}/ckpt/latest.pt'
        ckpt = torch.load(ckpt_path, map_location=torch.device('cuda'))
        
        # load ckpt to part model
        for key in parts_key:
            model.sketch_encoder_part[key].load_state_dict(ckpt[f'sketch_encoder_part_{key}'], strict=False)
            model.image_encoder_part[key].load_state_dict(ckpt[f'image_encoder_part_{key}'], strict=False)
            model.local_gen_part[key].load_state_dict(ckpt[f'local_gen_part_{key}'], strict=False)

        model.global_gen.load_state_dict(ckpt[f'global_gen'], strict=False)

    def load_ckpt(self, G, D, opt_G, opt_D):
        try:
            # set path
            ckpt_path = f'{self.args.save_root}/{self.args.ckpt_id}/ckpt/latest.pt'
            
            # load ckpt
            ckpt = torch.load(ckpt_path, map_location=torch.device('cuda'))
            
            # load state dict
            G.load_state_dict(ckpt["G"], strict=False)
            D.load_state_dict(ckpt["D"], strict=False)
            opt_G.load_state_dict(ckpt["opt_G"], strict=False)
            opt_D.load_state_dict(ckpt["opt_D"], strict=False)

        except Exception as e:
            print(e)

    def save_ckpt(self, global_step, G, D, opt):
        os.makedirs(f'{self.args.save_root}/{self.args.run_id}/ckpt', exist_ok=True)

        ckpt_dict = {
            "G": G.state_dict(),
            "D": D.state_dict(),
            "opt": opt.state_dict()
        }

        ckpt_path = f'{self.args.save_root}/{self.args.run_id}/ckpt/{global_step}.pt'
        torch.save(ckpt_dict, ckpt_path)

        ckpt_path_latest = f'{self.args.save_root}/{self.args.run_id}/ckpt/latest.pt'
        torch.save(ckpt_dict, ckpt_path_latest)
        
    def geometry_load_ckpt(self, E, D, opt):
        try:
            # set path
            ckpt_path = f'{self.args.save_root}/{self.args.ckpt_id}/ckpt/latest.pt'
            
            # load ckpt
            ckpt = torch.load(ckpt_path, map_location=torch.device('cuda'))
            
            # load state dict
            E.load_state_dict(ckpt["E"], strict=False)
            D.load_state_dict(ckpt["D"], strict=False)
            opt.load_state_dict(ckpt["opt"], strict=False)

        except Exception as e:
            print(e)


    def geometry_save_skpt(self, global_step, E, D, opt):
        os.makedirs(f'{self.args.save_root}/{self.args.run_id}/ckpt', exist_ok=True)

        ckpt_dict = {
            "E": E.state_dict(),
            "D": D.state_dict(),
            "opt": opt.state_dict()
        }

        ckpt_path = f'{self.args.save_root}/{self.args.run_id}/ckpt/{global_step}.pt'
        torch.save(ckpt_dict, ckpt_path)

        ckpt_path_latest = f'{self.args.save_root}/{self.args.run_id}/ckpt/latest.pt'
        torch.save(ckpt_dict, ckpt_path_latest)