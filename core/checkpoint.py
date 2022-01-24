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

    def img_encoder_load_ckpt_at1st(self, E, D1, D2):
        try:
            # set path
            ckpt_path = f'{self.args.save_root}/{self.args.ckpt_id}/ckpt/latest.pt'
            
            # load ckpt
            ckpt = torch.load(ckpt_path, map_location=torch.device('cuda'))
            
            # load state dict
            E.load_state_dict(ckpt["E"], strict=False)
            D1.load_state_dict(ckpt["D"], strict=False)
            D2.load_state_dict(ckpt["D"], strict=False)

        except Exception as e:
            print(e)

    def img_encoder_load_ckpt(self, Sketch_E, Image_E, Sketch_D, Image_D, opt):
        try:
            # set path
            ckpt_path = f'{self.args.save_root}/{self.args.ckpt_id}/ckpt/latest.pt'
            
            # load ckpt
            ckpt = torch.load(ckpt_path, map_location=torch.device('cuda'))
            
            # load state dict
            Sketch_E.load_state_dict(ckpt["Sketch_E"], strict=False)
            Image_E.load_state_dict(ckpt["Image_E"], strict=False)
            Sketch_D.load_state_dict(ckpt["D"], strict=False)
            Image_D.load_state_dict(ckpt["D"], strict=False)
            opt.load_state_dict(ckpt["opt"], strict=False)


        except Exception as e:
            print(e)

    def img_encoder_save_ckpt(self, global_step, Sketch_E, Image_E, Sketch_D, opt):
        os.makedirs(f'{self.args.save_root}/{self.args.run_id}/ckpt', exist_ok=True)

        ckpt_dict = {
            "Sketch_E": Sketch_E.state_dict(),
            "Image_E": Image_E.state_dict(),
            "D": Sketch_D.state_dict(),
            "opt": opt.state_dict()
        }

        ckpt_path = f'{self.args.save_root}/{self.args.run_id}/ckpt/{str(global_step).zfill(8)}.pt'
        torch.save(ckpt_dict, ckpt_path)

        ckpt_path_latest = f'{self.args.save_root}/{self.args.run_id}/ckpt/latest.pt'
        torch.save(ckpt_dict, ckpt_path_latest)
