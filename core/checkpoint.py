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

    # step 1 
    def geometry_load_ckpt(self, Sketch_E, Sketch_D, opt):
        try:
            # set path
            ckpt_path = f'{self.args.save_root}/{self.args.ckpt_id}/ckpt/latest.pt'
            
            # load ckpt
            ckpt = torch.load(ckpt_path, map_location=torch.device('cuda'))
            
            # load state dict
            Sketch_E.load_state_dict(ckpt["Sketch_E"], strict=False)
            Sketch_D.load_state_dict(ckpt["Sketch_D"], strict=False)
            opt.load_state_dict(ckpt["opt"], strict=False)

        except Exception as e:
            print(e)

    def geometry_save_ckpt(self, global_step, Sketch_E, Sketch_D, opt):
        os.makedirs(f'{self.args.save_root}/{self.args.run_id}/ckpt', exist_ok=True)

        ckpt_dict = {
            "Sketch_E": Sketch_E.state_dict(),
            "Sketch_D": Sketch_D.state_dict(),
            "opt": opt.state_dict()
        }

        ckpt_path = f'{self.args.save_root}/{self.args.run_id}/ckpt/{global_step}.pt'
        torch.save(ckpt_dict, ckpt_path)

        ckpt_path_latest = f'{self.args.save_root}/{self.args.run_id}/ckpt/latest.pt'
        torch.save(ckpt_dict, ckpt_path_latest)

    # step 2
    def img_encoder_load_ckpt(self, first_train, Sketch_E, Image_E, Sketch_D, Image_D, opt):
        if first_train:
            ckpt_path = f'{self.args.save_root}/{self.args.ckpt_id}/ckpt/latest.pt'
            ckpt = torch.load(ckpt_path, map_location=torch.device('cuda'))

            Sketch_E.load_state_dict(ckpt["Sketch_E"], strict=False)
            Sketch_D.load_state_dict(ckpt["Sketch_D"], strict=False)
            Image_D.load_state_dict(ckpt["Sketch_D"], strict=False)

        else:
            # set path
            ckpt_path = f'{self.args.save_root}/{self.args.ckpt_id}/ckpt/latest.pt'
            
            # load ckpt
            ckpt = torch.load(ckpt_path, map_location=torch.device('cuda'))
            
            # load state dict
            Sketch_E.load_state_dict(ckpt["Sketch_E"], strict=False)
            Image_E.load_state_dict(ckpt["Image_E"], strict=False)
            Sketch_D.load_state_dict(ckpt["Sketch_D"], strict=False)
            Image_D.load_state_dict(ckpt["Sketch_D"], strict=False)
            opt.load_state_dict(ckpt["opt"])

    def img_encoder_save_ckpt(self, global_step, Sketch_E, Image_E, Sketch_D, opt):
        os.makedirs(f'{self.args.save_root}/{self.args.run_id}/ckpt', exist_ok=True)

        ckpt_dict = {
            "Sketch_E": Sketch_E.state_dict(),
            "Image_E": Image_E.state_dict(),
            "Sketch_D": Sketch_D.state_dict(),
            "opt": opt.state_dict()
        }

        ckpt_path = f'{self.args.save_root}/{self.args.run_id}/ckpt/{str(global_step).zfill(8)}.pt'
        torch.save(ckpt_dict, ckpt_path)

        ckpt_path_latest = f'{self.args.save_root}/{self.args.run_id}/ckpt/latest.pt'
        torch.save(ckpt_dict, ckpt_path_latest)

    # step 3
    def LD_module_load_ckpt(self, first_train, Image_E, Style_E, LD_G, LD_D, opt_G, opt_D):
        if first_train:
            # set path
            ckpt_path = f'{self.args.save_root}/{self.args.ckpt_id}/ckpt/latest.pt'
            
            # load ckpt
            ckpt = torch.load(ckpt_path, map_location=torch.device('cuda'))
            
            # load state dict
            Image_E.load_state_dict(ckpt["Image_E"], strict=False)

        else:
            # set path
            ckpt_path = f'{self.args.save_root}/{self.args.ckpt_id}/ckpt/latest.pt'
            
            # load ckpt
            ckpt = torch.load(ckpt_path, map_location=torch.device('cuda'))
            
            # load state dict
            Image_E.load_state_dict(ckpt["Image_E"], strict=False)
            Style_E.load_state_dict(ckpt["Style_E"], strict=False)
            LD_G.load_state_dict(ckpt["LD_G"], strict=False)
            LD_D.load_state_dict(ckpt["LD_D"], strict=False)
            opt_G.load_state_dict(ckpt["opt_G"], strict=False)
            opt_D.load_state_dict(ckpt["opt_D"], strict=False)


    def LD_module_save_ckpt(self, global_step, Image_E, Style_E, LD_G, LD_D, opt_G, opt_D):
        os.makedirs(f'{self.args.save_root}/{self.args.run_id}/ckpt', exist_ok=True)

        ckpt_dict = {
            "Image_E": Image_E.state_dict(),
            "Style_E": Style_E.state_dict(),
            "LD_G": LD_G.state_dict(),
            "LD_D": LD_D.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict()
        }

        ckpt_path = f'{self.args.save_root}/{self.args.run_id}/ckpt/{str(global_step).zfill(8)}.pt'
        torch.save(ckpt_dict, ckpt_path)

        ckpt_path_latest = f'{self.args.save_root}/{self.args.run_id}/ckpt/latest.pt'
        torch.save(ckpt_dict, ckpt_path_latest)


    # step 4
    def GF_module_load_ckpt(self, first_train, Image_E, Style_E, LD_G, GF_G, D, opt_G, opt_D):
        if first_train:
            ckpt_path = f'{self.args.save_root}/{self.args.ckpt_id}/ckpt/latest.pt'
            
            ckpt = torch.load(ckpt_path, map_location=torch.device('cuda'))

            Image_E.load_state_dict(ckpt["Image_E"], strict=False)
            Style_E.load_state_dict(ckpt["Style_E"], strict=False)
            LD_G.load_state_dict(ckpt["LD_G"], strict=False)

        else:
            # set path
            ckpt_path = f'{self.args.save_root}/{self.args.ckpt_id}/ckpt/latest.pt'
            
            # load ckpt
            ckpt = torch.load(ckpt_path, map_location=torch.device('cuda'))
            
            # load state dict
            Image_E.load_state_dict(ckpt["Image_E"], strict=False)
            Style_E.load_state_dict(ckpt["Style_E"], strict=False)
            LD_G.load_state_dict(ckpt["LD_G"], strict=False)
            GF_G.load_state_dict(ckpt["GF_G"], strict=False)
            D.load_state_dict(ckpt["D"], strict=False)
            opt_G.load_state_dict(ckpt["opt_G"], strict=False)
            opt_D.load_state_dict(ckpt["opt_D"], strict=False)

    def GF_module_save_ckpt(self, global_step, Image_E, Style_E, L_G, G_G, D, opt_G, opt_D):
        os.makedirs(f'{self.args.save_root}/{self.args.run_id}/ckpt', exist_ok=True)

        ckpt_dict = {
            "Image_E": Image_E.state_dict(),
            "Style_E": Style_E.state_dict(),
            "L_G": L_G.state_dict(),
            "G_G": G_G.state_dict(),
            "D": D.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict()
        }

        ckpt_path = f'{self.args.save_root}/{self.args.run_id}/ckpt/{str({global_step}).zfill(8)}.pt'
        torch.save(ckpt_dict, ckpt_path)

        ckpt_path_latest = f'{self.args.save_root}/{self.args.run_id}/ckpt/latest.pt'
        torch.save(ckpt_dict, ckpt_path_latest)