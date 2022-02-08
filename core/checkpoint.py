import torch
import os

class ckptIO():
    def __init__(self, args):
        super(ckptIO, self).__init__()
        self.args = args

    def load_ckpt(self, sketch_E=None,sketch_D=None,sketch_opt=None,image_E=None, image_D=None, image_opt=None,style_E=None,LD_G=None,LD_D=None,LD_opt_G=None,LD_opt_D=None):
        try:
            ckpt_path = f'{self.args.save_root}/{self.args.ckpt_id}/ckpt/latest.pt'
            ckpt = torch.load(ckpt_path, map_location=torch.device('cuda'))
            
            None if sketch_E is None else sketch_E.load_state_dict(ckpt["sketch_E"], strict=False) 
            None if sketch_D is None else sketch_D.load_state_dict(ckpt["sketch_D"], strict=False)
            None if image_D is None else image_D.load_state_dict(ckpt["sketch_D"], strict=False)
            None if sketch_opt is None else sketch_opt.load_state_dict(ckpt["sketch_opt"])

            None if image_E is None else image_E.load_state_dict(ckpt["image_E"], strict=False)
            None if image_opt is None else image_opt.load_state_dict(ckpt["image_opt"])
            
            None if style_E is None else style_E.load_state_dict(ckpt["style_E"], strict=False)
            None if LD_G is None else LD_G.load_state_dict(ckpt["LD_G"], strict=False)
            None if LD_D is None else LD_D.load_state_dict(ckpt["LD_D"], strict=False)
            None if LD_opt_G is None else LD_opt_G.load_state_dict(ckpt["LD_opt_G"])
            None if LD_opt_D is None else LD_opt_D.load_state_dict(ckpt["LD_opt_D"])
        
        except Exception as e:
            print(e)
            

    def save_ckpt(self, global_step,sketch_E=None,sketch_D=None,sketch_opt=None,image_E=None,image_opt=None,style_E=None,LD_G=None,LD_D=None,LD_opt_G=None,LD_opt_D=None):
        try:
            ckpt_path = f'{self.args.save_root}/{self.args.ckpt_id}/ckpt/latest.pt'
            ckpt = torch.load(ckpt_path, map_location=torch.device('cuda'))
        except Exception as e:
            ckpt = {
                "sketch_E" : None,
                "sketch_D" : None,
                "sketch_opt" : None,
                "image_E" : None,
                "image_opt" : None,
                "style_E" : None,
                "image_opt" : None,
                "style_E" : None,
                "LD_G" : None,
                "LD_D" : None,
                "LD_opt_G" : None,
                "LD_opt_D" : None
            }
            print(e)
            print("set all model parameter to None")

        os.makedirs(f'{self.args.save_root}/{self.args.run_id}/ckpt', exist_ok=True)
        ckpt_dict = {
            "sketch_E": ckpt["sketch_E"] if sketch_E is None else sketch_E.state_dict(),
            "sketch_D": ckpt["sketch_D"] if sketch_D is None else sketch_D.state_dict(),
            "sketch_opt": ckpt["sketch_opt"] if sketch_opt is None else sketch_opt.state_dict(),
            "image_E": ckpt["image_E"] if image_E is None else image_E.state_dict(),
            "image_opt": ckpt["image_opt"] if image_opt is None else image_opt.state_dict(),
            "style_E": ckpt["style_E"] if style_E is None else style_E.state_dict(),
            "LD_G": ckpt["LD_G"] if LD_G is None else LD_G.state_dict(),
            "LD_D": ckpt["LD_D"] if LD_D is None else LD_D.state_dict(),
            "LD_opt_G": ckpt["LD_opt_G"] if LD_opt_G is None else LD_opt_G.state_dict(),
            "LD_opt_D": ckpt["LD_opt_D"] if LD_opt_D is None else LD_opt_D.state_dict()
        }

        ckpt_path = f'{self.args.save_root}/{self.args.run_id}/ckpt/{global_step}.pt'
        torch.save(ckpt_dict, ckpt_path)

        ckpt_path_latest = f'{self.args.save_root}/{self.args.run_id}/ckpt/latest.pt'
        torch.save(ckpt_dict, ckpt_path_latest)

