import os
import torch
  
def load_checkpoint(args,LD_G=None,LD_D=None,LD_opt_G=None,LD_opt_D=None):    
    ckpt_path = f'{args.save_root}/{args.ckpt_id}/ckpt/latest.pt'

    try:
        ckpt = torch.load(ckpt_path, map_location=torch.device('cuda'))
        None if LD_G is None else LD_G.Local_G.load_state_dict(ckpt["LD_G"], strict=False)
        None if LD_D is None else LD_D.load_state_dict(ckpt["LD_D"], strict=False)
        None if LD_opt_G is None else LD_opt_G.load_state_dict(ckpt["LD_opt_G"])
        None if LD_opt_D is None else LD_opt_D.load_state_dict(ckpt["LD_opt_D"])

    except:
        if args.isMaster:
            print(f"Failed to load checkpoint.")
        return 0

def save_checkpoint(args, global_step,LD_G=None,LD_D=None,LD_opt_G=None,LD_opt_D=None):
    try:
        ckpt_path = f'{args.save_root}/{args.ckpt_id}/ckpt/latest.pt'
        ckpt = torch.load(ckpt_path, map_location=torch.device('cuda'))
    except Exception as e:
        ckpt = {
            "LD_G" : None,
            "LD_D" : None,
            "LD_opt_G" : None,
            "LD_opt_D" : None
        }
        print(e)
        print("set all model parameter to None")

    os.makedirs(f'{args.save_root}/{args.run_id}/ckpt', exist_ok=True)
    ckpt_dict = {
        "LD_G": ckpt["LD_G"] if LD_G is None else LD_G.state_dict(),
        "LD_D": ckpt["LD_D"] if LD_D is None else LD_D.state_dict(),
        "LD_opt_G": ckpt["LD_opt_G"] if LD_opt_G is None else LD_opt_G.state_dict(),
        "LD_opt_D": ckpt["LD_opt_D"] if LD_opt_D is None else LD_opt_D.state_dict()
    }
    ckpt_path = f'{args.save_root}/{args.run_id}/ckpt/{global_step}.pt'
    torch.save(ckpt_dict, ckpt_path)

    ckpt_path_latest = f'{args.save_root}/{args.run_id}/ckpt/latest.pt'
    torch.save(ckpt_dict, ckpt_path_latest)
    