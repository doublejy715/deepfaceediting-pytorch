
import argparse

def train_options():
    parser = argparse.ArgumentParser(description='Deepfaceediting-pytorch-implementation')
    
    # ids
    parser.add_argument('--gpu_id', type=int, default=0) 
    parser.add_argument('--project_id', type=str, default="deepfaceediting")
    parser.add_argument('--run_id', type=str, required=True) 
    parser.add_argument('--ckpt_id', type=str, default='')

    # dataset
    parser.add_argument("--dataset", type=str, default = "./datasets/train", help = "the path of geometry image")
    
    # log
    parser.add_argument('--loss_cycle', type=str, default=10)
    parser.add_argument('--test_cycle', type=str, default=100)
    parser.add_argument('--ckpt_cycle', type=str, default=5000)
    parser.add_argument('--save_root', type=str, default="train")

    # hyperparameters
    parser.add_argument('--batch_size', type=str, default=8)
    parser.add_argument('--test_batch_size', type=str, default=2)
    parser.add_argument('--max_step', type=str, default=200000)

    # weights
    # LD
    # Generator
    # Recon
    parser.add_argument('--W_recon', type=int, default=1)
    parser.add_argument('--W_recon_Lab', type=int, default=0)
    parser.add_argument('--W_recon_FM', type=int, default=0)
    parser.add_argument('--W_recon_VGG', type=int, default=10)

    # Cycle
    parser.add_argument('--W_cycle', type=int, default=1)
    parser.add_argument('--W_cycle_Lab', type=int, default=0)
    parser.add_argument('--W_cycle_FM', type=int, default=0)
    parser.add_argument('--W_cycle_VGG', type=int, default=10)

    # Swap
    parser.add_argument('--W_swap', type=int, default=1)
    parser.add_argument('--W_swap_geo', type=int, default=1)
    parser.add_argument('--W_swap_cycle', type=int, default=1)

    # adv
    parser.add_argument('--W_adv', type=int, default=5)
    parser.add_argument('--W_adv_geo', type=int, default=0.5)
    parser.add_argument('--W_adv_app', type=int, default=0.5)
    parser.add_argument('--W_adv_recon_geo', type=int, default=0.33)
    parser.add_argument('--W_adv_recon_app', type=int, default=0.33)
    parser.add_argument('--W_adv_mix', type=int, default=0.33)

    # Discriminator
    # adv
    parser.add_argument('--W_D', type=int, default=1)
    parser.add_argument('--W_D_geo', type=int, default=0.5)
    parser.add_argument('--W_D_app', type=int, default=0.5)
    parser.add_argument('--W_D_recon_geo', type=int, default=0.33)
    parser.add_argument('--W_D_recon_app', type=int, default=0.33)
    parser.add_argument('--W_D_mix', type=int, default=0.33)


    # learning rate
    parser.add_argument('--lr', type=str, default=4e-4)
    parser.add_argument('--lr_G', type=str, default=4e-4)
    parser.add_argument('--lr_D', type=str, default=1e-5)
    parser.add_argument('--beta1', type=str, default=0)

    # multi GPU
    parser.add_argument('--isMaster', default=False)
    parser.add_argument('--use_mGPU', action='store_true')

    # use wandb
    parser.add_argument('--use_wandb', action='store_true')

    # etc
    parser.add_argument('--num_works', type=int, default=16, help='number of threads for data loader to use')

    return parser.parse_args()