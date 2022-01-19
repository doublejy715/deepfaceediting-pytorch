
import argparse

def train_options():
    parser = argparse.ArgumentParser(description='Deepfaceediting-pytorch-implementation')
    
    # ids
    parser.add_argument('--gpu_id', type=int, default=0) 
    parser.add_argument('--project_id', type=str, default="deepfaceediting")
    parser.add_argument('--run_id', type=str, required=True) 
    parser.add_argument('--ckpt_id', type=str, default='geometery')

    # dataset
    parser.add_argument("--dataset", type=str, default = "datasets/sketch", help = "the path of geometry image")
    
    # log
    parser.add_argument('--loss_cycle', type=str, default=10)
    parser.add_argument('--test_cycle', type=str, default=1000)
    parser.add_argument('--ckpt_cycle', type=str, default=10000)
    parser.add_argument('--save_root', type=str, default="training_result")

    # hyperparameters
    parser.add_argument('--batch_size', type=str, default=1)
    parser.add_argument('--test_batch_size', type=str, default=1)
    parser.add_argument('--max_step', type=str, default=200000)
    

    # learning rate
    parser.add_argument('--lr', type=str, default=4e-4)
    parser.add_argument('--beta1', type=str, default=0.5)

    # multi GPU
    parser.add_argument('--isMaster', default=False)
    parser.add_argument('--use_mGPU', action='store_false')

    # use wandb
    parser.add_argument('--use_wandb', action='store_false')

    # etc
    parser.add_argument('--num_works', type=int, default=16, help='number of threads for data loader to use')

    return parser.parse_args()