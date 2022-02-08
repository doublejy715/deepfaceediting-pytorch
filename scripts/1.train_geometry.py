import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
import sys
sys.path.append(os.getcwd())

from utils import utils
from core.checkpoint import ckptIO
from core.loss import lossCollector
from opts.train_options import train_options
from nets.encoder import Sketch_Encoder
from nets.decoder import Sketch_Decoder

from core.dataset import Sketch_Encoder_Dataset

def train(gpu, args): 
    # set gpu
    torch.cuda.set_device(gpu)

    # build model
    Sketch_E = Sketch_Encoder(1,256).cuda(gpu).train()
    Sketch_D = Sketch_Decoder(256,1).cuda(gpu).train()
    
    # load and initialize the optimizer
    opt = optim.Adam([*Sketch_E.parameters(),*Sketch_D.parameters()], lr=args.lr, betas=(args.beta1, 0.999))

    # load checkpoint
    ckptio = ckptIO(args)
    ckptio.load_ckpt(sketch_E = Sketch_E, sketch_D = Sketch_D, sketch_opt = opt)
    
    # build a dataset
    train_set = Sketch_Encoder_Dataset(f"{args.dataset}/train/")
    #test_set = Sketch_Encoder_Dataset(f"{args.dataset}/test")

    train_sampler = None
    #test_sampler = None

    if args.use_mGPU:
        args.isMaster = gpu==0

        # DDP setup
        utils.setup_ddp(gpu, args.gpu_num)

        # Distributed Data Parallel
        Sketch_E = torch.nn.parallel.DistributedDataParallel(Sketch_E, device_ids=[gpu], broadcast_buffers=False, find_unused_parameters=True).module
        Sketch_D = torch.nn.parallel.DistributedDataParallel(Sketch_D, device_ids=[gpu]).module

        # make sampler 
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        #test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)

    # build a dataloader
    train_data_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,sampler=train_sampler, num_workers=args.num_works,drop_last=True)
    #test_data_loader = DataLoader(dataset=test_set, batch_size=args.test_batch_size,sampler=test_sampler, num_workers=args.num_works,drop_last=True)

    train_batch_iterator = iter(train_data_loader)
    #test_batch_iterator = iter(test_data_loader)
    
    # build loss
    loss_collector = lossCollector(args)

    # initialize wandb
    # if args.isMaster:
    #     wandb.init(project=args.project_id, name=args.run_id)

    global_step = -1
    while global_step < args.max_step:
        global_step += 1
        try:
            I = next(train_batch_iterator)
        except StopIteration:
            train_batch_iterator = iter(train_data_loader)
            I = next(train_batch_iterator)
        
        # transfer data to the gpus
        I = I.to(gpu)

        ###########
        #  train  #
        ###########
        I_feature = Sketch_E(I)
        I_recon, _ = Sketch_D(I_feature)

        loss = loss_collector.get_L1_loss(I, I_recon)

        utils.update_net(opt, loss)

        # log and print loss
        if args.isMaster and global_step % args.loss_cycle==0:
            
        #     # log loss on wandb
        #     wandb.log(loss_collector.loss_dict)
            loss_collector.print_L1_loss(global_step)

        # save image
        if args.isMaster and global_step % args.test_cycle == 0:
            # try:
            #     test_I = next(test_batch_iterator)
            # except StopIteration:
            #     test_batch_iterator = iter(test_data_loader)
            #     test_I = next(test_batch_iterator)

            # test_I = test_I.to(gpu)
            # test_feature_map = E(test_I)
            # test_I_recon = D(test_feature_map)

            # loss_collector.get_L1_loss(test_I, test_I_recon,test=True)

            # utils.save_img(args, global_step, "imgs", [test_I, test_I_recon])
            utils.save_img(args, global_step, "imgs", [I, I_recon])

        # save ckpt
        if global_step % args.ckpt_cycle == 0:
            ckptio.save_ckpt(global_step, sketch_E = Sketch_E, sketch_D = Sketch_D, sketch_opt = opt)

if __name__ == "__main__":
    
    # get args
    args = train_options()

    # make train dir
    os.makedirs(args.save_root, exist_ok=True)

    # setup multi-GPUs env
    if args.use_mGPU:

        # get gpu number
        args.gpu_num = torch.cuda.device_count()
        
        # divide by gpu num
        args.batch_size = int(args.batch_size / args.gpu_num)

        # start multi-GPUs train
        torch.multiprocessing.spawn(train, nprocs=args.gpu_num, args=(args, ))

    # if use single-GPU
    else:
        # set isMaster
        args.isMaster = True
        # start single-GPU train
        train(args.gpu_id, args)