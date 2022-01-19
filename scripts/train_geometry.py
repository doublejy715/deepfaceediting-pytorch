import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
import sys
sys.path.append(os.getcwd())

from utils import utils
from core.checkpoint import ckptIO
from core.sketch_loss import lossCollector
from opts.train_options import train_options
from nets.encoder import Sketch_Encoder_Part
from nets.decoder import Sketch_Decoder_Part

from core.dataset import SketchDataset

def train(gpu, args): 
    # set gpu
    torch.cuda.set_device(gpu)

    # build model
    E = Sketch_Encoder_Part(3,256).cuda(gpu).train()
    D = Sketch_Decoder_Part(256,3).cuda(gpu).train()
    
    # load and initialize the optimizer
    opt = optim.Adam([*E.parameters(),*D.parameters()], lr=args.lr, betas=(args.beta1, 0.999))

    # load checkpoint
    ckptio = ckptIO(args)
    ckptio.geometry_load_ckpt(E, D, opt)
    
    # build a dataset
    train_set = SketchDataset(f"{args.dataset}/train")
    #test_set = SketchDataset(f"{args.dataset}/test")

    train_sampler = None
    #test_sampler = None

    if args.use_mGPU:
        args.isMaster = gpu==0

        # DDP setup
        utils.setup_ddp(gpu, args.gpu_num)

        # Distributed Data Parallel
        E = torch.nn.parallel.DistributedDataParallel(E, device_ids=[gpu], broadcast_buffers=False, find_unused_parameters=True).module
        D = torch.nn.parallel.DistributedDataParallel(D, device_ids=[gpu]).module

        # make sampler 
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        #test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)

    # build a dataloader
    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,sampler=train_sampler, num_workers=args.num_works,drop_last=True)
    #testing_data_loader = DataLoader(dataset=test_set, batch_size=args.test_batch_size,sampler=test_sampler, num_workers=args.num_works,drop_last=True)

    training_batch_iterator = iter(training_data_loader)
    #testing_batch_iterator = iter(testing_data_loader)
    
    # build loss
    loss_collector = lossCollector(args)

    # initialize wandb
    if args.isMaster:
        wandb.init(project=args.project_id, name=args.run_id)

    global_step = -1
    while global_step < args.max_step:
        print(global_step)
        global_step += 1
        try:
            sketch_real = next(training_batch_iterator)
        except StopIteration:
            training_batch_iterator = iter(training_data_loader)
            sketch_real = next(training_batch_iterator)
        
        # transfer data to the gpus
        sketch_real = sketch_real.to(gpu)

        ###########
        #  train  #
        ###########
        sktech_feature = E(sketch_real)
        sketch_recon = D(sktech_feature)

        loss = loss_collector.get_L1_loss(sketch_real, sketch_recon)

        utils.update_net(opt, loss)

        # log and print loss
        # if args.isMaster and global_step % args.loss_cycle==0:
            
            # log loss on wandb
            wandb.log(loss_collector.loss_dict)
            loss_collector.print_L1_loss(global_step)

        # save image
        if args.isMaster and global_step % args.test_cycle == 0:
            # try:
            #     test_sketch_real = next(testing_batch_iterator)
            # except StopIteration:
            #     testing_batch_iterator = iter(testing_data_loader)
            #     test_sketch_real = next(testing_batch_iterator)

            # test_sketch_real = test_sketch_real.to(gpu)
            # test_feature_map = E(test_sketch_real)
            # test_sketch_recon = D(test_feature_map)

            # loss_collector.get_L1_loss(test_sketch_real, test_sketch_recon,test=True)

            # utils.save_img(args, global_step, "imgs", [test_sketch_real, test_sketch_recon])
            utils.save_img(args, global_step, "imgs", [sketch_real, sketch_recon])

        # save ckpt
        if global_step % args.ckpt_cycle == 0:
            ckptio.geometry_save_skpt(global_step, E, D, opt)

if __name__ == "__main__":
    
    # get args
    args = train_options()

    # make training dir
    os.makedirs(args.save_root, exist_ok=True)

    # setup multi-GPUs env
    if args.use_mGPU:

        # get gpu number
        args.gpu_num = torch.cuda.device_count()
        
        # divide by gpu num
        args.batch_size = int(args.batch_size / args.gpu_num)

        # start multi-GPUs training
        torch.multiprocessing.spawn(train, nprocs=args.gpu_num, args=(args, ))

    # if use single-GPU
    else:
        # set isMaster
        args.isMaster = True
        # start single-GPU training
        train(args.gpu_id, args)