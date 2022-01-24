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
from nets.encoder import Sketch_Encoder_Part, Image_Encoder_Part
from nets.decoder import Sketch_Decoder_Part

from core.dataset import Img_Encoder_Dataset

def train(gpu, args): 
    # set gpu
    torch.cuda.set_device(gpu)

    # build model
    Sketch_E = Sketch_Encoder_Part(1,256).cuda(gpu).eval()
    Sketch_D = Sketch_Decoder_Part(256,1).cuda(gpu).eval()


    Image_E = Image_Encoder_Part(3,256).cuda(gpu).train()
    Image_D = Sketch_Decoder_Part(256,1).cuda(gpu).eval()
    
    # load and initialize the optimizer
    opt = optim.Adam(Image_E.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # load checkpoint
    ckptio = ckptIO(args)
    ckptio.img_encoder_load_ckpt_at1st(Sketch_E, Sketch_D, Image_D)
    
    # build a dataset
    train_set = Img_Encoder_Dataset(f"{args.dataset}/train")
    #test_set = Img_Encoder_Dataset(f"{args.dataset}/test")

    train_sampler = None
    #test_sampler = None

    if args.use_mGPU:
        args.isMaster = gpu==0

        # DDP setup
        utils.setup_ddp(gpu, args.gpu_num)

        # Distributed Data Parallel
        Sketch_E = torch.nn.parallel.DistributedDataParallel(Sketch_E, device_ids=[gpu], broadcast_buffers=False, find_unused_parameters=True).module
        Image_E = torch.nn.parallel.DistributedDataParallel(Image_E, device_ids=[gpu], broadcast_buffers=False, find_unused_parameters=True).module
        Sketch_D = torch.nn.parallel.DistributedDataParallel(Sketch_D, device_ids=[gpu]).module
        Image_D = torch.nn.parallel.DistributedDataParallel(Image_D, device_ids=[gpu]).module

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
        global_step += 1
        try:
            image_real, sketch_real = next(training_batch_iterator)
        except StopIteration:
            training_batch_iterator = iter(training_data_loader)
            image_real, sketch_real = next(training_batch_iterator)
        
        # transfer data to the gpus
        image_real, sketch_real = image_real.to(gpu), sketch_real.to(gpu)

        ###########
        #  train  #
        ###########
        sktech_feature = Sketch_E(sketch_real)
        sketch_recon, sketch_ftmap_layers = Sketch_D(sktech_feature)

        image_feature = Image_E(image_real)
        image_recon, image_ftmap_layers = Image_D(image_feature)

        loss = loss_collector.get_img_encoder_loss(sketch_ftmap_layers, image_ftmap_layers)

        utils.update_net(opt, loss)

        # # log and print loss
        if args.isMaster and global_step % args.loss_cycle==0:
            
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
            utils.save_img(args, global_step, "imgs", [sketch_recon, image_recon])

        # save ckpt
        if global_step % args.ckpt_cycle == 0:
            ckptio.img_encoder_save_ckpt(global_step, Sketch_E, Image_E, Sketch_D, opt)

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