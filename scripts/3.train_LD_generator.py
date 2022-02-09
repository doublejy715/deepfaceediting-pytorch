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
from core.dataset import Dataset
from opts.train_options import train_options
from nets.encoder import Image_Encoder, Style_Encoder
from nets.generator import Local_G
from nets.decoder import Sketch_Decoder
from nets.discriminator import MultiscaleDiscriminator
from utils.nets_utils import assign_adain_params

def train(gpu, args): 
    # set gpu
    torch.cuda.set_device(gpu)
    args.gpu_id = gpu

    # build models
    LD_G = Local_G(256,3).cuda(gpu).train()
    LD_D = MultiscaleDiscriminator(3).cuda(gpu).train()
    Image_E = Image_Encoder(3,256).cuda(gpu).eval()
    Style_E = Style_Encoder(3, LD_G.style_dim).cuda(gpu).train()
    Sketch_D = Sketch_Decoder(256,1).cuda(gpu).eval()
    

    for params in Image_E.parameters():
        params.requires_grad = False
        
    # build a dataset
    train_set = Dataset(f"{args.dataset}/train")
    #test_set = Dataset(f"{args.dataset}/test")

    # load and initialize the optimizer
    # opt_G에 들어갈 것 : G 학습 과정 중 포함되는 모든 네트워크
    opt_G = optim.Adam([*LD_G.parameters(), *Style_E.parameters()], lr=args.lr_G, betas=(args.beta1, 0.999))
    opt_D = optim.Adam(LD_D.parameters(), lr=args.lr_D, betas=(args.beta1, 0.999))

    train_sampler = None
    #test_sampler = None

    if args.use_mGPU:
        args.isMaster = gpu==0

        # DDP setup
        utils.setup_ddp(gpu, args.gpu_num)

        # Distributed Data Parallel
        Style_E = torch.nn.parallel.DistributedDataParallel(Style_E, device_ids=[gpu], broadcast_buffers=False, find_unused_parameters=True).module
        LD_G = torch.nn.parallel.DistributedDataParallel(LD_G, device_ids=[gpu], broadcast_buffers=False, find_unused_parameters=True).module
        LD_D = torch.nn.parallel.DistributedDataParallel(LD_D, device_ids=[gpu], broadcast_buffers=False, find_unused_parameters=True).module
        Sketch_D = torch.nn.parallel.DistributedDataParallel(Sketch_D, device_ids=[gpu]).module

        # make sampler 
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        #test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)

    # build a dataloader
    train_data_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,sampler=train_sampler, pin_memory=True, num_workers=args.num_works,drop_last=True)
    #test_data_loader = DataLoader(dataset=test_set, batch_size=args.test_batch_size,sampler=test_sampler, num_workers=args.num_works,drop_last=True)
    
    # load checkpoint
    ckptio = ckptIO(args)
    ckptio.load_ckpt(image_E = Image_E, sketch_D = Sketch_D, style_E = Style_E, LD_G = LD_G, LD_D = LD_D, LD_opt_G = opt_G, LD_opt_D = opt_D)

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
            I_s, I_t = next(train_batch_iterator)
        except StopIteration:
            train_batch_iterator = iter(train_data_loader)
            I_s, I_t = next(train_batch_iterator)
        
        # transfer data to the gpus
        I_s, I_t = I_s.to(gpu), I_t.to(gpu)
        
        ###########
        #  train  #
        ###########
        # train step 1. G 관련 

        # run G : True gradient
        # generate mixed img
        I_t_feature = Image_E(I_t)
        I_t_sketch, _ = Sketch_D(I_t_feature)
        I_s_adain_params = Style_E(I_s)
        I_r = LD_G.rgb_forward(I_t_feature, I_s_adain_params)

        # recon geometry image
        I_t_adain_params = Style_E(I_t)
        I_t_recon = LD_G.rgb_forward(I_t_feature, I_t_adain_params)

        # recon appear image
        I_s_feature = Image_E(I_s)
        I_r_adain_params = Style_E(I_r)
        I_s_recon = LD_G.rgb_forward(I_s_feature, I_r_adain_params)

        # D
        g_I_t = LD_D(I_t)
        g_I_s = LD_D(I_s)
        g_I_t_recon = LD_D(I_t_recon)
        g_I_s_recon = LD_D(I_s_recon)
        g_I_r = LD_D(I_r)

        # G loss
        loss_G = loss_collector.get_loss_G(I_t, I_s, I_t_recon, I_s_recon, g_I_t, g_I_s, g_I_t_recon, g_I_s_recon, g_I_r)
        utils.update_net(opt_G, loss_G)

        # run D : False gradient (use .detach())
        # 앞에서 같은 과정이 있어도 .detach()로 다시 해준다. loss_D에 들어가는 모든 것은 loss_G와 분리 시켜 줘야한다.
        d_I_t = LD_D(I_t)
        d_I_s = LD_D(I_s)
        d_I_t_recon = LD_D(I_t_recon.detach())
        d_I_s_recon = LD_D(I_s_recon.detach())
        d_I_r = LD_D(I_r.detach())
        
        # D loss
        loss_D = loss_collector.get_loss_D(d_I_t, d_I_s, d_I_t_recon, d_I_s_recon, d_I_r)
        utils.update_net(opt_D, loss_D)

        # # log and print loss
        if args.isMaster and global_step % args.loss_cycle==0:
        #     # log loss on wandb
        #     wandb.log(loss_collector.loss_dict)
            loss_collector.print_loss(global_step)

        # save image
        if args.isMaster and global_step % args.test_cycle == 0:
            # try:
            #     test_I_s, test_I_t = next(test_batch_iterator)
            # except StopIteration:
            #     test_batch_iterator = iter(test_data_loader)
            #     test_I_s, test_I_t = next(test_batch_iterator)

            
            utils.save_img(args, global_step, "imgs", [I_s, I_s_recon, I_t, I_t_sketch, I_t_recon, I_r])

        # save ckpt
        if global_step % args.ckpt_cycle == 0:
            ckptio.save_ckpt(global_step, image_E = Image_E, sketch_D = Sketch_D, style_E = Style_E, LD_G = LD_G, LD_D = LD_D, LD_opt_G = opt_G, LD_opt_D = opt_D)

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