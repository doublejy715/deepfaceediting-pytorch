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
from core.dataset import LD_G_Dataset
from opts.train_options import train_options
from nets.encoder import Sketch_Encoder, Style_Encoder
from nets.generator import Local_G
from nets.discriminator import MultiscaleDiscriminator
from utils.nets_utils import assign_adain_params




def train(gpu, args): 
    # set gpu
    torch.cuda.set_device(gpu)

    # build models
    G = Local_G(256,3).cuda(gpu).train()
    D = MultiscaleDiscriminator(3).cuda(gpu).train()
    Sketch_E = Sketch_Encoder(1,256).cuda(gpu).eval()
    Style_E = Style_Encoder(3, G).cuda(gpu).train()

        
    # build a dataset
    train_set = LD_G_Dataset(f"{args.dataset}/train")
    #test_set = Img_Encoder_Dataset(f"{args.dataset}/test")

    # load and initialize the optimizer
    # opt_G에 들어갈 것 : G 학습 과정 중 포함되는 모든 네트워크
    opt_G = optim.Adam([*G.parameters(),*Style_E.parameters()], lr=args.lr_G, betas=(args.beta1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr_D, betas=(args.beta1, 0.999))

    train_sampler = None
    #test_sampler = None

    if args.use_mGPU:
        args.isMaster = gpu==0

        # DDP setup
        utils.setup_ddp(gpu, args.gpu_num)

        # Distributed Data Parallel
        Sketch_E = torch.nn.parallel.DistributedDataParallel(Sketch_E, device_ids=[gpu], broadcast_buffers=False, find_unused_parameters=True).module
        Style_E = torch.nn.parallel.DistributedDataParallel(Style_E, device_ids=[gpu], broadcast_buffers=False, find_unused_parameters=True).module
        G = torch.nn.parallel.DistributedDataParallel(G, device_ids=[gpu], broadcast_buffers=False, find_unused_parameters=True).module
        D = torch.nn.parallel.DistributedDataParallel(D, device_ids=[gpu], broadcast_buffers=False, find_unused_parameters=True).module

        # make sampler 
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        #test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)

    # build a dataloader
    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,sampler=train_sampler, num_workers=args.num_works,drop_last=True)
    #testing_data_loader = DataLoader(dataset=test_set, batch_size=args.test_batch_size,sampler=test_sampler, num_workers=args.num_works,drop_last=True)
    
    # load checkpoint
    ckptio = ckptIO(args)
    ckptio.LD_module_load_ckpt(Sketch_E, Style_E, G, D, opt_G, opt_D)

    training_batch_iterator = iter(training_data_loader)
    #testing_batch_iterator = iter(testing_data_loader)
    
    # build loss
    loss_collector = lossCollector(args, D)

    # # initialize wandb
    # if args.isMaster:
    #     wandb.init(project=args.project_id, name=args.run_id)

    global_step = -1
    while global_step < args.max_step:
        global_step += 1
        try:
            app_real, geo_real = next(training_batch_iterator)
        except StopIteration:
            training_batch_iterator = iter(training_data_loader)
            app_real, geo_real = next(training_batch_iterator)
        
        # transfer data to the gpus
        app_real, geo_real = app_real.to(gpu), geo_real.to(gpu)
        
        ###########
        #  train  #
        ###########
        # train step 1. G 관련 

        # run G : True gradient
        # generate mixed img
        geo_feature = Sketch_E(geo_real)

        app_adain_params = Style_E(app_real)

        assign_adain_params(G, app_adain_params)

        mix_img = G.rgb_forward(geo_feature)

        # recon geometry image
        geo_adain_params = Style_E(geo_real)

        assign_adain_params(G, geo_adain_params)

        recon_geo_img = G.rgb_forward(geo_feature)

        # recon appear image
        app_feature = Sketch_E(app_real)

        mix_adain_params = Style_E(mix_img)

        assign_adain_params(G, mix_adain_params)

        recon_app_img = G.rgb_forward(app_feature)

        # D
        d_geo_real = D(geo_real)
        d_app_real = D(app_real)
        d_recon_geo_img = D(recon_geo_img)
        d_recon_app_img = D(recon_app_img)
        d_mix_img = D(mix_img)

        # G loss

        loss_G = loss_collector.get_loss_G(geo_real, app_real, recon_geo_img, recon_app_img, d_geo_real, d_app_real, d_recon_geo_img, d_recon_app_img, d_mix_img)
        utils.update_net(opt_G, loss_G)

        # run D : False gradient (use .detach())
        # 앞에서 같은 과정이 있어도 .detach()로 다시 해준다.
        d_recon_geo_img = D(recon_geo_img.detach())
        d_recon_app_img = D(recon_app_img.detach())
        d_mix_img = D(mix_img.detach())
        
        
        # D loss
        loss_D = loss_collector.get_loss_D(d_geo_real, d_recon_geo_img, d_app_real, d_recon_app_img, d_mix_img)
        utils.update_net(opt_D, loss_D)

        # # # log and print loss
        # if args.isMaster and global_step % args.loss_cycle==0:
            
        #     # log loss on wandb
        #     wandb.log(loss_collector.loss_dict)
        #     loss_collector.print_loss(global_step)

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
            utils.save_img(args, global_step, "imgs", [app_real, recon_app_img, mix_img])

        # save ckpt
        if global_step % args.ckpt_cycle == 0:
            ckptio.LD_module_save_ckpt(global_step, Sketch_E, Style_E, G, D, opt_G, opt_D)

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