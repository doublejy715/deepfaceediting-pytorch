import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import torchvision
import os
from torch.optim import lr_scheduler
import functools

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((512, 512), Image.BICUBIC)
    return img

def get_grid_row(images):
    # get 8 images
    images = images[:8]

    # make one row
    grid_row = torchvision.utils.make_grid(images.detach().cpu(), nrow=images.shape[0])
    # grid_row = torchvision.utils.make_grid(images.detach().cpu(), nrow=images.shape[0]) * 0.5 + 0.5


    return grid_row

def test_save_img(image, path):
    image = image.squeeze(0).detach().numpy()
    # image = (np.transpose(image, (1, 2, 0)) + 1) / 2.0 * 255.0
    image = np.transpose(image, (1, 2, 0)) * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)

def save_img(args, global_step, dir, images):

    # make dir
    os.makedirs(f'{args.save_root}/{args.run_id}/{dir}', exist_ok=True)
    
    # make grid
    sample_image = make_image(images).transpose([1,2,0])*255
    
    # set path
    save_path = f'{args.save_root}/{args.run_id}/{dir}/e{global_step}.jpg'

    # save image
    cv2.imwrite(save_path, sample_image[:,:,::-1])

def make_image(images):

    grid_rows = []

    # convert each image tensor to row
    for image in images:

        # get one row
        grid_row = get_grid_row(image)

        # append row
        grid_rows.append(grid_row)

    # make grid
    grid = torch.cat(grid_rows, dim=1).numpy()

    return grid

def setup_ddp(gpu, ngpus_per_node):
    
    # setup ddp
    torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:3456',
            world_size=ngpus_per_node,
            rank=gpu)

def update_net(optimizer, loss):

    # clear old gradients
    optimizer.zero_grad()
    
    # computes derivative of loss 
    loss.backward()

    # take one step
    optimizer.step()


