import torch
import torch.nn as nn
import torchvision
import cv2
import os

class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True) # b, c, 1, 1
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.fc = nn.Linear(style_dim, num_features*2)
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, s):
        h = self.fc(s.squeeze()) # s.shape : b, c, 1, 1 / s.squeeze().shape : b, c
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


def set_norm_layer(norm_type, norm_dim):
    if norm_type == 'bn':
        norm = nn.BatchNorm(norm_dim)
    elif norm_type == 'in':
        norm = nn.InstanceNorm2d(norm_dim)
    elif norm_type == 'none':
        norm = None
    else:
        assert 0, "Unsupported normalization: {}".format(norm)
    return norm

def set_activate_layer(types):
    # initialize activation
    if types == 'relu':
        activation = nn.ReLU()
    elif types == 'tanh':
        activation = nn.Tanh()
    elif types == 'sig':
        activation = nn.Sigmoid()
    elif types == 'none':
        activation = None
    else:
        assert 0, f"Unsupported activation: {types}"
    return activation

# def weight_init(m):
#     if isinstance(m, nn.Linear):
#         m.weight.data.normal_(0, 0.001)
#         m.bias.data.zero_()
        
#     if isinstance(m, nn.Conv2d):
#         nn.init.xavier_normal_(m.weight.data)

#     if isinstance(m, nn.ConvTranspose2d):
#         nn.init.xavier_normal_(m.weight.data)

def update_net(optimizer, loss):
    optimizer.zero_grad()  
    loss.backward()   
    optimizer.step()  

def setup_ddp(gpu, ngpus_per_node):
    torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:3456',
            world_size=ngpus_per_node,
            rank=gpu)

def save_image(args, global_step, dir, images):
    dir_path = f'{args.save_root}/{args.run_id}/{dir}'
    os.makedirs(dir_path, exist_ok=True)
    
    sample_image = make_grid_image(images).transpose([1,2,0]) * 255
    cv2.imwrite(dir_path + f'/e{global_step}.jpg', sample_image[:,:,::-1])

def make_grid_image(images):
    grid_rows = []

    for image_list in images:
        image_list = image_list[:8] # Drop images if there are more than 8 images in the list
        grid_row = torchvision.utils.make_grid(image_list.detach().cpu(), nrow=image_list.shape[0]) * 0.5 + 0.5
        grid_rows.append(grid_row)

    grid = torch.cat(grid_rows, dim=1).numpy()
    return grid
