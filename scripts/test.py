from __future__ import print_function
import os
import time
import sys
sys.path.append(os.getcwd())

import torch
import torchvision.transforms as transforms

from opts.test_options import train_options
from utils.utils import is_image_file, load_img, test_save_img
from core.checkpoint import ckptIO
from nets.model import define_G

transform_image = transforms.Compose([
    transforms.Resize(size=512),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

if __name__ == "__main__":
    start_time = time.time()

    # Testing settings
    args = train_options()
    device = torch.device("cuda" if args.cuda else "cpu")

    # load image
    geo_img = load_img(args.geo)
    appear_img = load_img(args.appear)

    geo_img = transform_image(geo_img).unsqueeze(0)
    appear_img = transform_image(appear_img).unsqueeze(0)

    # define model
    G = define_G(args)

    # load model parameter
    ckptio = ckptIO(args)
    ckptio.test_load_ckpt(G)
    G.to(device)

    swap_image = G.forward(geo_img,appear_img,args.geo_type)
    test_save_img(swap_image,args.save_root)

    # check time
    end_time = time.time()
    take_time = start_time - end_time
    print("Sketch Job End!!")
    print(f"It takes {take_time}s")
