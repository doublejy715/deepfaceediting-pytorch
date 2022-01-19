
import argparse

def train_options():
    parser = argparse.ArgumentParser(description='Deepfaceediting-pytorch-implementation')
    
    # root
    parser.add_argument('--save_root', type=str, default='')
    parser.add_argument("--geo", type=str, default = "", help = "the path of geometry image")
    parser.add_argument("--appear", type=str, default = "", help = "the path of appearance image")
    parser.add_argument("--output", type=str, default = "", help = "the path of output image")
    
    # ids
    parser.add_argument('--ckpt_id', type=str, default='test')
    
    # etc
    parser.add_argument("--cuda", type=int, default = 1, help = "use cuda or cpu: 0 , cpu; 1 , gpu")
    parser.add_argument("--geo_type", type=str, default="sketch", help = "extract geometry from image or sketch: sketch / image")
    parser.add_argument("--gen_sketch", action='store_true', help = "with --gen_sketch, extract sketch from real image")

    return parser.parse_args()