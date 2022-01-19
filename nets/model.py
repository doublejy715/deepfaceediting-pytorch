import torch
import torch.nn as nn

from nets.encoder import Sketch_Encoder_Part, Image_Encoder_Part
from nets.generator import Local_Generator, Global_Generator
from utils.nets_utils import get_generated_part_feature, combine_feature_map


def define_G(args):
    netG = Combine_Model().to(args.gpu_id)
    return netG

class Combine_Model(nn.Module):
    def __init__(self):
        self.part = {'bg': (0, 0, 512),
                     'eye1': (108, 156, 128),
                     'eye2': (255, 156, 128),
                     'nose': (182, 232, 160),
                     'mouth': (169, 301, 192)}
                     
        self.face_components = self.part.keys()
        self.sketch_encoder_part = {}
        self.image_encoder_part = {}
        self.local_gen_part = {}
        
        # define network
        for key in self.part.keys():
            self.sketch_encoder_part[key] = Sketch_Encoder_Part(input_nc=3,output_nc=1024)
            self.image_encoder_part[key] = Image_Encoder_Part(input_nc=3,output_nc=1024)
            self.local_gen_part[key] = Local_Generator(input_nc=1024, output_nc=3)

        # Global Fusion
        self.global_gen = Global_Generator(input_nc = 64, output_nc = 3)
        
    def forward(self, sketch, appear, geo_type):
        part_features = {}
        for component in self.face_components:
            x,y,size = self.part[component]
            sketch_component_encoder = self.sketch_encoder_part[component]
            image_component_encoder = self.image_encoder_part[component]
            fake_component_generator = self.local_gen_part[component]

            sketch_part = sketch[:,:,y: y + size, x: x + size]
            appear_part = appear[:,:,y: y + size, x: x + size]
            with torch.no_grad():
                if geo_type == "sketch":
                    sketch_feature = sketch_component_encoder(sketch_part)
                else:
                    sketch_feature = image_component_encoder(sketch_part)

                part_features[component] = get_generated_part_feature(fake_component_generator, sketch_feature, appear_part)
        
        combined_feature_map = combine_feature_map(part_features)
   
        with torch.no_grad():
            fake_image = self.global_gen(combined_feature_map)

        return fake_image