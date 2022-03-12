import torch
import torch.nn as nn

from lib.block import ResnetBlock, ConvBlock, ResnetBlock_Adain
from .tmp import Image_Encoder_, Sketch_Decoder_


class DFE_generator(nn.Module):

    def __init__(self):
        super(DFE_generator, self).__init__()

        self.Local_G = Local_G(256,3).train()
        self.Style_E = Style_Encoder(3).train()

        # self.Image_E = Image_Encoder(3,256).eval()
        # self.Image_E.load_state_dict(torch.load('./train/latest/ckpt/latest.pt',map_location=torch.device('cuda'))["image_E"], strict=False)
        # for params in self.Image_E.parameters():
        #     params.requires_grad = False
            
        # self.Sketch_D = Sketch_Decoder(256,1).eval()
        # self.Sketch_D.load_state_dict(torch.load('./train/latest/ckpt/latest.pt',map_location=torch.device('cuda'))["sketch_D"], strict=False)
        # for params in self.Sketch_D.parameters():
        #     params.requires_grad = False

        self.Image_E = Image_Encoder_(3,256).eval()
        self.Image_E.load_state_dict(torch.load('./train/latest/ckpt/latest.pt',map_location=torch.device('cuda'))["image_E"], strict=False)
        for params in self.Image_E.parameters():
            params.requires_grad = False
            
        self.Sketch_D = Sketch_Decoder_(256,1).eval()
        self.Sketch_D.load_state_dict(torch.load('./train/latest/ckpt/latest.pt',map_location=torch.device('cuda'))["sketch_D"], strict=False)
        for params in self.Sketch_D.parameters():
            params.requires_grad = False
        
    def forward(self, I_source, I_target):
        I_target_feature = self.Image_E(I_target)
        I_source_adain_params = self.Style_E(I_source)
        I_result = self.Local_G(I_target_feature, I_source_adain_params)
        return I_result, I_target_feature

    def get_sketch_from_image(self, image):
        with torch.no_grad():
            feature = self.Image_E(image)
            sketch = self.Sketch_D(feature)[0]
            return sketch

    def get_sketch_from_feature(self, feature):
        with torch.no_grad():
            sketch = self.Sketch_D(feature)[0]
            return sketch


#-----------------
# Encoder
#-----------------
# extract feature map from sketch
class Sketch_Encoder(nn.Module):
    def __init__(self, input_nc,output_nc=0, pad_type='reflect', norm_type="in", activation_type='relu'):
        super(Sketch_Encoder, self).__init__() 
        
        self.Block = ConvBlock(input_nc, 64, 7, stride=1, pad_type=pad_type, conv_padding=3, norm_type=norm_type, activation_type=activation_type)
        
        # downsample
        self.DownBlock1 = ConvBlock(64, 128, 3, stride=2, conv_padding=1, norm_type=norm_type, activation_type=activation_type)
        self.DownBlock2 = ConvBlock(128, 256, 3, stride=2, conv_padding=1, norm_type=norm_type, activation_type=activation_type)
        self.DownBlock3 = ConvBlock(256, 256, 3, stride=2, conv_padding=1, norm_type=norm_type, activation_type=activation_type)
        self.DownBlock4 = ConvBlock(256, output_nc, 3, stride=2, conv_padding=1, norm_type=norm_type, activation_type=activation_type)
        
        self.ResBlock = ResnetBlock(output_nc, norm_type=norm_type, pad_type=pad_type)
    
    def forward(self,input):
        x = self.Block(input)
        x = self.DownBlock1(x)
        x = self.DownBlock2(x)
        x = self.DownBlock3(x)
        x = self.DownBlock4(x)
        x = self.ResBlock(x)
        return x

# extract feature map from Image
class Image_Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, pad_type='reflect', norm_type="in", activation_type='relu'):
        super(Image_Encoder, self).__init__()

        self.Block = ConvBlock(input_nc, 64, 7, stride=1, pad_type=pad_type, conv_padding=3, norm_type=norm_type, activation_type=activation_type)
        
        # downsample
        self.DownBlock1 = ConvBlock(64, 128, 3, stride=2, conv_padding=1, norm_type=norm_type, activation_type=activation_type)
        self.DownBlock2 = ConvBlock(128, 256, 3, stride=2, conv_padding=1, norm_type=norm_type, activation_type=activation_type)
        self.DownBlock3 = ConvBlock(256, 512, 3, stride=2, conv_padding=1, norm_type=norm_type, activation_type=activation_type)
        self.DownBlock4 = ConvBlock(512, output_nc, 3, stride=2, conv_padding=1, norm_type=norm_type, activation_type=activation_type)
        
        self.ResBlock1 = ResnetBlock(output_nc, norm_type=norm_type, pad_type=pad_type)
        self.ResBlock2 = ResnetBlock(output_nc, norm_type=norm_type, pad_type=pad_type)
        self.ResBlock3 = ResnetBlock(output_nc, norm_type=norm_type, pad_type=pad_type)
        self.ResBlock4 = ResnetBlock(output_nc, norm_type=norm_type, pad_type=pad_type)
        self.ResBlock5 = ResnetBlock(output_nc, norm_type=norm_type, pad_type=pad_type)
        self.ResBlock6 = ResnetBlock(output_nc, norm_type=norm_type, pad_type=pad_type)

    def forward(self,input):
        x = self.Block(input)
        x = self.DownBlock1(x)
        x = self.DownBlock2(x)
        x = self.DownBlock3(x)
        x = self.DownBlock4(x)
        x = self.ResBlock1(x)
        x = self.ResBlock2(x)
        x = self.ResBlock3(x)
        x = self.ResBlock4(x)
        x = self.ResBlock5(x)
        x = self.ResBlock6(x)
        return x

class Style_Encoder(nn.Module):
    def __init__(self, input_cn, pad_type='reflect', norm_type='none', activation_type='relu'):
        super(Style_Encoder, self).__init__()

        self.ConvBlock1 = ConvBlock(input_cn, 64, 7, stride=1, pad_type=pad_type, conv_padding=3, norm_type=norm_type, activation_type=activation_type)
        self.ConvBlock2 = ConvBlock(64, 128, 3, stride=2, pad_type=pad_type, conv_padding=1, norm_type=norm_type, activation_type=activation_type)
        self.ConvBlock3 = ConvBlock(128, 256, 3, stride=2, pad_type=pad_type, conv_padding=1, norm_type=norm_type, activation_type=activation_type)
        self.ConvBlock4 = ConvBlock(256, 256, 3, stride=2, pad_type=pad_type, conv_padding=1, norm_type=norm_type, activation_type=activation_type)
        self.ConvBlock5 = ConvBlock(256, 256, 3, stride=2, pad_type=pad_type, conv_padding=1, norm_type=norm_type, activation_type=activation_type)

        self.Gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, input):
        x = self.ConvBlock1(input)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.Gap(x)

        return x


#-----------------
# Decoder
#-----------------
class Sketch_Decoder(nn.Module):
    def __init__(self, input_nc, output_nc, pad_type='reflect', norm_type="in", activation_type='relu'):
        super(Sketch_Decoder, self).__init__() 
        # resnet block
        self.ResBlock = ResnetBlock(input_nc, norm_type=norm_type, pad_type=pad_type)
        
        # downsample
        self.DownBlock1 = ConvBlock(input_nc, 256, 3, stride=2, conv_padding=1, upsample=True, norm_type=norm_type, activation_type=activation_type)
        self.DownBlock2 = ConvBlock(256, 256, 3, stride=2, conv_padding=1, upsample=True, norm_type=norm_type, activation_type=activation_type)
        self.DownBlock3 = ConvBlock(256, 128, 3, stride=2, conv_padding=1, upsample=True, norm_type=norm_type, activation_type=activation_type)
        self.DownBlock4 = ConvBlock(128, 64, 3, stride=2, conv_padding=1, upsample=True, norm_type=norm_type, activation_type=activation_type)
        
        self.Block = ConvBlock(64, output_nc, kernel_size=1, stride=1, pad_type=pad_type, conv_padding=0, norm_type='none', activation_type='sig')
    
    def forward(self,input):
        ft_map_1 = self.ResBlock(input)
        ft_map_2 = self.DownBlock1(ft_map_1)
        ft_map_3 = self.DownBlock2(ft_map_2)
        ft_map_4 = self.DownBlock3(ft_map_3)
        ft_map_5 = self.DownBlock4(ft_map_4)
        output = self.Block(ft_map_5)

        layers = [input,ft_map_1,ft_map_2,ft_map_3,ft_map_4,ft_map_5,output]

        return output, layers


#-----------------
# Generator
#-----------------
class Local_G(nn.Module):
    def __init__(self, input_nc, output_nc, pad_type='reflect', norm_type='in', activation_type='relu'):
        super(Local_G, self).__init__()
        self.ResBlock1 = ResnetBlock_Adain(input_nc)
        self.ResBlock2 = ResnetBlock_Adain(input_nc)
        self.ResBlock3 = ResnetBlock_Adain(input_nc)
        self.ResBlock4 = ResnetBlock_Adain(input_nc)

        self.ConvBlock1 = ConvBlock(input_nc, 256, 3, stride=2, conv_padding=1, upsample=True, norm_type=norm_type, activation_type=activation_type)
        self.ConvBlock2 = ConvBlock(256, 256, 3, stride=2, conv_padding=1, upsample=True, norm_type=norm_type, activation_type=activation_type)
        self.ConvBlock3 = ConvBlock(256, 128, 3, stride=2, conv_padding=1, upsample=True, norm_type=norm_type, activation_type=activation_type)
        self.ConvBlock4 = ConvBlock(128, 64, 3, stride=2, conv_padding=1, upsample=True, norm_type=norm_type, activation_type=activation_type)
        
        # feature map to rgb image convert layer
        self.ConvBlock5 = ConvBlock(64, output_nc, 7, stride=1, pad_type=pad_type, conv_padding=3, norm_type='none', activation_type='tanh')


    def forward(self, input, I_s_adain_params, RGB=True):
        x = self.ResBlock1(input, I_s_adain_params)
        x = self.ResBlock2(x, I_s_adain_params)
        x = self.ResBlock3(x, I_s_adain_params)
        x = self.ResBlock4(x, I_s_adain_params)

        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)

        if RGB:
            x = self.ConvBlock5(x)

        return x


class Global_G(nn.Module):
    def __init__(self, input_nc=64, output_nc=3, pad_type='reflect', norm_type='in', activation_type='relu'):
        super(Global_G, self).__init__()        
        self.ConvBlock1 = ConvBlock(input_nc, 64, 7, stride=1, pad_type=pad_type, conv_padding=3, norm_type=norm_type, activation_type=activation_type)
        
        # downsample
        self.DownBlock1 = ConvBlock(64, 128, 3, stride=2, conv_padding=1, norm_type=norm_type, activation_type=activation_type)
        self.DownBlock2 = ConvBlock(128, 256, 3, stride=2, conv_padding=1, norm_type=norm_type, activation_type=activation_type)
        self.DownBlock3 = ConvBlock(256, 512, 3, stride=2, conv_padding=1, norm_type=norm_type, activation_type=activation_type)
        self.DownBlock4 = ConvBlock(512, 1024, 3, stride=2, conv_padding=1, norm_type=norm_type, activation_type=activation_type)

        # resnet blocks
        self.ResBlock1 = ResnetBlock(1024, norm_type=norm_type, pad_type=pad_type)
        self.ResBlock2 = ResnetBlock(1024, norm_type=norm_type, pad_type=pad_type)
        self.ResBlock3 = ResnetBlock(1024, norm_type=norm_type, pad_type=pad_type)
        self.ResBlock4 = ResnetBlock(1024, norm_type=norm_type, pad_type=pad_type)

        self.UpBlock1 = ConvBlock(1024, 512, 3, stride=2, conv_padding=1, upsample=True, norm_type=norm_type, activation_type=activation_type)
        self.UpBlock2 = ConvBlock(512, 256, 3, stride=2, conv_padding=1, upsample=True, norm_type=norm_type, activation_type=activation_type)
        self.UpBlock3 = ConvBlock(256, 128, 3, stride=2, conv_padding=1, upsample=True, norm_type=norm_type, activation_type=activation_type)
        self.UpBlock4 = ConvBlock(128, 64, 3, stride=2, conv_padding=1, upsample=True, norm_type=norm_type, activation_type=activation_type)

        self.ConvBlock2 = ConvBlock(64, output_nc, 7, stride=1, pad_type=pad_type, conv_padding=3, norm_type='none', activation_type='tanh')
            
    def forward(self, input):
        x = self.ConvBlock1(input)

        x = self.DownBlock1(x)
        x = self.DownBlock2(x)
        x = self.DownBlock3(x)
        x = self.DownBlock4(x)

        x = self.ResBlock1(x)
        x = self.ResBlock2(x)
        x = self.ResBlock3(x)
        x = self.ResBlock4(x)

        x = self.UpBlock1(x)
        x = self.UpBlock2(x)
        x = self.UpBlock3(x)
        x = self.UpBlock4(x)

        x = self.ConvBlock2(x)
        return x
