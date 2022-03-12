import torch
import torch.nn as nn

from lib.block_ import ResnetBlock, ConvBlock, ResnetBlock_Adain

# extract feature map from Image
class Image_Encoder_(nn.Module):
    def __init__(self, input_nc, output_nc, pad_type='reflect', norm_type="in", activation_type='relu'):
        super(Image_Encoder_, self).__init__()

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
class Sketch_Decoder_(nn.Module):
    def __init__(self, input_nc, output_nc, pad_type='reflect', norm_type="in", activation_type='relu'):
        super(Sketch_Decoder_, self).__init__() 
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
