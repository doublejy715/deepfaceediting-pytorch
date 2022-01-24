import torch
import torch.nn as nn

from nets.block import ResnetBlock, ConvBlock

# extract feature map from sketch
class Sketch_Decoder_Part(nn.Module):
    def __init__(self, input_nc, output_nc, pad_type='reflect', norm="in", activation='relu'):
        super(Sketch_Decoder_Part, self).__init__() 
        # resnet block
        self.ResBlock = ResnetBlock(input_nc, norm_type=norm, pad_type=pad_type)
        
        # downsample
        self.DownBlock1 = ConvBlock(input_nc, 256, 3, stride=2, conv_padding=1, transpose=True, norm=norm, activation=activation)
        self.DownBlock2 = ConvBlock(256, 256, 3, stride=2, conv_padding=1, transpose=True, norm=norm, activation=activation)
        self.DownBlock3 = ConvBlock(256, 128, 3, stride=2, conv_padding=1, transpose=True, norm=norm, activation=activation)
        self.DownBlock4 = ConvBlock(128, 64, 3, stride=2, conv_padding=1, transpose=True, norm=norm, activation=activation)
        
        self.Block = ConvBlock(64, output_nc, kernel_size=1, stride=1, pad_type=pad_type, conv_padding=0, norm='none', activation='sig')
    
    def forward(self,input):
        ft_map_1 = self.ResBlock(input)
        ft_map_2 = self.DownBlock1(ft_map_1)
        ft_map_3 = self.DownBlock2(ft_map_2)
        ft_map_4 = self.DownBlock3(ft_map_3)
        ft_map_5 = self.DownBlock4(ft_map_4)
        output = self.Block(ft_map_5)

        layers = [input,ft_map_1,ft_map_2,ft_map_3,ft_map_4,ft_map_5,output]

        return output, layers