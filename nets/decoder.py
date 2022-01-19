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
        
        self.Block = ConvBlock(64, output_nc, 7, stride=1, pad_type=pad_type, conv_padding=3, norm=norm, activation=activation)
    
    def forward(self,input):
        x = self.ResBlock(input)
        x = self.DownBlock1(x)
        x = self.DownBlock2(x)
        x = self.DownBlock3(x)
        x = self.DownBlock4(x)
        x = self.Block(x)
        return x