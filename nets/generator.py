import torch
import torch.nn as nn

from nets.block import ResnetBlock, ConvBlock
from nets.encoder import StyleEncoder

class Local_Generator(nn.Module):
    def __init__(self, input_nc, output_nc, pad_type='reflect', norm='adain', activation='relu'):
        super(Local_Generator, self).__init__()
        self.style_encoder = StyleEncoder()

        self.ResBlock1 = ResnetBlock(input_nc, norm_type=norm, pad_type=pad_type)
        self.ResBlock2 = ResnetBlock(input_nc, norm_type=norm, pad_type=pad_type)
        self.ResBlock3 = ResnetBlock(input_nc, norm_type=norm, pad_type=pad_type)
        self.ResBlock4 = ResnetBlock(input_nc, norm_type=norm, pad_type=pad_type)

        self.ConvBlock1 = ConvBlock(input_nc, 512, 3, stride=2, conv_padding=1, transpose=True, norm=norm, activation=activation)
        self.ConvBlock2 = ConvBlock(512, 256, 3, stride=2, conv_padding=1, transpose=True, norm=norm, activation=activation)
        self.ConvBlock3 = ConvBlock(256, 128, 3, stride=2, conv_padding=1, transpose=True, norm=norm, activation=activation)
        self.ConvBlock4 = ConvBlock(128, 64, 3, stride=2, conv_padding=1, transpose=True, norm=norm, activation=activation)
        
        self.ConvBlock5 = ConvBlock(64, output_nc, 7, stride=1, pad_type=pad_type, conv_padding=3, norm='zeros', activation='tanh')

    def forward(self, input):
        x = self.ResBlock1(input)
        x = self.ResBlock2(x)
        x = self.ResBlock3(x)
        x = self.ResBlock4(x)

        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)

        x = self.ConvBlock5(x)

        return x

class Global_Generator(nn.Module):
    def __init__(self, input_nc=64, output_nc=3, pad_type='reflect', norm='in', activation='relu'):
        super(Global_Generator, self).__init__()        
        self.ConvBlock1 = ConvBlock(input_nc, 64, 7, stride=1, pad_type=pad_type, conv_padding=3, norm=norm, activation=activation)
        
        # downsample
        self.DownBlock1 = ConvBlock(64, 128, 3, stride=2, conv_padding=1, norm=norm, activation=activation)
        self.DownBlock2 = ConvBlock(128, 256, 3, stride=2, conv_padding=1, norm=norm, activation=activation)
        self.DownBlock3 = ConvBlock(256, 512, 3, stride=2, conv_padding=1, norm=norm, activation=activation)
        self.DownBlock4 = ConvBlock(512, 1024, 3, stride=2, conv_padding=1, norm=norm, activation=activation)

        # resnet blocks
        self.ResBlock1 = ResnetBlock(1024, norm_type=norm, pad_type=pad_type)
        self.ResBlock2 = ResnetBlock(1024, norm_type=norm, pad_type=pad_type)
        self.ResBlock3 = ResnetBlock(1024, norm_type=norm, pad_type=pad_type)
        self.ResBlock4 = ResnetBlock(1024, norm_type=norm, pad_type=pad_type)

        self.UpBlock1 = ConvBlock(1024, 512, 3, stride=2, conv_padding=1, transpose=True, norm=norm, activation=activation)
        self.UpBlock2 = ConvBlock(512, 256, 3, stride=2, conv_padding=1, transpose=True, norm=norm, activation=activation)
        self.UpBlock3 = ConvBlock(256, 128, 3, stride=2, conv_padding=1, transpose=True, norm=norm, activation=activation)
        self.UpBlock4 = ConvBlock(128, 64, 3, stride=2, conv_padding=1, transpose=True, norm=norm, activation=activation)

        self.ConvBlock2 = ConvBlock(64, output_nc, 7, stride=1, pad_type=pad_type, conv_padding=3, norm='none', activation='tanh')
            
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
