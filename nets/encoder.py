import torch
import torch.nn as nn

from nets.block import ResnetBlock, ConvBlock

# extract feature map from sketch
class Sketch_Encoder(nn.Module):
    def __init__(self, input_nc,output_nc=0, pad_type='reflect', norm="in", activation='relu'):
        super(Sketch_Encoder, self).__init__() 
        
        self.Block = ConvBlock(input_nc, 64, 7, stride=1, pad_type=pad_type, conv_padding=3, norm=norm, activation=activation)
        
        # downsample
        self.DownBlock1 = ConvBlock(64, 128, 3, stride=2, conv_padding=1, norm=norm, activation=activation)
        self.DownBlock2 = ConvBlock(128, 256, 3, stride=2, conv_padding=1, norm=norm, activation=activation)
        self.DownBlock3 = ConvBlock(256, 256, 3, stride=2, conv_padding=1, norm=norm, activation=activation)
        self.DownBlock4 = ConvBlock(256, output_nc, 3, stride=2, conv_padding=1, norm=norm, activation=activation)
        
        # resnet block
        self.ResBlock = ResnetBlock(output_nc, norm_type=norm, pad_type=pad_type)
    
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
    def __init__(self, input_nc, output_nc, pad_type='reflect', norm="in", activation='relu'):
        super(Image_Encoder, self).__init__()

        self.Block = ConvBlock(input_nc, 64, 7, stride=1, pad_type=pad_type, conv_padding=3, norm=norm, activation=activation)
        # downsample
        self.DownBlock1 = ConvBlock(64, 128, 3, stride=2, conv_padding=1, norm=norm, activation=activation)
        self.DownBlock2 = ConvBlock(128, 256, 3, stride=2, conv_padding=1, norm=norm, activation=activation)
        self.DownBlock3 = ConvBlock(256, 512, 3, stride=2, conv_padding=1, norm=norm, activation=activation)
        self.DownBlock4 = ConvBlock(512, output_nc, 3, stride=2, conv_padding=1, norm=norm, activation=activation)
        
        # resnet block
        self.ResBlock1 = ResnetBlock(output_nc, norm_type=norm, pad_type=pad_type)
        self.ResBlock2 = ResnetBlock(output_nc, norm_type=norm, pad_type=pad_type)
        self.ResBlock3 = ResnetBlock(output_nc, norm_type=norm, pad_type=pad_type)
        self.ResBlock4 = ResnetBlock(output_nc, norm_type=norm, pad_type=pad_type)
        self.ResBlock5 = ResnetBlock(output_nc, norm_type=norm, pad_type=pad_type)
        self.ResBlock6 = ResnetBlock(output_nc, norm_type=norm, pad_type=pad_type)

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
    def __init__(self, input_cn, pad_type='reflect', norm='none', activation='relu'):
        super(Style_Encoder, self).__init__()

        self.ConvBlock1 = ConvBlock(input_cn, 64, 7, stride=1, pad_type=pad_type, conv_padding=3, norm=norm, activation=activation)
        self.ConvBlock2 = ConvBlock(64, 128, 3, stride=2, pad_type=pad_type, conv_padding=1, norm=norm, activation=activation)
        self.ConvBlock3 = ConvBlock(128, 256, 3, stride=2, pad_type=pad_type, conv_padding=1, norm=norm, activation=activation)
        self.ConvBlock4 = ConvBlock(256, 256, 3, stride=2, pad_type=pad_type, conv_padding=1, norm=norm, activation=activation)
        self.ConvBlock5 = ConvBlock(256, 256, 3, stride=2, pad_type=pad_type, conv_padding=1, norm=norm, activation=activation)

        self.Gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, input):
        x = self.ConvBlock1(input)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.Gap(x)

        return x
