import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_type, pad_type):
        super(ResnetBlock, self).__init__()
        self.convblock1 = ConvBlock(dim ,dim, 3, 1, conv_padding=1, norm=norm_type, activation='relu', pad_type=pad_type)
        self.convblock2 = ConvBlock(dim ,dim, 3, 1, conv_padding=1, norm=norm_type, activation='none', pad_type=pad_type)

    def forward(self, x):
        residual_x = self.convblock1(x)
        residual_x = self.convblock2(residual_x)
        return x + residual_x

# Definition of normalization layer
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features

        self.norm = nn.InstanceNorm2d(num_features, affine=False)

        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
    
    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        out = self.norm(x)
        out = out * self.weight + self.bias
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class ConvBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride, conv_padding=0, transpose=False,
                 pad_type='zeros', norm='none', activation='relu'):
        super(ConvBlock, self).__init__()
        self.use_bias = True
        if transpose: 
            self.up = nn.Upsample(scale_factor=stride)
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, 1, padding=conv_padding, padding_mode=pad_type,bias=self.use_bias)

        else:
            self.up = False
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding=conv_padding, padding_mode=pad_type, bias=self.use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        if self.up:
            x = self.up(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


