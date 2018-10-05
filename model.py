import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.util import concat_input
import math

class Content_Encoder(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=4, norm='in', activation='relu'):
        super(Content_Encoder, self).__init__()
        layers = []
        layers += [ConvBlock(3, conv_dim, 7, 1, 3, norm=norm, activation=activation)]

        # Down-sampling layers
        curr_dim = conv_dim
        for i in range(2):
            layers += [ConvBlock(curr_dim, curr_dim*2, 4, 2, 1, norm=norm, activation=activation)]
            curr_dim = curr_dim * 2

        # Bottleneck layers
        for i in range(repeat_num):
            layers += [ResidualBlock(dim=curr_dim, norm=norm, activation=activation)]

        self.main = nn.Sequential(*layers)
        self.curr_dim = curr_dim

    def forward(self, x):
        return self.main(x)


class Style_Encoder(nn.Module):
    def __init__(self, conv_dim=64, style_dim=8, norm='ln', activation='relu'):
        super(Style_Encoder, self).__init__()
        shared_layers = []
        specific_layers = []
        curr_dim = conv_dim

        shared_layers += [ConvBlock(3, conv_dim, 7, 1, 3, norm=norm, activation=activation)]

        # Down-sampling layers (dim*2)
        curr_dim = conv_dim
        for i in range(2):
            shared_layers += [ConvBlock(curr_dim, curr_dim*2, 4, 2, 1, norm=norm, activation=activation)]
            curr_dim = curr_dim * 2

        # Down-sampling layers (keep dim)
        for i in range(2): # original: 2
            specific_layers += [ConvBlock(curr_dim, curr_dim, 4, 2, 1, norm=norm, activation=activation)] # (16,16,256), (8,8,256), (4,4,256)

        specific_layers += [nn.AdaptiveAvgPool2d(1)]

        specific_layers += [nn.Conv2d(curr_dim, style_dim, 1, 1, 0)]

        self.shared = nn.Sequential(*shared_layers)
        self.foreground = nn.Sequential(*specific_layers)
        self.background = nn.Sequential(*specific_layers)
        self.curr_dim = curr_dim

    def forward(self, x, isf):
        x = self.shared(x)
        if isf:
            return self.foreground(x)
        else:
            return self.background(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, num_block=1, activation='relu'):
        super(MLP, self).__init__()
        layers = []
        curr_dim = dim
        layers += [LinearBlock(input_dim, curr_dim, norm='none', activation=activation)]

        for _ in range(num_block):
            layers += [LinearBlock(curr_dim, curr_dim, norm='none', activation=activation)] # 16 => 64 => 256
            
        layers += [LinearBlock(curr_dim, output_dim, norm='none', activation='none')] # no output activations
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x.view(x.size(0), -1))


class Decoder(nn.Module):
    def __init__(self, input_dim, style_dim, mlp_dim, repeat_num=4, norm='ln'):
        super(Decoder, self).__init__()
        layers = []
        curr_dim = input_dim

        # Bottleneck layers
        for i in range(repeat_num):
            layers += [ResidualBlock(dim=curr_dim, norm=norm)]

        # Up-sampling layers
        for i in range(2):
            layers += [Upsample(scale_factor=2, mode='nearest')]
            layers += [ConvBlock(curr_dim, curr_dim//2, 5, 1, 2, norm=norm)]
            curr_dim = curr_dim // 2

        # Output layer
        layers += [ConvBlock(curr_dim, 3, 7, 1, 3, norm='none', activation='tanh')]

        self.main = nn.Sequential(*layers)

        # get affine parameters
        self.f_mlp = MLP(style_dim, input_dim*2, dim=mlp_dim, num_block=repeat_num-3, activation='lrelu')
        self.b_mlp = MLP(style_dim, input_dim*2, dim=mlp_dim, num_block=repeat_num-3, activation='lrelu')

    def forward(self, c_A, m_A, s_fB, s_bA):
        A_adain_params = self.b_mlp(s_bA)
        B_adain_params = self.f_mlp(s_fB)
        
        transformed_c_A, _, _ = self.highway_AdaIN(A_adain_params, B_adain_params, m_A, c_A)
        return self.main(transformed_c_A)

    def highway_AdaIN(self, A_adain_params, B_adain_params, m_A, c_A):
        n_affine_param = B_adain_params.size(1)//2
        c_A_mean = torch.mean(torch.mean(c_A, dim=3), dim=2).unsqueeze(2).unsqueeze(3)
        c_A_std = torch.std(torch.std(c_A, dim=3), dim=2).unsqueeze(2).unsqueeze(3)

        s_A_mean = A_adain_params[:, :n_affine_param].unsqueeze(2).unsqueeze(3)
        s_A_log_var = A_adain_params[:, n_affine_param:n_affine_param*2].unsqueeze(2).unsqueeze(3)
        s_A_std = torch.exp(0.5*s_A_log_var)

        s_B_mean = B_adain_params[:, :n_affine_param].unsqueeze(2).unsqueeze(3)
        s_B_log_var = B_adain_params[:, n_affine_param:n_affine_param*2].unsqueeze(2).unsqueeze(3)
        s_B_std = torch.exp(0.5*s_B_log_var)

        eps = 1e-5
        norm_c_A = (c_A - c_A_mean) / (c_A_std + eps)

        return (m_A * (s_B_std * norm_c_A + s_B_mean) + (1 - m_A) * (s_A_std * norm_c_A + s_A_mean)
                , s_B_mean.view(s_B_mean.size(0), -1), s_B_std.view(s_B_std.size(0), -1))


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, repeat_num=8, style_dim=8, mlp_dim=256, c_dim=3):
        super(Generator, self).__init__()

        self.c_encoder = Content_Encoder(conv_dim, repeat_num//2)
        self.s_encoder = Style_Encoder(conv_dim, style_dim, norm= 'none', activation='lrelu')
        self.mask = Mask(self.c_encoder.curr_dim, 3, norm='bn', activation='relu')

        self.decoder = Decoder(self.c_encoder.curr_dim, style_dim, mlp_dim, repeat_num//2, norm='ln')

    def forward(self, x_A, x_B):
        # Replicate spatially and concatenate domain information.
        c_A = self.c_encoder(x_A)
        c_B = self.c_encoder(x_B)

        m_A = self.mask(c_A, c_B)
        m_B = self.mask(c_B, c_A)

        up_im_A = F.interpolate(1-m_A, None, 4, 'bilinear', align_corners=False)
        up_m_B = F.interpolate(m_B, None, 4, 'bilinear', align_corners=False)

        s_fB = self.s_encoder(up_m_B*x_B, isf=True)
        s_bA = self.s_encoder(up_im_A*x_A, isf=False)


        out = self.decoder(c_A, m_A, s_fB, s_bA)

        return out


class Coseg_Attention(nn.Module):
    def __init__(self, curr_dim, norm='in'):
        super(Coseg_Attention, self).__init__()
        
        self.gap = nn.AdaptiveAvgPool2d(1) # B,256,1,1

        layers = []
        layers += [LinearBlock(curr_dim, curr_dim*2, norm=norm, activation='tanh')]
        layers += [LinearBlock(curr_dim*2, curr_dim, norm=norm, activation='sigmoid')]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = self.gap(x)
        return self.main(x.view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)


class Mask(nn.Module):
    def __init__(self, input_dim=256, repeat_num=3, norm='ln', activation='relu'):
        super(Mask, self).__init__()
        layers = []

        conv_dim = input_dim
        for i in range(repeat_num):
            layers += [ConvBlock(conv_dim, conv_dim, 3, 1, 1, norm=norm, activation=activation)] # 264,32,32 => 132,32,32 => 66,32,32
            layers += [ConvBlock(conv_dim, conv_dim, 3, 1, 1, norm=norm, activation=activation)] # 264,32,32 => 132,32,32 => 66,32,32

        # final layer: [0,1]
        layers += [ConvBlock(conv_dim, 1, 1, 1, 0, norm='none', activation='sigmoid')] # 1,32,32

        self.main = nn.Sequential(*layers)
        self.co_attn = Coseg_Attention(curr_dim=input_dim, norm='none')

    def forward(self, c_A, c_B):
        ch_m = self.co_attn(c_B)
        return self.main(ch_m*c_A)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers += [ConvBlock(3, conv_dim, 4, 2, 1, norm='none', activation='lrelu')]

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers += [ConvBlock(curr_dim, curr_dim*2, 4, 2, 1, norm='none', activation='lrelu')]
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))

        self.main = nn.Sequential(*layers)
        self.conv1 = ConvBlock(curr_dim, 1, 3, 1, 1, norm='none', activation='none')
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim, norm='in', activation='relu', use_affine=True):
        super(ResidualBlock, self).__init__()
        layers = []
        layers += [ConvBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, use_affine=use_affine)]
        layers += [ConvBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', use_affine=use_affine)]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.main(x)


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, k, s, p, dilation=False, norm='in',
                        activation='relu', pad_type='mirror', use_affine=True, use_bias=True):
        super(ConvBlock, self).__init__()

        # Init Normalization
        if norm == 'in':
            self.norm = nn.InstanceNorm2d(output_dim, affine=use_affine, track_running_stats=True)
        elif norm == 'ln':
            # LayerNorm(output_dim, affine=use_affine)
            self.norm = nn.GroupNorm(1, output_dim)
        elif norm == 'bn':
            self.norm = nn.BatchNorm2d(output_dim)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(32, output_dim)
        elif norm == 'none':
            self.norm = None

        # Init Activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.01, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU(num_parameters=1, init=0.25)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None

        # Init pad-type
        if pad_type == 'mirror':
            self.pad = nn.ReflectionPad2d(p)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(p)

        # initialize convolution
        if dilation:
            self.conv = nn.Conv2d(input_dim, output_dim, k, s, dilation=p, bias=use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, k, s, bias=use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='ln', activation='relu', use_affine=True):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # Init Normalization
        if norm == 'ln':
            # self.norm = LayerNorm(output_dim, affine=use_affine)
            self.norm = nn.GroupNorm(1, output_dim)
        elif norm == 'none':
            self.norm = None

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.01, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU(num_parameters=1, init=0.25)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info