import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tools.HyCAS import *
from model.tools.MultiHeadSEBlock import MultiHeadSEBlock


class HimsNeXt_Block(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 exp_r: int = 4,
                 kernel_sizes=[3, 5, 7],
                 do_res: int = True,
                 n_groups: int or None = None,
                 dim='2d',
                 grn=True,
                 num_heads=4
                 ):

        super().__init__()

        self.do_res = do_res

        assert dim in ['2d', '3d']
        self.dim = dim
        if self.dim == '2d':
            conv = nn.Conv2d
        elif self.dim == '3d':
            conv = nn.Conv3d

        # First convolution layer with DepthWise Convolutions
        self.conv1 = nn.ModuleList([
            conv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=in_channels if n_groups is None else n_groups
            )
            for kernel_size in kernel_sizes
        ])

        # Pointwise convolution to combine channels
        self.pointwise = nn.Conv2d(
            len(kernel_sizes) * in_channels,
            in_channels,
            kernel_size=1
        )

        # Normalization
        self.gn = nn.GroupNorm(in_channels, in_channels)
        self.bn2 = nn.BatchNorm2d(exp_r * in_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = conv(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # GELU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.mhse = MultiHeadSEBlock(in_channels=out_channels, reduction=8, num_heads=num_heads)

        self.grn = grn
        if grn:
            if dim == '3d':
                self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)
            elif dim == '2d':
                self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True)

    def forward(self, x, dummy_tensor=None):

        x1 = x

        multi_scale_features = [conv(x1) for conv in self.conv1]
        x1 = torch.cat(multi_scale_features, dim=1)

        x1 = self.pointwise(x1)

        x1 = self.act(self.gn(x1))

        x1 = self.act(self.bn2(self.conv2(x1)))

        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
            if self.dim == '3d':
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            elif self.dim == '2d':
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1

        x1 = self.act(self.bn3(self.conv3(x1)))

        x1 = self.mhse(x1)

        if self.do_res:
            x1 = x + x1

        return x1


class HimsNeXt_DownBlock(HimsNeXt_Block):

    def __init__(self,
                 in_channels,
                 out_channels,
                 exp_r=4,
                 kernel_sizes=[3, 5, 7],
                 do_res=False,
                 dim='3d',
                 grn=True,
                 num_heads=2
                 ):

        super().__init__(in_channels, out_channels, exp_r, kernel_sizes=kernel_sizes, do_res=False, dim=dim,
                         grn=grn, num_heads=num_heads)  # Pass num_heads to the parent block.

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        self.conv1 = nn.ModuleList([
            conv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                groups=in_channels
            )
            for kernel_size in kernel_sizes
        ])

    def forward(self, x, dummy_tensor=None):

        x1 = super().forward(x)

        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class HimsNeXt_UpBlock(HimsNeXt_Block):

    def __init__(self,
                 in_channels,
                 out_channels,
                 exp_r=4,
                 kernel_sizes=[3, 5, 7],
                 do_res=False,
                 dim='2d',
                 grn=True,
                 num_heads=2
                 ):
        super().__init__(in_channels, out_channels, exp_r, kernel_sizes=kernel_sizes, do_res=False, dim=dim,
                         grn=grn, num_heads=num_heads)

        self.resample_do_res = do_res

        self.dim = dim
        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        self.conv1 = nn.ModuleList([
            conv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                groups=in_channels
            )
            for kernel_size in kernel_sizes
        ])

    def forward(self, x, dummy_tensor=None):

        x1 = super().forward(x)

        if self.dim == '2d':
            x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0))  # Padding left and top
        elif self.dim == '3d':
            x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0, 1, 0))  # Padding left, top, and front

        if self.resample_do_res:
            res = self.res_conv(x)

            if self.dim == '2d':
                res = torch.nn.functional.pad(res, (1, 0, 1, 0))  # Padding left and top
            elif self.dim == '3d':
                res = torch.nn.functional.pad(res, (1, 0, 1, 0, 1, 0))  # Padding left, top, and front

            if x1.shape != res.shape:  # Ensure their shapes are the same before adding
                if self.dim == '2d':
                    x1 = torch.nn.functional.pad(x1, (0, res.size(3) - x1.size(3), 0, res.size(2) - x1.size(2)))
                elif self.dim == '3d':
                    x1 = torch.nn.functional.pad(x1, (
                        0, res.size(4) - x1.size(4), 0, res.size(3) - x1.size(3), 0, res.size(2) - x1.size(2)))

            x1 = x1 + res

        return x1


class OutBlock(nn.Module):

    def __init__(self, in_channels, n_classes, dim):
        super().__init__()

        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        self.block = CASWithHybridGate(dim=in_channels)
        self.conv_out = conv(in_channels, n_classes, kernel_size=1)

    def forward(self, x, dummy_tensor=None):
        x = self.block(x)
        x = self.conv_out(x)

        return x


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # beta
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # gamma
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x, dummy_tensor=False):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


if __name__ == "__main__":
    # network = nnUNeXtBlock(in_channels=12, out_channels=12, do_res=False).cuda()

    # with torch.no_grad():
    #     print(network)
    #     x = torch.zeros((2, 12, 8, 8, 8)).cuda()
    #     print(network(x).shape)

    # network = DownsampleBlock(in_channels=12, out_channels=24, do_res=False)

    # with torch.no_grad():
    #     print(network)
    #     x = torch.zeros((2, 12, 128, 128, 128))
    #     print(network(x).shape)

    network = HimsNeXt_Block(in_channels=12, out_channels=12, do_res=True, grn=True, norm_type='group').cuda()
    # network = LayerNorm(normalized_shape=12, data_format='channels_last').cuda()
    # network.eval()
    with torch.no_grad():
        print(network)
        x = torch.zeros((1, 3, 64, 64)).cuda()
        print(network(x).shape)
