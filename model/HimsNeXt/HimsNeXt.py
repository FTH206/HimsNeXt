import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from model.HimsNeXt.blocks_himsnext import *


class HimsNeXt(nn.Module):

    def __init__(self,
                 in_channels: int,
                 n_channels: int,
                 n_classes: int,
                 exp_r: int = 4,  # Expansion ratio as in Swin Transformers
                 kernel_sizes=[3, 5, 7],
                 deep_supervision: bool = False,  # Can be used to test2 deep supervision
                 do_res: bool = False,  # Can be used to individually test2 residual connection
                 do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
                 checkpoint_style: bool = None,  # Either inside block or outside block
                 block_counts: list = [2, 2, 2, 2, 2, 2, 2, 2, 2],  # Can be used to test2 staging ratio:
                 # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
                 norm_type='group',
                 dim='2d',  # 2d or 3d
                 grn=True
                 ):

        super().__init__()

        self.do_ds = deep_supervision
        assert checkpoint_style in [None, 'outside_block']
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        assert dim in ['2d', '3d']


        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d

        self.stem = conv(in_channels, n_channels, kernel_size=1)
        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]

        self.enc_block_0 = nn.Sequential(*[
            HimsNeXt_Block(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[0],
                kernel_sizes=kernel_sizes,
                do_res=do_res,
                dim=dim,
                grn=grn,
                num_heads=2
            )
            for i in range(block_counts[0])]
                                         )

        self.down_0 = HimsNeXt_DownBlock(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[1],
            kernel_sizes=kernel_sizes,
            do_res=do_res_up_down,
            dim=dim,
            grn=grn,
            num_heads=2
        )

        self.enc_block_1 = nn.Sequential(*[
            HimsNeXt_Block(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[1],
                kernel_sizes=kernel_sizes,
                do_res=do_res,
                dim=dim,
                grn=grn,
                num_heads=4
            )
            for i in range(block_counts[1])]
                                         )

        self.down_1 = HimsNeXt_DownBlock(
            in_channels=2 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[2],
            kernel_sizes=kernel_sizes,
            do_res=do_res_up_down,
            dim=dim,
            grn=grn,
            num_heads=4
        )

        self.enc_block_2 = nn.Sequential(*[
            HimsNeXt_Block(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[2],
                kernel_sizes=kernel_sizes,
                do_res=do_res,
                dim=dim,
                grn=grn,
                num_heads=8
            )
            for i in range(block_counts[2])]
                                         )

        self.down_2 = HimsNeXt_DownBlock(
            in_channels=4 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[3],
            kernel_sizes=kernel_sizes,
            do_res=do_res_up_down,
            dim=dim,
            grn=grn,
            num_heads=8
        )

        self.enc_block_3 = nn.Sequential(*[
            HimsNeXt_Block(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[3],
                kernel_sizes=kernel_sizes,
                do_res=do_res,
                dim=dim,
                grn=grn,
                num_heads=8
            )
            for i in range(block_counts[3])]
                                         )

        self.down_3 = HimsNeXt_DownBlock(
            in_channels=8 * n_channels,
            out_channels=16 * n_channels,
            exp_r=exp_r[4],
            kernel_sizes=kernel_sizes,
            do_res=do_res_up_down,
            dim=dim,
            grn=grn,
            num_heads=8
        )

        self.bottleneck = nn.Sequential(*[
            HimsNeXt_Block(
                in_channels=n_channels * 16,
                out_channels=n_channels * 16,
                exp_r=exp_r[4],
                kernel_sizes=kernel_sizes,
                do_res=do_res,
                dim=dim,
                grn=grn,
                num_heads=8
            )
            for i in range(block_counts[4])]
                                        )

        self.up_3 = HimsNeXt_UpBlock(
            in_channels=16 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[5],
            kernel_sizes=kernel_sizes,
            do_res=do_res_up_down,
            dim=dim,
            grn=grn,
            num_heads=8
        )

        self.dec_block_3 = nn.Sequential(*[
            HimsNeXt_Block(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[5],
                kernel_sizes=kernel_sizes,
                do_res=do_res,
                dim=dim,
                grn=grn,
                num_heads=8
            )
            for i in range(block_counts[5])]
                                         )

        self.up_2 = HimsNeXt_UpBlock(
            in_channels=8 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[6],
            kernel_sizes=kernel_sizes,
            do_res=do_res_up_down,
            dim=dim,
            grn=grn,
            num_heads=8
        )

        self.dec_block_2 = nn.Sequential(*[
            HimsNeXt_Block(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[6],
                kernel_sizes=kernel_sizes,
                do_res=do_res,
                dim=dim,
                grn=grn,
                num_heads=8
            )
            for i in range(block_counts[6])]
                                         )

        self.up_1 = HimsNeXt_UpBlock(
            in_channels=4 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[7],
            kernel_sizes=kernel_sizes,
            do_res=do_res_up_down,
            dim=dim,
            grn=grn,
            num_heads=4
        )

        self.dec_block_1 = nn.Sequential(*[
            HimsNeXt_Block(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[7],
                kernel_sizes=kernel_sizes,
                do_res=do_res,
                dim=dim,
                grn=grn,
                num_heads=4
            )
            for i in range(block_counts[7])]
                                         )

        self.up_0 = HimsNeXt_UpBlock(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_sizes=kernel_sizes,
            do_res=do_res_up_down,
            dim=dim,
            grn=grn,
            num_heads=2
        )

        self.dec_block_0 = nn.Sequential(*[
            HimsNeXt_Block(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_sizes=kernel_sizes,
                do_res=do_res,
                dim=dim,
                grn=grn,
                num_heads=2
            )
            for i in range(block_counts[8])]
                                         )

        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)

        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels * 2, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=n_channels * 4, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=n_channels * 8, n_classes=n_classes, dim=dim)
            self.out_4 = OutBlock(in_channels=n_channels * 16, n_classes=n_classes, dim=dim)

        self.block_counts = block_counts


    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x

    def forward(self, x):

        x = self.stem(x)
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor)
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor)
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor)
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor)

            x = self.iterative_checkpoint(self.bottleneck, x)
            if self.do_ds:
                x_ds_4 = checkpoint.checkpoint(self.out_4, x, self.dummy_tensor)

            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
            if x_res_3.shape != x_up_3.shape:
                x_up_3 = nn.functional.interpolate(x_up_3, size=x_res_3.shape[2:], mode='bilinear', align_corners=False)
            dec_x = x_res_3 + x_up_3
            x = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if self.do_ds:
                x_ds_3 = checkpoint.checkpoint(self.out_3, x, self.dummy_tensor)
            del x_res_3, x_up_3

            x_up_2 = checkpoint.checkpoint(self.up_2, x, self.dummy_tensor)
            if x_res_2.shape != x_up_2.shape:
                x_up_2 = nn.functional.interpolate(x_up_2, size=x_res_2.shape[2:], mode='bilinear', align_corners=False)
            dec_x = x_res_2 + x_up_2
            x = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x, self.dummy_tensor)
            del x_res_2, x_up_2

            x_up_1 = checkpoint.checkpoint(self.up_1, x, self.dummy_tensor)
            if x_res_1.shape != x_up_1.shape:
                x_up_1 = nn.functional.interpolate(x_up_1, size=x_res_1.shape[2:], mode='bilinear', align_corners=False)
            dec_x = x_res_1 + x_up_1
            x = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x, self.dummy_tensor)
            del x_res_1, x_up_1

            x_up_0 = checkpoint.checkpoint(self.up_0, x, self.dummy_tensor)
            if x_res_0.shape != x_up_0.shape:
                x_up_0 = nn.functional.interpolate(x_up_0, size=x_res_0.shape[2:], mode='bilinear', align_corners=False)
            dec_x = x_res_0 + x_up_0
            x = self.iterative_checkpoint(self.dec_block_0, dec_x)
            del x_res_0, x_up_0, dec_x

            x = checkpoint.checkpoint(self.out_0, x, self.dummy_tensor)


        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else:
            return {"out": x}

if __name__ == '__main__':

    network = HimsNeXt(
        in_channels=3,
        n_channels=32,
        n_classes=2,
        # exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],  # Expansion ratio as in Swin Transformers
        exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        kernel_sizes=[3,5,7],  # Can test2 kernel_size
        deep_supervision=False,  # Can be used to test2 deep supervision
        do_res=True,  # Can be used to individually test2 residual connection
        do_res_up_down=True,
        # block_counts = [2,2,2,2,2,2,2,2,2],
        block_counts=[3, 4, 4, 4, 4, 4, 4, 4, 3],
        checkpoint_style = 'outside_block',
        dim='2d',
        grn=True

    ).cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(count_parameters(network))

    # from fvcore.nn import FlopCountAnalysis
    # from fvcore.nn import parameter_count_table

    # model = ResTranUnet(img_size=128, in_channels=1, num_classes=14, dummy=False).cuda()
    # x = torch.zeros((1, 3, 672, 672), requires_grad=False).cuda()
    # flops = FlopCountAnalysis(network, x)
    # print(flops.total())

    with torch.no_grad():
        print(network)
        x = torch.zeros((1, 3, 128, 128)).cuda()
        print(network(x)['out'].shape)
