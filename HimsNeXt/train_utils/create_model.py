from model.AttUNet import AttU_Net
from model.CMUNeXt.CMUNeXt import CMUNeXt
from model.TransUnet import TransUnet
from model.UNeXt.UNeXt import UNext
from model.MedT.axialnet import *
from model.UNet import UNet
from model.MN.MN import mn
from model.HimsNeXt.HimsNeXt import HimsNeXt
from model.UNet3plus.UNet3plus import UNet3plus
from model.UNetPP import NestedUNet


# Create a segmentation model for the requested number of classes.
def create_Unet(num_classes):

    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model

def create_mednextv1_medium(num_classes):

    model = mn(
        in_channels = 3,
        n_channels = 32,
        n_classes = num_classes,
        exp_r=[2,3,4,4,4,4,4,3,2],
        kernel_size=3,
        deep_supervision=True,
        do_res=True,
        do_res_up_down = True,
        block_counts = [3,4,4,4,4,4,4,4,3],
        checkpoint_style = 'outside_block'
    )
    return model


def create_HimsNeXt(num_classes,deep_supervision=True):

    model = HimsNeXt(
        in_channels=3,
        n_channels=32,
        n_classes=num_classes,
        exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        kernel_sizes=[3,5,7],
        deep_supervision=True,
        do_res=True,
        do_res_up_down=True,
        block_counts=[3, 4, 4, 4, 4, 4, 4, 4, 3],
        checkpoint_style='outside_block'
    )
    return model

def create_MedT(pretrained=False, **kwargs):
    model = medt_net(AxialBlock_dynamic,AxialBlock_wopos, [1, 2, 4, 1], s= 0.125,  **kwargs)
    return model

def create_UnetPP(num_classes):

    model = NestedUNet(num_classes)
    return model

def create_UNeXt(num_classes):

    model = UNext(num_classes)
    return model

def create_TransUnet(num_classes):

    model = TransUnet(in_channels=3, img_dim=512, vit_blocks=1, vit_dim_linear_mhsa_block=512, classes=num_classes)
    return model

def create_AttUNet(num_classes):

    model = AttU_Net(output_ch=num_classes)
    return model

def create_UNet3plus(num_classes):

    model = UNet3plus(n_classes=num_classes)
    return model

def create_CMUNeXt_l(num_classes):

    model = CMUNeXt(dims=[32, 64, 128, 256, 512], depths=[1, 1, 1, 6, 3], kernels=[3, 3, 7, 7, 7], input_channel=3, num_classes=num_classes)
    return model
