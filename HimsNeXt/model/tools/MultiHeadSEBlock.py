import torch
import torch.nn as nn

class MultiHeadSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, num_heads=4, use_spatial=True):
        super(MultiHeadSEBlock, self).__init__()
        assert in_channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(self.head_dim, self.head_dim // reduction, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.head_dim // reduction)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(self.head_dim // reduction, self.head_dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.head_dim)
        self.sigmoid = nn.Sigmoid()
        self.use_spatial = use_spatial
        if use_spatial:
            self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
            self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        x_split = x.view(b * self.num_heads, self.head_dim, h, w)
        y = self.global_avg_pool(x_split)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.gelu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.sigmoid(y)
        x_split = x_split * y
        x_out = x_split.view(b, c, h, w)
        if self.use_spatial:
            avg_out = torch.mean(x_out, dim=1, keepdim=True)
            max_out, _ = torch.max(x_out, dim=1, keepdim=True)
            spatial_att = torch.cat([avg_out, max_out], dim=1)
            spatial_att = self.spatial_conv(spatial_att)
            spatial_att = self.spatial_sigmoid(spatial_att)
            x_out = x_out * spatial_att
        x_out = x_out + x
        return x_out

# MultiHeadSEBlock
if __name__ == "__main__":
    input_tensor = torch.randn(8, 64, 32, 32)
    se_block = MultiHeadSEBlock(in_channels=64, reduction=16, num_heads=4)
    output_tensor = se_block(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
