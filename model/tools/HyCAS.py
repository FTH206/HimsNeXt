import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
"""
HyCAS (Hybrid Contextual Attention and Selection)

"""
class HybridGatingMechanism(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.feature_branch = nn.Sequential(
            nn.Conv2d(dim, dim//4, 1),
            nn.ReLU(True),
            nn.Conv2d(dim//4, dim, 1)
        )

        self.gate_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//8, 1),
            nn.ReLU(True),
            nn.Conv2d(dim//8, dim, 1),
            nn.Sigmoid()
        )

        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        features = self.feature_branch(x)
        gates = self.gate_branch(x)

        return x * (self.gamma * gates + (1 - self.gamma) * features)

class CASWithHybridGate(nn.Module):
    def __init__(self, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)

        self.pos_embedding = nn.Parameter(torch.zeros(1, dim, 32, 32))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.offset_predictor = nn.Sequential(
            nn.Conv2d(dim, dim//4, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(dim//4, 2, 3, padding=1)  # Predict offsets in the x and y directions.
        )
        
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.hybrid_gate = HybridGatingMechanism(dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape

        offset = self.offset_predictor(x)  # B, 2, H, W

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'  # Use matrix indexing for the sampling grid.
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # B, H, W, 2

        offset = offset.permute(0, 2, 3, 1)  # B, H, W, 2
        offset = offset * 0.1
        sampling_grid = grid + offset

        if H != self.pos_embedding.shape[2] or W != self.pos_embedding.shape[3]:
            pos_embed = F.interpolate(self.pos_embedding, size=(H, W), mode='bilinear', align_corners=True)
        else:
            pos_embed = self.pos_embedding
            
        pos_embed = pos_embed.expand(B, -1, -1, -1)
        deformed_pos_embed = F.grid_sample(pos_embed, sampling_grid, align_corners=True, mode='bilinear')

        x_with_pos = x + deformed_pos_embed

        q, k, v = self.qkv(x_with_pos).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)

        f_sim = torch.matmul(q.flatten(2), k.flatten(2).transpose(-1, -2)) / (q.size(-1) ** 0.5)
        f_sim = f_sim.softmax(dim=-1)

        context = torch.matmul(f_sim, v.flatten(2)).view_as(v)

        context = self.hybrid_gate(context)

        out = self.proj(context)
        out = self.proj_drop(out)
        return out

class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

if __name__ == '__main__':

    input = torch.randn(1, 512, 64, 64)

    model = CASWithHybridGate(dim=512)

    output = model(input)
    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")
