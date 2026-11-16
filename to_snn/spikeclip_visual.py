"""Self-contained SpikeCLIP rate-coded Spikingformer visual.

Architecture follows SpikeCLIP's training model. Requires
spikingjelly==0.0.0.0.14; uses its standard clock_driven torch backend.
Each call advances one timestep; the artifact owns batch reset boundaries.
"""
import torch
from torch import nn
from torch.nn import functional as F
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

backend = "torch"

class Vision_Tokenizer(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256):
        super().__init__()

        self.block0_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.block0_bn = nn.BatchNorm2d(embed_dims // 8)
        self.block1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        
        self.block1_conv = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn = nn.BatchNorm2d(embed_dims // 4)
        self.block2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        
        self.block2_conv = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn = nn.BatchNorm2d(embed_dims // 2)
        self.block3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        
        self.block3_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block3_conv = nn.Conv2d(embed_dims // 2, embed_dims // 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn = nn.BatchNorm2d(embed_dims // 1)
        self.block4_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        
        self.block4_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.block0_conv(x.flatten(0, 1))
        x = self.block0_bn(x).reshape(T, B, -1, H, W)
        x = self.block1_lif(x).flatten(0, 1)
        
        x = self.block1_conv(x)
        x = self.block1_bn(x).reshape(T, B, -1, H, W)
        x = self.block2_lif(x).flatten(0, 1)
        
        x = self.block2_conv(x)
        x = self.block2_bn(x).reshape(T, B, -1, H, W)
        x = self.block3_mp(x.flatten(0, 1)).reshape(T, B, -1, int(H / 2), int(W / 2))
        x = self.block3_lif(x).flatten(0, 1)
        
        x = self.block3_conv(x)
        x = self.block3_bn(x).reshape(T, B, -1, int(H / 2), int(W / 2))
        x = self.block4_mp(x.flatten(0, 1)).reshape(T, B, -1, int(H / 4), int(W / 4))
        x = self.block4_lif(x).flatten(0, 1)
        
        x = self.block4_conv(x)
        x = self.block4_bn(x).reshape(T, B, -1, int(H / 4), int(W / 4))
        
        return x
   

class Vision_MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)

        self.mlp2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.mlp1_lif(x)
        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)

        x = self.mlp2_lif(x)
        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)
        return x


class Vision_SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=backend)
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_lif(x)

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N)
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * 0.125

        x = x.transpose(3, 4).reshape(T, B, C, N)
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_bn(self.proj_conv(x)).reshape(T, B, C, H, W)
        return x


class Vision_Transformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop_path=None):
        super().__init__()
        self.attn = Vision_SelfAttention(dim, num_heads=num_heads)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Vision_MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SpikeCLIPVisual(nn.Module):
    """Verified training architecture; head count cannot be inferred from weights."""
    training_timesteps = 8

    def __init__(self):
        super().__init__()
        self.patch_embed = Vision_Tokenizer(in_channels=3, embed_dims=384)
        self.block = nn.ModuleList([
            Vision_Transformer(384, num_heads=8, mlp_ratio=4) for _ in range(4)
        ])
        self.proj = nn.Linear(384, 512)

    def reset_state(self):
        for module in self.modules():
            if isinstance(module, MultiStepLIFNode):
                module.reset()

    def forward(self, images):
        images = F.interpolate(images, size=(32, 32), mode="bilinear",
                               align_corners=False)
        x = self.patch_embed(images.unsqueeze(0))
        for block in self.block:
            x = block(x)
        return self.proj(x.flatten(3).mean(3).squeeze(0))
