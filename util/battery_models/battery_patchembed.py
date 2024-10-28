from torch import nn as nn
import torch
from loguru import logger
# from helpers import to_2tuple
from .trace_utils import _assert

class Battery_PatchEmbed(nn.Module):
    """
    battery data to Patch Embedding
    input: 196 * 10 * 11
    output: 196 * 768
    """
    def __init__(
            self,
            tokens_len=196,
            patch_size=7,
            embed_dim=768,
            features_num = 18,
            norm_layer=None,
            bias=True,
    ):
        super().__init__()
        self.patch_size = patch_size # 2
        self.features_num = features_num  # 10
        self.num_patches = tokens_len  # 39
        # 不进行重叠
        self.proj = nn.Conv1d(in_channels=features_num, out_channels=embed_dim, kernel_size=(patch_size,), stride=(patch_size,),bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()


    def forward(self, x):
        B, L, C = x.shape #  batch_size * (39 * 2) * 10

        _assert(L == self.num_patches * self.patch_size, f"Input data length ({L}) doesn't match model ({self.num_patches * self.patch_size}).")
        x = x.permute(0, 2, 1)  # batch * 10 * (39 * 2)
        x = self.proj(x)
        x = x.permute(0, 2, 1)   # batch * 49 * 768
        # logger.info(x.shape)
        x = self.norm(x)
        return x

