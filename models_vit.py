# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from loguru import  logger
import timm.models.vision_transformer

from battery_models.battery_patchembed import Battery_PatchEmbed


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self,tokens_len=49, patch_size=2, features_num=7, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        embed_dim = kwargs['embed_dim']
        self.patch_embed = Battery_PatchEmbed(tokens_len, patch_size, embed_dim, features_num)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            # embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]


        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # 这里什么鬼？
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def battery_mae_vit_base(**kwargs):
    model = VisionTransformer(tokens_len=49, patch_size=2,
         embed_dim=768, features_num = 7, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def battery_mae_vit_small(**kwargs):
    logger.error(kwargs)
    model = VisionTransformer(tokens_len=49, patch_size=1,
         embed_dim=768, features_num = 7, depth=6, num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def battery_mae_vit_tiny(**kwargs):
    model = VisionTransformer(tokens_len=49, patch_size=1,
         embed_dim=768, features_num = 7, depth=3, num_heads=3, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def battery_mae_vit_large(**kwargs):
    """
       gpu005
       :param kwargs:
       :return:
       """
    model = VisionTransformer(tokens_len=49, patch_size=1,
         embed_dim=1024, features_num = 7, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def battery_mae_vit_huge(**kwargs):
    """
       gpu005
       :param kwargs:
       :return:
       """
    model = VisionTransformer(tokens_len=49, patch_size=1,
         embed_dim=1280, features_num = 7, depth=32, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



