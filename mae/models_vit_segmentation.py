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

import timm.models.vision_transformer
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed_with_resolution
from lib.fpn import FCNHead, FPNHead
from lib.gpt import Block as GPTBlock




class PatchEmbedUnSafe(PatchEmbed):
    """Image to Patch Embedding"""

    def forward(self, x):
        B, C, H, W = x.shape
        # Dropped size check in timm
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self, global_pool=False, patch_size=16, in_chans=3, embed_dim=1024, decoder_embed_dim=512,norm_layer=nn.LayerNorm, **kwargs
    ):
        super().__init__(embed_dim=embed_dim, **kwargs)

        self.patch_embed = PatchEmbedUnSafe(
            img_size=kwargs["img_size"],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.fpn = FPNHead(decoder_embed_dim, share_weights=False)
        self.multiscale=False
        # Depending on the mode of decoding we are using, the decoder architecture is different
        decoder_num_heads = 16
        mlp_ratio = 4
        decoder_depth = 8
        if self.multiscale:
            self.decoder_blocks = nn.ModuleList(
                [
                    GPTBlock(
                        decoder_embed_dim,
                        decoder_num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                    )
                    for _ in range(decoder_depth)
                ]
            )
        else:
            self.decoder_blocks = nn.ModuleList(
                [
                    Block(
                        decoder_embed_dim,
                        decoder_num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                    )
                    for _ in range(decoder_depth)
                ]
            )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        fcn_dim = 256
        fcn_layers = 3
        self.fcn = FCNHead(decoder_embed_dim, fcn_dim, fcn_layers, 3)
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_features(self, x, input_res=None):
        B, _, h, w = x.shape
        print('Step 0: '+str(x.shape))
        x = self.patch_embed(x)
        print('After patch embedding: '+str(x.shape))
        input_res = input_res.cpu()

        num_patches = int(
            (h * w) / (self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1])
        )
        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            x.shape[-1],
            int(num_patches**0.5),
            input_res,
            cls_token=True,
            device=x.device,
        )

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        print('add cls tokens: '+str(x.shape))
        x = x + pos_embed
        x = self.pos_drop(x)
        print('pos drop: '+str(x.shape))

        for blk in self.blocks:
            x = blk(x)
        print('after transformer blocks: '+str(x.shape))
        
        # Added back to the mask token in decoder for decoding modes != "demasking"
        pos_embed_encoder = get_2d_sincos_pos_embed_with_resolution(
            self.decoder_embed_dim,
            int(num_patches**0.5),
            input_res,
            cls_token=True,
            device=x.device,
        )

        # forward decoder
        # embed tokens
        x = self.decoder_embed(x)  # N X L X d_emb_decoder
        print('after decoder embed: '+str(x.shape))
        n, l, d = pos_embed_encoder.shape
        l_dim = int((l - 1) ** 0.5)

        target_dim = h
        target_res = torch.Tensor(1)
        target_res = torch.ones_like(target_res).to(target_res.device)
        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            x.shape[-1], target_dim, target_res, cls_token=True, device=x.device
        )

        x = x + pos_embed_encoder
        print('add pos embed decoder: '+str(x.shape))

        pos_embed_raw = pos_embed
        ids = None
        x = x[:, 1:, :]
        print('removed one channel: '+str(x.shape))
        n, p_2, d = x.shape
        p = int(p_2**0.5)
        for blk in self.decoder_blocks:
            x = blk(x)
        print('after decoder blocks: '+str(x.shape))
        x = self.decoder_norm(x)
        print('after decoder norm: '+str(x.shape))
        # standard decoder
        x = x.view(n, p, p, d).permute(0, 3, 1, 2).contiguous()  # B X C X H X W
        print('after reshape: '+str(x.shape))
    


        

        '''if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        


        return outcome'''
        return x

    def forward(self, x, input_res=None):
        x = self.forward_features(x, input_res=input_res)
        print(x.shape)
        x = self.unpatchify(x)
        print(x.shape)
        x = self.head(x)
        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
