# =========================
# REPLACEMENT: SwinTransformerSys (3-branch encoder + bottleneck fusion)
# C_base = 48, C1 = 12, C2 = 192
# =========================

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class SwinTransformerSys(nn.Module):
    """
    Multi-branch Swin-UNet-like model:
    - Base branch: patch_size=4, embed_dim=C=48  -> bottleneck dim = 8C = 384 (W/32,H/32)
    - cp1 branch : patch_size=2, embed_dim=C1=12 -> bottleneck dim = 16C1 = 192 (W/32,H/32)
    - cp2 branch : patch_size=8, embed_dim=C2=192-> bottleneck dim = 4C2 = 768 (W/32,H/32)

    Fusion at bottleneck:
      concat([base, cp1, cp2]) -> Linear -> back to base bottleneck dim (384)
    Decoder:
      keep EXACTLY the same decoder logic as your original model (skip connections from base only).
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,

        embed_dim=48,                          # <-- C = 48 đặt cố định nhánh gốc 
        depths=[2, 2, 2, 2],                   
        num_heads=[3, 6, 12, 24],

        # decoder config 
        depths_decoder=[1, 2, 2, 2],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        final_upsample="expand_first",
        **kwargs
    ):
        super().__init__()

        # ===== constants from user =====
        self.C_base = 48
        self.C1 = 12
        self.C2 = 192

        # enforce base embed_dim = 48 
        embed_dim = self.C_base

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # base bottleneck dim = embed_dim * 2^(num_layers-1)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))  # 48 * 8 = 384 if num_layers=4
        base_bottleneck_dim = self.num_features

        # =========================
        # 1) BASE PATCH EMBED (ps=4, C=48)
        # =========================
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,  # default 4
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution  # (H/4, W/4)
        self.patches_resolution = patches_resolution

        # absolute position embedding 
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth for base
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # =========================
        # 2) BASE ENCODER
        # =========================
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.norm = norm_layer(base_bottleneck_dim)

        # =========================
        # 3) CP1 BRANCH (ps=2, C1=12)  -> W/2 start, 4 merges -> W/32, dim=192
        # =========================
        self.patch_embed_cp1 = PatchEmbed(
            img_size=img_size,
            patch_size=2,
            in_chans=in_chans,
            embed_dim=self.C1,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        patches_resolution_cp1 = self.patch_embed_cp1.patches_resolution  # (H/2, W/2)

        depths_cp1 = [2, 2, 2, 2, 1]  # x2,x2,x2,x2 then bottleneck x1 
        # heads for cp1 stages (W/2, W/4, W/8, W/16, W/32)
        num_heads_cp1 = [3, 3, 6, 12, 24]

        # stochastic depth for cp1
        dpr_cp1 = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_cp1))]

        self.layers_cp1 = nn.ModuleList()
        for i_layer in range(len(depths_cp1)):
            layer = BasicLayer(
                dim=int(self.C1 * 2 ** i_layer),  # 12,24,48,96,192
                input_resolution=(
                    patches_resolution_cp1[0] // (2 ** i_layer),
                    patches_resolution_cp1[1] // (2 ** i_layer),
                ),
                depth=depths_cp1[i_layer],
                num_heads=num_heads_cp1[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_cp1[sum(depths_cp1[:i_layer]): sum(depths_cp1[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < len(depths_cp1) - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_cp1.append(layer)

        cp1_bottleneck_dim = int(self.C1 * 2 ** (len(depths_cp1) - 1))  # 12 * 16 = 192
        self.norm_cp1 = norm_layer(cp1_bottleneck_dim)

        # =========================
        # 4) CP2 BRANCH (ps=8, C2=192) -> W/8 start, 2 merges -> W/32, dim=768
        # =========================
        self.patch_embed_cp2 = PatchEmbed(
            img_size=img_size,
            patch_size=8,
            in_chans=in_chans,
            embed_dim=self.C2,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        patches_resolution_cp2 = self.patch_embed_cp2.patches_resolution  # (H/8, W/8)

        depths_cp2 = [2, 2, 1]  # x2, x2, bottleneck x1
        num_heads_cp2 = [6, 12, 24]  # (W/8, W/16, W/32)

        dpr_cp2 = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_cp2))]

        self.layers_cp2 = nn.ModuleList()
        for i_layer in range(len(depths_cp2)):
            layer = BasicLayer(
                dim=int(self.C2 * 2 ** i_layer),  # 192,384,768
                input_resolution=(
                    patches_resolution_cp2[0] // (2 ** i_layer),
                    patches_resolution_cp2[1] // (2 ** i_layer),
                ),
                depth=depths_cp2[i_layer],
                num_heads=num_heads_cp2[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_cp2[sum(depths_cp2[:i_layer]): sum(depths_cp2[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < len(depths_cp2) - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_cp2.append(layer)

        cp2_bottleneck_dim = int(self.C2 * 2 ** (len(depths_cp2) - 1))  # 192 * 4 = 768
        self.norm_cp2 = norm_layer(cp2_bottleneck_dim)

        # =========================
        # 5) BOTTLENECK FUSION -> project back to base bottleneck dim (384)
        # =========================
        fuse_in_dim = base_bottleneck_dim + cp1_bottleneck_dim + cp2_bottleneck_dim  # 384+192+768=1344
        self.fuse_to_base = nn.Linear(fuse_in_dim, base_bottleneck_dim)

        # =========================
        # 6) DECODER 
        # =========================
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        for i_layer in range(self.num_layers):
            concat_linear = (
                nn.Linear(
                    2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                )
                if i_layer > 0
                else nn.Identity()
            )

            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(
                        patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    dim_scale=2,
                    norm_layer=norm_layer,
                )
            else:
                layer_up = BasicLayer_up(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[: (self.num_layers - 1 - i_layer)]):
                        sum(depths[: (self.num_layers - 1 - i_layer) + 1])
                    ],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                )

            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm_up = norm_layer(self.embed_dim)

        # final upsample head (unchanged)
        if self.final_upsample == "expand_first":
            self.up = FinalPatchExpand_X4(
                input_resolution=(img_size // patch_size, img_size // patch_size),
                dim_scale=4,
                dim=embed_dim,
            )
            self.output = nn.Conv2d(
                in_channels=embed_dim,
                out_channels=self.num_classes,
                kernel_size=1,
                bias=False,
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    # ===== Base encoder (returns skip list) =====
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)  # skips from BASE only
            x = layer(x)

        x = self.norm(x)  # (B, L32, 8C=384)
        return x, x_downsample

    # ===== cp1 encoder (bottleneck only) =====
    def forward_features_cp1(self, x):
        x = self.patch_embed_cp1(x)
        # (optional) could add its own pos_drop; keep simple:
        for layer in self.layers_cp1:
            x = layer(x)
        x = self.norm_cp1(x)  # (B, L32, 192)
        return x

    # ===== cp2 encoder (bottleneck only) =====
    def forward_features_cp2(self, x):
        x = self.patch_embed_cp2(x)
        for layer in self.layers_cp2:
            x = layer(x)
        x = self.norm_cp2(x)  # (B, L32, 768)
        return x

    # ===== Decoder (unchanged) =====
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], dim=-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)
        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"
        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)
            x = self.output(x)
        return x

    def forward(self, x):
        # base branch (with skips)
        x_base, x_downsample = self.forward_features(x)

        # cp branches (bottlenecks only)
        x_cp1 = self.forward_features_cp1(x)
        x_cp2 = self.forward_features_cp2(x)

        # bottleneck fusion -> back to base dim (384)
        x_fused = torch.cat([x_base, x_cp1, x_cp2], dim=-1)
        x = self.fuse_to_base(x_fused)

        # decoder stays as original, uses base skips
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
