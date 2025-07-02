import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop)
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        # x: (B, N, C)
        x_ = self.norm1(x)
        x_ = x_.transpose(0, 1)  # (N, B, C) for MultiheadAttention
        attn_out, _ = self.attn(x_, x_, x_)
        attn_out = attn_out.transpose(0, 1)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MaskedBlock(nn.Module):
    """
    包装一个Block, 为其每个参数引入可学习的mask logits。
    mask logits经过Gumbel-Softmax采样, 得到mask后与参数加权。
    """
    def __init__(self, block, temperature = 1.0, threshold=0.5):
        super().__init__()
        self.block = block
        self.temperature = temperature
        self.threshold = threshold
        self.mask_logits = nn.ParameterDict()
        for name, param in self.block.named_parameters():
            if param.requires_grad:
                safe_name = name.replace('.', '_')  # 替换点号，避免PyTorch报错
                self.mask_logits[safe_name] = nn.Parameter(
                    torch.randn_like(param) * 0.01, requires_grad=True
                )

    def forward(self, x):
        original_params = {}
        for name, param in self.block.named_parameters():
            safe_name = name.replace('.', '_')
            if safe_name in self.mask_logits:
                logits = self.mask_logits[safe_name]
                # Gumbel-Softmax采样，得到概率mask（0~1之间）
                mask = F.gumbel_softmax(
                    torch.stack([logits, torch.zeros_like(logits)], dim=-1),
                    tau=self.temperature, hard=False
                )[..., 0]
                original_params[name] = param.data.clone()
                param.data = param.data * mask
        out = self.block(x)
        # 恢复原始参数
        for name, param in self.block.named_parameters():
            if name in original_params:
                param.data = original_params[name]
        return out

    def get_masked_parameters(self):
        """
        训练结束后,sigmoid激活mask logits并与阈值比较,得到二值mask。
        返回被mask筛选后的参数。
        """
        masked_params = {}
        for name, param in self.block.named_parameters():
            safe_name = name.replace('.', '_')
            if safe_name in self.mask_logits:
                mask = (torch.sigmoid(self.mask_logits[safe_name]) > self.threshold).float()
                masked_params[name] = param.data * mask
            else:
                masked_params[name] = param.data.clone()
        return masked_params

    def update_temperature(self, new_temp):
        self.temperature = new_temp

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm,
                 use_cls_token=True, mask_training=False, temperature=1.0, threshold=0.5, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_cls_token = use_cls_token
        self.mask_training = mask_training
        self.temperature = temperature
        self.threshold = threshold

       # ----------- 新增 patch embedding 相关代码 -----------
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if mask_training:
            standard_blocks = nn.ModuleList([
                Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                      attn_drop=attn_drop_rate, drop_path=drop_path_rate,
                      norm_layer=norm_layer)
                for _ in range(depth)])
            self.blocks = nn.ModuleList([
                MaskedBlock(block, temperature=temperature, threshold=threshold)
                for block in standard_blocks
            ])
        else:
            self.blocks = nn.ModuleList([
                Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                      attn_drop=attn_drop_rate, drop_path=drop_path_rate,
                      norm_layer=norm_layer)
                for _ in range(depth)])
            
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # 初始化参数
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        # x = self.patch_embed(x)
        # if self.use_cls_token:
        #     cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        #     x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed
        # x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        # x = self.norm(x)
        # if self.use_cls_token:
        #     return x[:, 0]
        # else:
        #     return x.mean(dim=1)
        return x
    
    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        # ...后续流程...
        return self.head(self.forward_features(x))

    def get_masked_parameters(self):
        masked_params = {}
        for i, blk in enumerate(self.blocks):
            if hasattr(blk, 'get_masked_parameters'):
                blk_params = blk.get_masked_parameters()
                for name, param in blk_params.items():
                    masked_params[f'blocks.{i}.{name}'] = param
            else:
                for name, param in blk.named_parameters():
                    masked_params[f'blocks.{i}.{name}'] = param
        return masked_params