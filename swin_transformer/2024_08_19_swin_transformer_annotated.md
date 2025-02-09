# Swin Transformer

<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
</head>

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/LspWysWparE" frameborder="0" allowfullscreen></iframe>
</div>

## Acknowledgment:
I borrowed some code from [Swin Transformer github repository](https://github.com/microsoft/Swin-Transformer/tree/main).

## References:
```bibtex
@inproceedings{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

## Install dependencies and load libraries

```bash
!pip install timm
```

```python
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torchvision
import torchvision.transforms as transforms

# tqdm
from tqdm.notebook import tqdm_notebook
```

```python
# Setting the seed for reproducibility
random.seed(42)
g = torch.Generator().manual_seed(2147483647)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

plt.style.use('dark_background')
```

```console
Device: cpu
```

## Swin Transformer classes and helper functions
I copied these classes and functions from:

*   [https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py)
*   [https://github.com/microsoft/Swin-Transformer/blob/main/models/build.py](https://github.com/microsoft/Swin-Transformer/blob/main/models/build.py)
*   [https://github.com/microsoft/Swin-Transformer/blob/main/main.py](https://github.com/microsoft/Swin-Transformer/blob/main/main.py)

```python
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
```

```python
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows
```

```python
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
```

```python
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
```

```python
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        print('---------------Strat   of SwinTransformerBlock---------------')
        print(f'x.shape: {x.shape}')
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        print(f'x.shape after view: {x.shape}')

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        print(f'x_windows.shape: {x_windows.shape}')
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        print(f'x_windows.shape after view: {x_windows.shape}')

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        print(f'attn_windows.shape: {attn_windows.shape}')
        if self.attn_mask is not None:
            print(f'attn_mask.shape: {self.attn_mask.shape}')

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        print(f'attn_windows.shape after view: {attn_windows.shape}')

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        print(f'x.shape after shortcut + drop_path: {x.shape}')

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        print('---------------End     of SwinTransformerBlock---------------')
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops
```

```python
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops
```

```python
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops
```

```python
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
```

```python
class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        print(f'x.shape before training starts: {x.shape}')
        x = self.patch_embed(x)
        print(f'x.shape after PatchEmbed:       {x.shape}')
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        print(f'x.shape after PosEmbed:          {x.shape}')

        i = 0
        for layer in self.layers:
            print(f'---------------Start  of layer {i:03d}---------------')
            x = layer(x)
            print(f'x.shape after layer {i:03d}:           {x.shape}')
            print(f'---------------End    of layer {i:03d}---------------')
            i += 1

        x = self.norm(x)  # B L C
        print(f'x.shape after norm:               {x.shape}')
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        print(f'x.shape after avgpool:            {x.shape}')
        x = torch.flatten(x, 1)
        print(f'x.shape after flatten:            {x.shape}')
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
```

## Initialize the model

```python
model = SwinTransformer(img_size=32,
                          patch_size=2,
                          in_chans=3,
                          num_classes=10,
                          embed_dim=96,
                          depths=[2, 2, 6, 2],
                          num_heads=[3, 6, 12, 24],
                          window_size=4,
                          mlp_ratio=4,
                          qkv_bias=True,
                          qk_scale=None,
                          drop_rate=0.0,
                          drop_path_rate=0.1,
                          ape=False,
                          norm_layer=nn.LayerNorm,
                          patch_norm=True,
                          use_checkpoint=False,
                          fused_window_process=False)

# Transfer to GPU
model.to(device)
# setup the loss function
criterion = nn.CrossEntropyLoss()
# setup the optimizer with the learning rate
model_optimizer = optim.AdamW(model.parameters(), lr=5e-4)
```

```bash
model
```

```console
SwinTransformer(
  (patch_embed): PatchEmbed(
    (proj): Conv2d(3, 96, kernel_size=(2, 2), stride=(2, 2))
    (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
  )
  (pos_drop): Dropout(p=0.0, inplace=False)
  (layers): ModuleList(
    (0): BasicLayer(
      dim=96, input_resolution=(16, 16), depth=2
      (blocks): ModuleList(
        (0): SwinTransformerBlock(
          dim=96, input_resolution=(16, 16), num_heads=3, window_size=4, shift_size=0, mlp_ratio=4
          (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=96, window_size=(4, 4), num_heads=3
            (qkv): Linear(in_features=96, out_features=288, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=96, out_features=96, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): Identity()
          (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=96, out_features=384, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=384, out_features=96, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (1): SwinTransformerBlock(
          dim=96, input_resolution=(16, 16), num_heads=3, window_size=4, shift_size=2, mlp_ratio=4
          (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=96, window_size=(4, 4), num_heads=3
            (qkv): Linear(in_features=96, out_features=288, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=96, out_features=96, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.009)
          (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=96, out_features=384, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=384, out_features=96, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (downsample): PatchMerging(
        input_resolution=(16, 16), dim=96
        (reduction): Linear(in_features=384, out_features=192, bias=False)
        (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      )
    )
    (1): BasicLayer(
      dim=192, input_resolution=(8, 8), depth=2
      (blocks): ModuleList(
        (0): SwinTransformerBlock(
          dim=192, input_resolution=(8, 8), num_heads=6, window_size=4, shift_size=0, mlp_ratio=4
          (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=192, window_size=(4, 4), num_heads=6
            (qkv): Linear(in_features=192, out_features=576, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=192, out_features=192, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.018)
          (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=192, out_features=768, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=768, out_features=192, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (1): SwinTransformerBlock(
          dim=192, input_resolution=(8, 8), num_heads=6, window_size=4, shift_size=2, mlp_ratio=4
          (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=192, window_size=(4, 4), num_heads=6
            (qkv): Linear(in_features=192, out_features=576, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=192, out_features=192, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.027)
          (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=192, out_features=768, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=768, out_features=192, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (downsample): PatchMerging(
        input_resolution=(8, 8), dim=192
        (reduction): Linear(in_features=768, out_features=384, bias=False)
        (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
    )
    (2): BasicLayer(
      dim=384, input_resolution=(4, 4), depth=6
      (blocks): ModuleList(
        (0): SwinTransformerBlock(
          dim=384, input_resolution=(4, 4), num_heads=12, window_size=4, shift_size=0, mlp_ratio=4
          (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=384, window_size=(4, 4), num_heads=12
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.036)
          (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=1536, out_features=384, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (1): SwinTransformerBlock(
          dim=384, input_resolution=(4, 4), num_heads=12, window_size=4, shift_size=0, mlp_ratio=4
          (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=384, window_size=(4, 4), num_heads=12
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.045)
          (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=1536, out_features=384, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (2): SwinTransformerBlock(
          dim=384, input_resolution=(4, 4), num_heads=12, window_size=4, shift_size=0, mlp_ratio=4
          (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=384, window_size=(4, 4), num_heads=12
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.055)
          (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=1536, out_features=384, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (3): SwinTransformerBlock(
          dim=384, input_resolution=(4, 4), num_heads=12, window_size=4, shift_size=0, mlp_ratio=4
          (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=384, window_size=(4, 4), num_heads=12
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.064)
          (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=1536, out_features=384, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (4): SwinTransformerBlock(
          dim=384, input_resolution=(4, 4), num_heads=12, window_size=4, shift_size=0, mlp_ratio=4
          (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=384, window_size=(4, 4), num_heads=12
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.073)
          (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=1536, out_features=384, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (5): SwinTransformerBlock(
          dim=384, input_resolution=(4, 4), num_heads=12, window_size=4, shift_size=0, mlp_ratio=4
          (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=384, window_size=(4, 4), num_heads=12
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.082)
          (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=1536, out_features=384, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (downsample): PatchMerging(
        input_resolution=(4, 4), dim=384
        (reduction): Linear(in_features=1536, out_features=768, bias=False)
        (norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
      )
    )
    (3): BasicLayer(
      dim=768, input_resolution=(2, 2), depth=2
      (blocks): ModuleList(
        (0): SwinTransformerBlock(
          dim=768, input_resolution=(2, 2), num_heads=24, window_size=2, shift_size=0, mlp_ratio=4
          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=768, window_size=(2, 2), num_heads=24
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=768, out_features=768, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.091)
          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (1): SwinTransformerBlock(
          dim=768, input_resolution=(2, 2), num_heads=24, window_size=2, shift_size=0, mlp_ratio=4
          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=768, window_size=(2, 2), num_heads=24
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=768, out_features=768, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.100)
          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
      )
    )
  )
  (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  (avgpool): AdaptiveAvgPool1d(output_size=1)
  (head): Linear(in_features=768, out_features=10, bias=True)
)
```

## Import cifar10 dataset

```python
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# set the preprocess operations to be performed on train/val/test samples
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# download CIFAR10 training set and reserve 50000 for training
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_set, val_set = torch.utils.data.random_split(train_dataset, [40000, 10000])

# download CIFAR10 test set
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# define the data loaders using the datasets
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=128, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=128, shuffle=False)
```

```console
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
100%|██████████| 170498071/170498071 [00:03<00:00, 48346026.44it/s]
Extracting ./data/cifar-10-python.tar.gz to ./data
Files already downloaded and verified
```

## Training

```python
# Make sure gradient tracking is on, and do a pass over the data
model.train(True)
# Training loop
for epoch in range(1):
  for imgs, labels in tqdm_notebook(train_loader, desc='epoch '+str(epoch)):
    # Transfer to GPU
    imgs, labels = imgs.to(device), labels.to(device)
    # zero the parameter gradients
    model_optimizer.zero_grad()
    # Make predictions for this batch
    preds = model(imgs)
    # Compute the loss and its gradients
    loss = criterion(preds, labels)
    # backpropagate the loss
    loss.backward()
    # adjust parameters based on the calculated gradients
    model_optimizer.step()
    break
```

```console
epoch 0:   0%
 0/312 [00:16<?, ?it/s]
x.shape before training starts: torch.Size([128, 3, 32, 32])
x.shape after PatchEmbed:       torch.Size([128, 256, 96])
x.shape after PosEmbed:          torch.Size([128, 256, 96])
---------------Start  of layer 000---------------
---------------Strat   of SwinTransformerBlock---------------
x.shape: torch.Size([128, 256, 96])
x.shape after view: torch.Size([128, 16, 16, 96])
x_windows.shape: torch.Size([2048, 4, 4, 96])
x_windows.shape after view: torch.Size([2048, 16, 96])
attn_windows.shape: torch.Size([2048, 16, 96])
attn_windows.shape after view: torch.Size([2048, 4, 4, 96])
x.shape after shortcut + drop_path: torch.Size([128, 256, 96])
---------------End     of SwinTransformerBlock---------------
---------------Strat   of SwinTransformerBlock---------------
x.shape: torch.Size([128, 256, 96])
x.shape after view: torch.Size([128, 16, 16, 96])
x_windows.shape: torch.Size([2048, 4, 4, 96])
x_windows.shape after view: torch.Size([2048, 16, 96])
attn_windows.shape: torch.Size([2048, 16, 96])
attn_mask.shape: torch.Size([16, 16, 16])
attn_windows.shape after view: torch.Size([2048, 4, 4, 96])
x.shape after shortcut + drop_path: torch.Size([128, 256, 96])
---------------End     of SwinTransformerBlock---------------
x.shape after layer 000:           torch.Size([128, 64, 192])
---------------End    of layer 000---------------
---------------Start  of layer 001---------------
---------------Strat   of SwinTransformerBlock---------------
x.shape: torch.Size([128, 64, 192])
x.shape after view: torch.Size([128, 8, 8, 192])
x_windows.shape: torch.Size([512, 4, 4, 192])
x_windows.shape after view: torch.Size([512, 16, 192])
attn_windows.shape: torch.Size([512, 16, 192])
attn_windows.shape after view: torch.Size([512, 4, 4, 192])
x.shape after shortcut + drop_path: torch.Size([128, 64, 192])
---------------End     of SwinTransformerBlock---------------
---------------Strat   of SwinTransformerBlock---------------
x.shape: torch.Size([128, 64, 192])
x.shape after view: torch.Size([128, 8, 8, 192])
x_windows.shape: torch.Size([512, 4, 4, 192])
x_windows.shape after view: torch.Size([512, 16, 192])
attn_windows.shape: torch.Size([512, 16, 192])
attn_mask.shape: torch.Size([4, 16, 16])
attn_windows.shape after view: torch.Size([512, 4, 4, 192])
x.shape after shortcut + drop_path: torch.Size([128, 64, 192])
---------------End     of SwinTransformerBlock---------------
x.shape after layer 001:           torch.Size([128, 16, 384])
---------------End    of layer 001---------------
---------------Start  of layer 002---------------
---------------Strat   of SwinTransformerBlock---------------
x.shape: torch.Size([128, 16, 384])
x.shape after view: torch.Size([128, 4, 4, 384])
x_windows.shape: torch.Size([128, 4, 4, 384])
x_windows.shape after view: torch.Size([128, 16, 384])
attn_windows.shape: torch.Size([128, 16, 384])
attn_windows.shape after view: torch.Size([128, 4, 4, 384])
x.shape after shortcut + drop_path: torch.Size([128, 16, 384])
---------------End     of SwinTransformerBlock---------------
---------------Strat   of SwinTransformerBlock---------------
x.shape: torch.Size([128, 16, 384])
x.shape after view: torch.Size([128, 4, 4, 384])
x_windows.shape: torch.Size([128, 4, 4, 384])
x_windows.shape after view: torch.Size([128, 16, 384])
attn_windows.shape: torch.Size([128, 16, 384])
attn_windows.shape after view: torch.Size([128, 4, 4, 384])
x.shape after shortcut + drop_path: torch.Size([128, 16, 384])
---------------End     of SwinTransformerBlock---------------
---------------Strat   of SwinTransformerBlock---------------
x.shape: torch.Size([128, 16, 384])
x.shape after view: torch.Size([128, 4, 4, 384])
x_windows.shape: torch.Size([128, 4, 4, 384])
x_windows.shape after view: torch.Size([128, 16, 384])
attn_windows.shape: torch.Size([128, 16, 384])
attn_windows.shape after view: torch.Size([128, 4, 4, 384])
x.shape after shortcut + drop_path: torch.Size([128, 16, 384])
---------------End     of SwinTransformerBlock---------------
---------------Strat   of SwinTransformerBlock---------------
x.shape: torch.Size([128, 16, 384])
x.shape after view: torch.Size([128, 4, 4, 384])
x_windows.shape: torch.Size([128, 4, 4, 384])
x_windows.shape after view: torch.Size([128, 16, 384])
attn_windows.shape: torch.Size([128, 16, 384])
attn_windows.shape after view: torch.Size([128, 4, 4, 384])
x.shape after shortcut + drop_path: torch.Size([128, 16, 384])
---------------End     of SwinTransformerBlock---------------
---------------Strat   of SwinTransformerBlock---------------
x.shape: torch.Size([128, 16, 384])
x.shape after view: torch.Size([128, 4, 4, 384])
x_windows.shape: torch.Size([128, 4, 4, 384])
x_windows.shape after view: torch.Size([128, 16, 384])
attn_windows.shape: torch.Size([128, 16, 384])
attn_windows.shape after view: torch.Size([128, 4, 4, 384])
x.shape after shortcut + drop_path: torch.Size([128, 16, 384])
---------------End     of SwinTransformerBlock---------------
---------------Strat   of SwinTransformerBlock---------------
x.shape: torch.Size([128, 16, 384])
x.shape after view: torch.Size([128, 4, 4, 384])
x_windows.shape: torch.Size([128, 4, 4, 384])
x_windows.shape after view: torch.Size([128, 16, 384])
attn_windows.shape: torch.Size([128, 16, 384])
attn_windows.shape after view: torch.Size([128, 4, 4, 384])
x.shape after shortcut + drop_path: torch.Size([128, 16, 384])
---------------End     of SwinTransformerBlock---------------
x.shape after layer 002:           torch.Size([128, 4, 768])
---------------End    of layer 002---------------
---------------Start  of layer 003---------------
---------------Strat   of SwinTransformerBlock---------------
x.shape: torch.Size([128, 4, 768])
x.shape after view: torch.Size([128, 2, 2, 768])
x_windows.shape: torch.Size([128, 2, 2, 768])
x_windows.shape after view: torch.Size([128, 4, 768])
attn_windows.shape: torch.Size([128, 4, 768])
attn_windows.shape after view: torch.Size([128, 2, 2, 768])
x.shape after shortcut + drop_path: torch.Size([128, 4, 768])
---------------End     of SwinTransformerBlock---------------
---------------Strat   of SwinTransformerBlock---------------
x.shape: torch.Size([128, 4, 768])
x.shape after view: torch.Size([128, 2, 2, 768])
x_windows.shape: torch.Size([128, 2, 2, 768])
x_windows.shape after view: torch.Size([128, 4, 768])
attn_windows.shape: torch.Size([128, 4, 768])
attn_windows.shape after view: torch.Size([128, 2, 2, 768])
x.shape after shortcut + drop_path: torch.Size([128, 4, 768])
---------------End     of SwinTransformerBlock---------------
x.shape after layer 003:           torch.Size([128, 4, 768])
---------------End    of layer 003---------------
x.shape after norm:               torch.Size([128, 4, 768])
x.shape after avgpool:            torch.Size([128, 768, 1])
x.shape after flatten:            torch.Size([128, 768])
```

![image](https://github.com/user-attachments/assets/646c9b54-abcc-49f3-a659-a7e9fbb8e877)

![image](https://github.com/user-attachments/assets/5f6dfba3-0b36-4fd1-8640-758d934380e2)

```python
x = torch.tensor([[0, 1, 2, 3,],
                  [4, 5, 6, 7,],
                  [8, 9, 10, 11,],
                  [12, 13, 14, 15,]])
print(x)

x0 = x[0::2, 0::2]
x1 = x[1::2, 0::2]
x2 = x[0::2, 1::2]
x3 = x[1::2, 1::2]
print(x0)
print(x1)
print(x2)
print(x3)
```

```console
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15]])
tensor([[ 0,  2],
        [ 8, 10]])
tensor([[ 4,  6],
        [12, 14]])
tensor([[ 1,  3],
        [ 9, 11]])
tensor([[ 5,  7],
        [13, 15]])
```

![image](https://github.com/user-attachments/assets/ec70133c-a3ee-4fb5-8333-db0856b5eb06)

## Creating an attention mask based on a shifted window

```python
window_size = 4
shift_size = window_size//2
H = 16
W = 16
img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
h_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
w_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
cnt = 0
for h in h_slices:
    for w in w_slices:
        img_mask[:, h, w, :] = cnt
        cnt += 1

print(f'img_mask.shape: {img_mask.shape}')
mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
print(f'mask_windows.shape: {mask_windows.shape}')
mask_windows_unflaten = mask_windows
mask_windows = mask_windows.view(-1, window_size * window_size)
print(f'mask_windows.shape after view: {mask_windows.shape}')
attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # num_windows, Wh*Ww, Wh*Ww
print(f'attn_mask.shape: {attn_mask.shape}')
attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
print(f'attn_mask.shape after masked_fill: {attn_mask.shape}')
```

```console
img_mask.shape: torch.Size([1, 16, 16, 1])
mask_windows.shape: torch.Size([16, 4, 4, 1])
mask_windows.shape after view: torch.Size([16, 16])
attn_mask.shape: torch.Size([16, 16, 16])
attn_mask.shape after masked_fill: torch.Size([16, 16, 16])
```

```python
print(mask_windows.unsqueeze(1).shape)
print(mask_windows.unsqueeze(2).shape)
```

```console
torch.Size([16, 1, 16])
torch.Size([16, 16, 1])
```

```python
plt.imshow(img_mask.squeeze(0).squeeze(-1).detach().numpy())
plt.colorbar()
```

![image](https://github.com/user-attachments/assets/ba003bd9-2638-4cfd-8ee9-602a739ad6de)

```python
fig, axs = plt.subplots(4, mask_windows_unflaten.shape[0]//4, figsize=(6, 6))
for i, ax in enumerate(axs.ravel()):
  ax.imshow(mask_windows_unflaten[i,:,:].detach().numpy())
```

![image](https://github.com/user-attachments/assets/dff61b27-bbea-44aa-b302-071790f8f228)


<script>
  document.addEventListener("DOMContentLoaded", function() {
    renderMathInElement(document.body, {
      delimiters: [
        {left: '$$', right: '$$', display: true}, // Display math (e.g., equations on their own line)
        {left: '$', right: '$', display: false},  // Inline math (e.g., within a sentence)
        {left: '\\(', right: '\\)', display: false}, // Another way to write inline math
        {left: '\\[', right: '\\]', display: true}   // Another way to write display math
      ]
    });
  });
</script>

