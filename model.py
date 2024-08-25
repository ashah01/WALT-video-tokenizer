import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def swish(x):
    # swish
    return x * torch.sigmoid(x)


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, num_groups, use_conv_shortcut=False) -> None:
        super().__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = nn.GroupNorm(num_groups, in_filters, eps=1e-6)
        self.norm2 = nn.GroupNorm(num_groups, out_filters, eps=1e-6)

        self.conv1 = nn.Conv3d(
            in_filters, out_filters, kernel_size=(3, 3, 3), padding="same", bias=False
        )
        self.conv2 = nn.Conv3d(
            out_filters, out_filters, kernel_size=(3, 3, 3), padding="same", bias=False
        )

        if in_filters != out_filters:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv3d(
                    in_filters,
                    out_filters,
                    kernel_size=(3, 3, 3),
                    padding="same",
                    bias=False,
                )
            else:
                self.nin_shortcut = nn.Conv3d(
                    in_filters,
                    out_filters,
                    kernel_size=(1, 1, 1),
                    padding="same",
                    bias=False,
                )

    def forward(self, x, **kwargs):
        residual = x

        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_filters != self.out_filters:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return x + residual


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        in_channels,
        num_res_blocks,
        z_channels,
        ch_mult=(1, 2, 2, 4),
    ):
        super().__init__()

        self.z_channels = z_channels

        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(ch_mult)
        self.strides = (1, 2, 2)

        self.conv_in = nn.Conv3d(
            in_channels, ch, kernel_size=(3, 3, 3), padding="same", bias=False
        )

        ## construct the model
        self.down = nn.ModuleList()

        in_ch_mult = (1,) + tuple(ch_mult)
        for i_level in range(self.num_blocks):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]  # [1, 1, 2, 2, 4]
            block_out = ch * ch_mult[i_level]  # [1, 2, 2, 4]
            for _ in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out, ch // 2))
                block_in = block_out

            down = nn.Module()
            down.block = block
            if i_level < self.num_blocks - 1:
                down.downsample = nn.Conv3d(
                    block_out,
                    block_out,
                    kernel_size=(3, 3, 3),
                    stride=(self.strides[i_level], 2, 2),
                    padding=(1, 1, 1),
                )

            self.down.append(down)

        ### mid
        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in, ch // 2))

        ### end
        self.norm_out = nn.GroupNorm(ch // 2, block_out, eps=1e-6)
        self.conv_out = nn.Conv3d(
            block_out, z_channels, kernel_size=(1, 1, 1), padding="same"
        )

    def forward(self, x):

        ## down
        x = self.conv_in(x)
        for i_level in range(self.num_blocks):
            for i_block in range(self.num_res_blocks):
                x = self.down[i_level].block[i_block](x)

            if i_level < self.num_blocks - 1:
                x = self.down[i_level].downsample(x)

        ## mid
        for res in range(self.num_res_blocks):
            x = self.mid_block[res](x)

        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        num_res_blocks,
        z_channels,
        ch_mult=(1, 2, 2, 4),
    ) -> None:
        super().__init__()

        self.num_blocks = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.depths = (1, 2, 2)

        block_in = ch * ch_mult[self.num_blocks - 1]

        self.conv_in = nn.Conv3d(
            z_channels, block_in, kernel_size=(3, 3, 3), padding="same", bias=True
        )

        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in, ch // 2))

        self.up = nn.ModuleList()

        for i_level in reversed(range(self.num_blocks)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out, ch // 2))
                block_in = block_out

            up = nn.Module()
            up.block = block
            if i_level > 0:
                up.upsample = Upsampler(block_in, self.depths[i_level - 1])
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(ch // 2, block_in, eps=1e-6)

        self.conv_out = nn.Conv3d(
            block_in, out_ch, kernel_size=(3, 3, 3), padding="same"
        )

    def forward(self, z):
        z = self.conv_in(z)

        ## mid
        for res in range(self.num_res_blocks):
            z = self.mid_block[res](z)

        ## upsample
        for i_level in reversed(range(self.num_blocks)):
            for i_block in range(self.num_res_blocks):
                z = self.up[i_level].block[i_block](z)

            if i_level > 0:
                z = self.up[i_level].upsample(z)

        z = self.norm_out(z)
        z = swish(z)
        z = self.conv_out(z)

        return z


def depth_to_space(block_size: int, x: torch.Tensor) -> torch.Tensor:
    """Depth-to-Space DCR mode (depth-column-row) core implementation.

    Args:
        x (torch.Tensor): input tensor. The channel-first (*CTHW) layout is supported.
        block_size (tuple): block side sizes
    """
    # check inputs

    block_t, block_h, block_w = block_size
    c, t, h, w = x.shape[-4:]

    s = block_t * block_h * block_w
    if c % s != 0:
        raise ValueError(
            f"Expecting a channels-first (CTHW) tensor with C divisible by {s}, but got C={c} channels"
        )

    outer_dims = x.shape[:-4]

    # splitting three additional dimensions from the channel dimension
    x = x.view(-1, block_t, block_h, block_w, c // s, t, h, w)

    # putting the two new dimensions along H and W
    # x = x.permute(0, 3, 4, 1, 5, 2)
    x = x.permute(0, 4, 5, 1, 6, 2, 7, 3)

    # merging the two new dimensions with H and W
    x = x.contiguous().view(*outer_dims, c // s, t * block_t, h * block_h, w * block_w)

    return x


class Upsampler(nn.Module):
    def __init__(self, dim, depth, dim_out=None):
        super().__init__()
        dim_out = dim * (depth * 4)
        self.conv1 = nn.Conv3d(dim, dim_out, (3, 3, 3), padding="same")
        self.depth2space = partial(depth_to_space, (depth, 2, 2))

    def forward(self, x):
        """
        input_image: [B C H W]
        """
        out = self.conv1(x)
        out = self.depth2space(out)
        return out
