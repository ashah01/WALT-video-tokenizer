from __future__ import annotations

import copy
from pathlib import Path
from math import log2, ceil, sqrt
from functools import wraps, partial

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from torch.autograd import grad as torch_grad

import torchvision
from torchvision.models import VGG16_Weights

from collections import namedtuple

# from vector_quantize_pytorch import LFQ, FSQ

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from beartype import beartype
from beartype.typing import Tuple, List

from magvit2_pytorch.attend import Attend
from magvit2_pytorch.version import __version__

# from gateloop_transformer import SimpleGateLoopLayer

from taylor_series_linear_attention import TaylorSeriesLinearAttn

from kornia.filters import filter3d

from magvit2_pytorch.arnav_modules import Encoder, Decoder

import pickle

# helper


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def safe_get_index(it, ind, default=None):
    if ind < len(it):
        return it[ind]
    return default


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def identity(t, *args, **kwargs):
    return t


def divisible_by(num, den):
    return (num % den) == 0


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def append_dims(t, ndims: int):
    return t.reshape(*t.shape, *((1,) * ndims))


def is_odd(n):
    return not divisible_by(n, 2)


def maybe_del_attr_(o, attr):
    if hasattr(o, attr):
        delattr(o, attr)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


# tensor helpers


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


def pick_video_frame(video, frame_indices):
    batch, device = video.shape[0], video.device
    video = rearrange(video, "b c f ... -> b f c ...")
    batch_indices = torch.arange(batch, device=device)
    batch_indices = rearrange(batch_indices, "b -> b 1")
    images = video[batch_indices, frame_indices]
    images = rearrange(images, "b 1 c ... -> b c ...")
    return images


# gan related


def gradient_penalty(images, output):
    batch_size = images.shape[0]

    gradients = torch_grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=images.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = rearrange(gradients, "b ... -> b (...)")
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def leaky_relu(p=0.1):
    return nn.LeakyReLU(p)


def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()


def hinge_gen_loss(fake):
    return -fake.mean()


@autocast(enabled=False)
@beartype
def grad_layer_wrt_loss(loss: Tensor, layer: nn.Parameter):
    return torch_grad(
        outputs=loss,
        inputs=layer,
        grad_outputs=torch.ones_like(loss),
        retain_graph=True,
    )[0].detach()


# helper decorators


def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, "vgg")
        if has_vgg:
            vgg = self.vgg
            delattr(self, "vgg")

        out = fn(self, *args, **kwargs)

        if has_vgg:
            self.vgg = vgg

        return out

    return inner


# helper classes


def Sequential(*modules):
    modules = [*filter(exists, modules)]

    if len(modules) == 0:
        return nn.Identity()

    return nn.Sequential(*modules)


class Residual(Module):
    @beartype
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# for a bunch of tensor operations to change tensor to (batch, time, feature dimension) and back


class ToTimeSequence(Module):
    @beartype
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = rearrange(x, "b c f ... -> b ... f c")
        x, ps = pack_one(x, "* n c")

        o = self.fn(x, **kwargs)

        o = unpack_one(o, ps, "* n c")
        return rearrange(o, "b ... f c -> b c f ...")


class SqueezeExcite(Module):
    # global context network - attention-esque squeeze-excite variant (https://arxiv.org/abs/2012.13375)

    def __init__(self, dim, *, dim_out=None, dim_hidden_min=16, init_bias=-10):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.to_k = nn.Conv2d(dim, 1, 1)
        dim_hidden = max(dim_hidden_min, dim_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(dim, dim_hidden, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim_hidden, dim_out, 1),
            nn.Sigmoid(),
        )

        nn.init.zeros_(self.net[-2].weight)
        nn.init.constant_(self.net[-2].bias, init_bias)

    def forward(self, x):
        orig_input, batch = x, x.shape[0]
        is_video = x.ndim == 5

        if is_video:
            x = rearrange(x, "b c f h w -> (b f) c h w")

        context = self.to_k(x)

        context = rearrange(context, "b c h w -> b c (h w)").softmax(dim=-1)
        spatial_flattened_input = rearrange(x, "b c h w -> b c (h w)")

        out = einsum("b i n, b c n -> b c i", context, spatial_flattened_input)
        out = rearrange(out, "... -> ... 1")
        gates = self.net(out)

        if is_video:
            gates = rearrange(gates, "(b f) c h w -> b c f h w", b=batch)

        return gates * orig_input


# token shifting


class TokenShift(Module):
    @beartype
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        x, x_shift = x.chunk(2, dim=1)
        x_shift = pad_at_dim(x_shift, (1, -1), dim=2)  # shift time dimension
        x = torch.cat((x, x_shift), dim=1)
        return self.fn(x, **kwargs)


# rmsnorm


class RMSNorm(Module):
    def __init__(self, dim, channel_first=False, images=False, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return (
            F.normalize(x, dim=(1 if self.channel_first else -1))
            * self.scale
            * self.gamma
            + self.bias
        )


class AdaptiveRMSNorm(Module):
    def __init__(self, dim, *, dim_cond, channel_first=False, images=False, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.dim_cond = dim_cond
        self.channel_first = channel_first
        self.scale = dim**0.5

        self.to_gamma = nn.Linear(dim_cond, dim)
        self.to_bias = nn.Linear(dim_cond, dim) if bias else None

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)

        if bias:
            nn.init.zeros_(self.to_bias.weight)
            nn.init.zeros_(self.to_bias.bias)

    @beartype
    def forward(self, x: Tensor, *, cond: Tensor):
        batch = x.shape[0]
        assert cond.shape == (batch, self.dim_cond)

        gamma = self.to_gamma(cond)

        bias = 0.0
        if exists(self.to_bias):
            bias = self.to_bias(cond)

        if self.channel_first:
            gamma = append_dims(gamma, x.ndim - 2)

            if exists(self.to_bias):
                bias = append_dims(bias, x.ndim - 2)

        return (
            F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * gamma
            + bias
        )


# attention


class Attention(Module):
    @beartype
    def __init__(
        self,
        *,
        dim,
        dim_cond: int | None = None,
        causal=False,
        dim_head=32,
        heads=8,
        flash=False,
        dropout=0.0,
        num_memory_kv=4,
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.need_cond = exists(dim_cond)

        if self.need_cond:
            self.norm = AdaptiveRMSNorm(dim, dim_cond=dim_cond)
        else:
            self.norm = RMSNorm(dim)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange("b n (qkv h d) -> qkv b h n d", qkv=3, h=heads),
        )

        assert num_memory_kv > 0
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_memory_kv, dim_head))

        self.attend = Attend(causal=causal, dropout=dropout, flash=flash)

        self.to_out = nn.Sequential(
            Rearrange("b h n d -> b n (h d)"), nn.Linear(dim_inner, dim, bias=False)
        )

    @beartype
    def forward(self, x, mask: Tensor | None = None, cond: Tensor | None = None):
        maybe_cond_kwargs = dict(cond=cond) if self.need_cond else dict()

        x = self.norm(x, **maybe_cond_kwargs)

        q, k, v = self.to_qkv(x)

        mk, mv = map(lambda t: repeat(t, "h n d -> b h n d", b=q.shape[0]), self.mem_kv)
        k = torch.cat((mk, k), dim=-2)
        v = torch.cat((mv, v), dim=-2)

        out = self.attend(q, k, v, mask=mask)
        return self.to_out(out)


class LinearAttention(Module):
    """
    using the specific linear attention proposed in https://arxiv.org/abs/2106.09681
    """

    @beartype
    def __init__(
        self, *, dim, dim_cond: int | None = None, dim_head=8, heads=8, dropout=0.0
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.need_cond = exists(dim_cond)

        if self.need_cond:
            self.norm = AdaptiveRMSNorm(dim, dim_cond=dim_cond)
        else:
            self.norm = RMSNorm(dim)

        self.attn = TaylorSeriesLinearAttn(dim=dim, dim_head=dim_head, heads=heads)

    def forward(self, x, cond: Tensor | None = None):
        maybe_cond_kwargs = dict(cond=cond) if self.need_cond else dict()

        x = self.norm(x, **maybe_cond_kwargs)

        return self.attn(x)


class LinearSpaceAttention(LinearAttention):
    def forward(self, x, *args, **kwargs):
        x = rearrange(x, "b c ... h w -> b ... h w c")
        x, batch_ps = pack_one(x, "* h w c")
        x, seq_ps = pack_one(x, "b * c")

        x = super().forward(x, *args, **kwargs)

        x = unpack_one(x, seq_ps, "b * c")
        x = unpack_one(x, batch_ps, "* h w c")
        return rearrange(x, "b ... h w c -> b c ... h w")


class SpaceAttention(Attention):
    def forward(self, x, *args, **kwargs):
        x = rearrange(x, "b c t h w -> b t h w c")
        x, batch_ps = pack_one(x, "* h w c")
        x, seq_ps = pack_one(x, "b * c")

        x = super().forward(x, *args, **kwargs)

        x = unpack_one(x, seq_ps, "b * c")
        x = unpack_one(x, batch_ps, "* h w c")
        return rearrange(x, "b t h w c -> b c t h w")


class TimeAttention(Attention):
    def forward(self, x, *args, **kwargs):
        x = rearrange(x, "b c t h w -> b h w t c")
        x, batch_ps = pack_one(x, "* t c")

        x = super().forward(x, *args, **kwargs)

        x = unpack_one(x, batch_ps, "* t c")
        return rearrange(x, "b h w t c -> b c t h w")


class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=1)
        return F.gelu(gate) * x


class FeedForward(Module):
    @beartype
    def __init__(self, dim, *, dim_cond: int | None = None, mult=4, images=False):
        super().__init__()
        conv_klass = nn.Conv2d if images else nn.Conv3d

        rmsnorm_klass = (
            RMSNorm
            if not exists(dim_cond)
            else partial(AdaptiveRMSNorm, dim_cond=dim_cond)
        )

        maybe_adaptive_norm_klass = partial(
            rmsnorm_klass, channel_first=True, images=images
        )

        dim_inner = int(dim * mult * 2 / 3)

        self.norm = maybe_adaptive_norm_klass(dim)

        self.net = Sequential(
            conv_klass(dim, dim_inner * 2, 1), GEGLU(), conv_klass(dim_inner, dim, 1)
        )

    @beartype
    def forward(self, x: Tensor, *, cond: Tensor | None = None):
        maybe_cond_kwargs = dict(cond=cond) if exists(cond) else dict()

        x = self.norm(x, **maybe_cond_kwargs)
        return self.net(x)


# discriminator with anti-aliased downsampling (blurpool Zhang et al.)


class Blur(Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer("f", f)

    def forward(self, x, space_only=False, time_only=False):
        assert not (space_only and time_only)

        f = self.f

        if space_only:
            f = einsum("i, j -> i j", f, f)
            f = rearrange(f, "... -> 1 1 ...")
        elif time_only:
            f = rearrange(f, "f -> 1 f 1 1")
        else:
            f = einsum("i, j, k -> i j k", f, f, f)
            f = rearrange(f, "... -> 1 ...")

        is_images = x.ndim == 4

        if is_images:
            x = rearrange(x, "b c h w -> b c 1 h w")

        out = filter3d(x, f, normalized=True)

        if is_images:
            out = rearrange(out, "b c 1 h w -> b c h w")

        return out


class DiscriminatorBlock(Module):
    def __init__(
        self, input_channels, filters, downsample=True, antialiased_downsample=True
    ):
        super().__init__()
        self.conv_res = nn.Conv2d(
            input_channels, filters, 1, stride=(2 if downsample else 1)
        )

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu(),
        )

        self.maybe_blur = Blur() if antialiased_downsample else None

        self.downsample = (
            nn.Sequential(
                Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
                nn.Conv2d(filters * 4, filters, 1),
            )
            if downsample
            else None
        )

    def forward(self, x):
        res = self.conv_res(x)

        x = self.net(x)

        if exists(self.downsample):
            if exists(self.maybe_blur):
                x = self.maybe_blur(x, space_only=True)

            x = self.downsample(x)

        x = (x + res) * (2**-0.5)
        return x


class Discriminator(Module):
    @beartype
    def __init__(
        self,
        *,
        dim,
        image_size,
        channels=3,
        max_dim=512,
        attn_heads=8,
        attn_dim_head=32,
        linear_attn_dim_head=8,
        linear_attn_heads=16,
        ff_mult=4,
        antialiased_downsample=False,
    ):
        super().__init__()
        image_size = pair(image_size)
        min_image_resolution = min(image_size)

        num_layers = int(log2(min_image_resolution) - 2)

        blocks = []

        layer_dims = [channels] + [(dim * 4) * (2**i) for i in range(num_layers + 1)]
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))

        blocks = []
        attn_blocks = []

        image_resolution = min_image_resolution

        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(layer_dims_in_out) - 1)

            block = DiscriminatorBlock(
                in_chan,
                out_chan,
                downsample=is_not_last,
                antialiased_downsample=antialiased_downsample,
            )

            attn_block = Sequential(
                Residual(
                    LinearSpaceAttention(
                        dim=out_chan,
                        heads=linear_attn_heads,
                        dim_head=linear_attn_dim_head,
                    )
                ),
                Residual(FeedForward(dim=out_chan, mult=ff_mult, images=True)),
            )

            blocks.append(ModuleList([block, attn_block]))

            image_resolution //= 2

        self.blocks = ModuleList(blocks)

        dim_last = layer_dims[-1]

        downsample_factor = 2**num_layers
        last_fmap_size = tuple(map(lambda n: n // downsample_factor, image_size))

        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last

        self.to_logits = Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding=1),
            leaky_relu(),
            Rearrange("b ... -> b (...)"),
            nn.Linear(latent_dim, 1),
            Rearrange("b 1 -> b"),
        )

    def forward(self, x):

        for block, attn_block in self.blocks:
            x = block(x)
            x = attn_block(x)

        return self.to_logits(x)


# modulatable conv from Karras et al. Stylegan2
# for conditioning on latents


class CausalConv3d(Module):
    @beartype
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: int | Tuple[int, int, int],
        pad_mode="constant",
        **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_pad = time_pad
        self.time_causal_padding = (
            width_pad,
            width_pad,
            height_pad,
            height_pad,
            time_pad,
            0,
        )

        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = nn.Conv3d(
            chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs
        )

    def forward(self, x):
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else "constant"

        x = F.pad(x, self.time_causal_padding, mode=pad_mode)
        return self.conv(x)


@beartype
def ResidualUnit(
    dim, kernel_size: int | Tuple[int, int, int], pad_mode: str = "constant"
):
    net = Sequential(
        CausalConv3d(dim, dim, kernel_size, pad_mode=pad_mode),
        nn.ELU(),
        nn.Conv3d(dim, dim, 1),
        nn.ELU(),
        SqueezeExcite(dim),
    )

    return Residual(net)


class CausalConvTranspose3d(Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: int | Tuple[int, int, int],
        *,
        time_stride,
        **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        self.upsample_factor = time_stride

        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        stride = (time_stride, 1, 1)
        padding = (0, height_pad, width_pad)

        self.conv = nn.ConvTranspose3d(
            chan_in, chan_out, kernel_size, stride, padding=padding, **kwargs
        )

    def forward(self, x):
        assert x.ndim == 5
        t = x.shape[2]

        out = self.conv(x)

        out = out[..., : (t * self.upsample_factor), :, :]
        return out


# video tokenizer class

LossBreakdown = namedtuple(
    "LossBreakdown",
    [
        "recon_loss",
        "perceptual_loss",
        "adversarial_gen_loss",
        "adaptive_adversarial_weight",
        "multiscale_gen_losses",
        "multiscale_gen_adaptive_weights",
    ],
)

DiscrLossBreakdown = namedtuple(
    "DiscrLossBreakdown", ["discr_loss", "multiscale_discr_losses", "gradient_penalty"]
)


class VideoTokenizer(Module):
    @beartype
    def __init__(
        self,
        *,
        image_size,
        channels=3,
        init_dim=64,
        num_res_blocks=2,
        z_channels=256,
        ch_mult=(1, 2, 2, 4),
        quantizer_aux_loss_weight=1.0,
        use_fsq=False,
        vgg: Module | None = None,
        vgg_weights: VGG16_Weights = VGG16_Weights.DEFAULT,
        perceptual_loss_weight=1e-1,
        discr_kwargs: dict | None = None,
        multiscale_discrs: Tuple[Module, ...] = tuple(),
        use_gan=True,
        adversarial_loss_weight=1.0,
        grad_penalty_loss_weight=10.0,
        multiscale_adversarial_loss_weight=1.0,
    ):
        super().__init__()

        # for autosaving the config

        _locals = locals()
        _locals.pop("self", None)
        _locals.pop("__class__", None)
        self._configs = pickle.dumps(_locals)

        # image size

        self.channels = channels
        self.image_size = image_size

        # initial encoder
        self.encoder = Encoder(
            ch=init_dim,
            in_channels=3,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            resolution=image_size,
            ch_mult=ch_mult
        )

        self.decoder = Decoder(
            out_ch=3,
            z_channels=z_channels,
            num_res_blocks=num_res_blocks,
            ch=init_dim,
            resolution=image_size,
            ch_mult=ch_mult
        )

        self.time_downsample_factor = 4
        self.time_padding = 4 - 1

        self.fmap_size = image_size / 8

        # quantizer related

        self.use_fsq = use_fsq

        self.quantizer_aux_loss_weight = quantizer_aux_loss_weight

        # dummy loss

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        # perceptual loss related

        use_vgg = channels in {1, 3, 4} and perceptual_loss_weight > 0.0

        self.vgg = None
        self.perceptual_loss_weight = perceptual_loss_weight

        if use_vgg:
            if not exists(vgg):
                vgg = torchvision.models.vgg16(weights=vgg_weights)

                vgg.classifier = Sequential(*vgg.classifier[:-2])

            self.vgg = vgg

        self.use_vgg = use_vgg

        # main flag for whether to use GAN at all

        self.use_gan = use_gan

        # discriminator

        discr_kwargs = default(
            discr_kwargs,
            dict(dim=z_channels, image_size=image_size, channels=channels, max_dim=512),
        )

        self.discr = Discriminator(**discr_kwargs)

        self.adversarial_loss_weight = adversarial_loss_weight
        self.grad_penalty_loss_weight = grad_penalty_loss_weight

        self.has_gan = use_gan and adversarial_loss_weight > 0.0

        # multi-scale discriminators

        self.has_multiscale_gan = use_gan and multiscale_adversarial_loss_weight > 0.0

        self.multiscale_discrs = ModuleList([*multiscale_discrs])

        self.multiscale_adversarial_loss_weight = multiscale_adversarial_loss_weight

        self.has_multiscale_discrs = (
            use_gan
            and multiscale_adversarial_loss_weight > 0.0
            and len(multiscale_discrs) > 0
        )

    @property
    def device(self):
        return self.zero.device

    @classmethod
    def init_and_load_from(cls, path, strict=True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location="cpu")

        assert "config" in pkg, "model configs were not found in this saved checkpoint"

        config = pickle.loads(pkg["config"])
        tokenizer = cls(**config)
        tokenizer.load(path, strict=strict)
        return tokenizer

    def parameters(self):
        return [
            *self.encoder.parameters(),
            *self.decoder.parameters(),
        ]

    def discr_parameters(self):
        return self.discr.parameters()

    def copy_for_eval(self):
        device = self.device
        vae_copy = copy.deepcopy(self.cpu())

        maybe_del_attr_(vae_copy, "discr")
        maybe_del_attr_(vae_copy, "vgg")
        maybe_del_attr_(vae_copy, "multiscale_discrs")

        vae_copy.eval()
        return vae_copy.to(device)

    @remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    @remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def save(self, path, overwrite=True):
        path = Path(path)
        assert overwrite or not path.exists(), f"{str(path)} already exists"

        pkg = dict(
            model_state_dict=self.state_dict(),
            version=__version__,
            config=self._configs,
        )

        torch.save(pkg, str(path))

    def load(self, path, strict=True):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path))
        state_dict = pkg.get("model")
        version = pkg.get("version")

        assert exists(state_dict)

        if exists(version):
            print(f"loading checkpointed tokenizer from version {version}")

        self.load_state_dict(state_dict, strict=strict)

    @beartype
    def encode(
        self,
        video: Tensor,
        quantize=False,
        cond: Tensor | None = None,
        video_contains_first_frame=True,
    ):
        # encoder layers

        return self.encoder(video)

    @beartype
    def decode_from_code_indices(
        self, codes: Tensor, cond: Tensor | None = None, video_contains_first_frame=True
    ):

        return self.decoder(codes)

    @beartype
    def decode(
        self,
        quantized: Tensor,
        cond: Tensor | None = None,
        video_contains_first_frame=True,
    ):
        decode_first_frame_separately = False
        batch = quantized.shape[0]

        # conditioning, if needed

        x = quantized

        x = self.decoder(x)

        return x

    @torch.no_grad()
    def tokenize(self, video):
        self.eval()
        return self.forward(video, return_codes=True)

    @beartype
    def forward(
        self,
        video_or_images: Tensor,
        cond: Tensor | None = None,
        return_loss=False,
        return_codes=False,
        return_recon=False,
        return_discr_loss=False,
        return_recon_loss_only=False,
        apply_gradient_penalty=True,
        video_contains_first_frame=False,
        adversarial_loss_weight=None,
        multiscale_adversarial_loss_weight=None,
    ):
        adversarial_loss_weight = default(
            adversarial_loss_weight, self.adversarial_loss_weight
        )
        multiscale_adversarial_loss_weight = default(
            multiscale_adversarial_loss_weight, self.multiscale_adversarial_loss_weight
        )

        assert (return_loss + return_codes + return_discr_loss) <= 1
        assert video_or_images.ndim in {4, 5}

        assert video_or_images.shape[-2:] == (self.image_size, self.image_size)

        # accept images for image pretraining (curriculum learning from images to video)

        is_image = video_or_images.ndim == 4

        if is_image:
            video = rearrange(video_or_images, "b c ... -> b c 1 ...")
            video_contains_first_frame = True
        else:
            video = video_or_images

        batch, channels, frames = video.shape[:3]

        assert divisible_by(
            frames - int(video_contains_first_frame), self.time_downsample_factor
        ), f"number of frames {frames} minus the first frame ({frames - int(video_contains_first_frame)}) must be divisible by the total downsample factor across time {self.time_downsample_factor}"

        # encoder

        x = self.encode(
            video, cond=cond, video_contains_first_frame=video_contains_first_frame
        )

        if return_codes and not return_recon:
            return x

        # decoder

        recon_video = self.decode(
            x, cond=cond, video_contains_first_frame=video_contains_first_frame
        )

        if return_codes:
            return recon_video
        # reconstruction loss

        if not (return_loss or return_discr_loss or return_recon_loss_only):
            return recon_video

        recon_loss = F.mse_loss(video, recon_video)

        # for validation, only return recon loss

        if return_recon_loss_only:
            return recon_loss, recon_video

        # gan discriminator loss

        if return_discr_loss:
            assert self.has_gan
            assert exists(self.discr)

            # pick a random frame for image discriminator

            frame_indices = torch.randn((batch, frames)).topk(1, dim=-1).indices

            real = pick_video_frame(video, frame_indices)

            if apply_gradient_penalty:
                real = real.requires_grad_()

            fake = pick_video_frame(recon_video, frame_indices)

            real_logits = self.discr(real)
            fake_logits = self.discr(fake.detach())

            discr_loss = hinge_discr_loss(fake_logits, real_logits)

            # multiscale discriminators

            multiscale_discr_losses = []

            if self.has_multiscale_discrs:
                for discr in self.multiscale_discrs:
                    multiscale_real_logits = discr(video)
                    multiscale_fake_logits = discr(recon_video.detach())

                    multiscale_discr_loss = hinge_discr_loss(
                        multiscale_fake_logits, multiscale_real_logits
                    )

                    multiscale_discr_losses.append(multiscale_discr_loss)
            else:
                multiscale_discr_losses.append(self.zero)

            # gradient penalty

            if apply_gradient_penalty:
                gradient_penalty_loss = gradient_penalty(real, real_logits)
            else:
                gradient_penalty_loss = self.zero

            # total loss

            total_loss = (
                discr_loss
                + gradient_penalty_loss * self.grad_penalty_loss_weight
                + sum(multiscale_discr_losses) * self.multiscale_adversarial_loss_weight
            )

            discr_loss_breakdown = DiscrLossBreakdown(
                discr_loss, multiscale_discr_losses, gradient_penalty_loss
            )

            return total_loss, discr_loss_breakdown

        # perceptual loss

        if self.use_vgg:

            frame_indices = torch.randn((batch, frames)).topk(1, dim=-1).indices

            input_vgg_input = pick_video_frame(video, frame_indices)
            recon_vgg_input = pick_video_frame(recon_video, frame_indices)

            if channels == 1:
                input_vgg_input = repeat(input_vgg_input, "b 1 h w -> b c h w", c=3)
                recon_vgg_input = repeat(recon_vgg_input, "b 1 h w -> b c h w", c=3)

            elif channels == 4:
                input_vgg_input = input_vgg_input[:, :3]
                recon_vgg_input = recon_vgg_input[:, :3]

            input_vgg_feats = self.vgg(input_vgg_input)
            recon_vgg_feats = self.vgg(recon_vgg_input)

            perceptual_loss = F.mse_loss(input_vgg_feats, recon_vgg_feats)
        else:
            perceptual_loss = self.zero

        # get gradient with respect to perceptual loss for last decoder layer
        # needed for adaptive weighting

        last_dec_layer = self.decoder.conv_out.weight

        norm_grad_wrt_perceptual_loss = None

        if (
            self.training
            and self.use_vgg
            and (self.has_gan or self.has_multiscale_discrs)
        ):
            norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(
                perceptual_loss, last_dec_layer
            ).norm(p=2)

        # per-frame image discriminator

        recon_video_frames = None

        if self.has_gan:
            frame_indices = torch.randn((batch, frames)).topk(1, dim=-1).indices
            recon_video_frames = pick_video_frame(recon_video, frame_indices)

            fake_logits = self.discr(recon_video_frames)
            gen_loss = hinge_gen_loss(fake_logits)

            adaptive_weight = 1.0

            if exists(norm_grad_wrt_perceptual_loss):
                norm_grad_wrt_gen_loss = grad_layer_wrt_loss(
                    gen_loss, last_dec_layer
                ).norm(p=2)
                adaptive_weight = (
                    norm_grad_wrt_perceptual_loss
                    / norm_grad_wrt_gen_loss.clamp(min=1e-3)
                )
                adaptive_weight.clamp_(max=1e3)

                if torch.isnan(adaptive_weight).any():
                    adaptive_weight = 1.0
        else:
            gen_loss = self.zero
            adaptive_weight = 0.0

        # multiscale discriminator losses

        multiscale_gen_losses = []
        multiscale_gen_adaptive_weights = []

        if self.has_multiscale_gan and self.has_multiscale_discrs:
            if not exists(recon_video_frames):
                recon_video_frames = pick_video_frame(recon_video, frame_indices)

            for discr in self.multiscale_discrs:
                fake_logits = recon_video_frames
                multiscale_gen_loss = hinge_gen_loss(fake_logits)

                multiscale_gen_losses.append(multiscale_gen_loss)

                multiscale_adaptive_weight = 1.0

                if exists(norm_grad_wrt_perceptual_loss):
                    norm_grad_wrt_gen_loss = grad_layer_wrt_loss(
                        multiscale_gen_loss, last_dec_layer
                    ).norm(p=2)
                    multiscale_adaptive_weight = (
                        norm_grad_wrt_perceptual_loss
                        / norm_grad_wrt_gen_loss.clamp(min=1e-5)
                    )
                    multiscale_adaptive_weight.clamp_(max=1e3)

                multiscale_gen_adaptive_weights.append(multiscale_adaptive_weight)

        # calculate total loss

        total_loss = (
            recon_loss
            + perceptual_loss * self.perceptual_loss_weight
            + gen_loss * adaptive_weight * adversarial_loss_weight
        )

        if self.has_multiscale_discrs:

            weighted_multiscale_gen_losses = sum(
                loss * weight
                for loss, weight in zip(
                    multiscale_gen_losses, multiscale_gen_adaptive_weights
                )
            )

            total_loss = (
                total_loss
                + weighted_multiscale_gen_losses * multiscale_adversarial_loss_weight
            )

        # loss breakdown

        loss_breakdown = LossBreakdown(
            recon_loss,
            perceptual_loss,
            gen_loss,
            adaptive_weight,
            multiscale_gen_losses,
            multiscale_gen_adaptive_weights,
        )

        return total_loss, loss_breakdown


# main class


class MagViT2(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
