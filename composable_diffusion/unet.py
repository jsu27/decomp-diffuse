import math
from abc import abstractmethod
from collections import OrderedDict

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import avg_pool_nd, conv_nd, linear, normalization, timestep_embedding, zero_module


def swish(x):
    """Allegedly better activation func"""
    return x * th.sigmoid(x)

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, encoder_out=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, AttentionBlock):
                x = layer(x, encoder_out)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


# TODO check if this matches ResBlock/how to modify
class CondResBlockNoLatent(nn.Module):
    def __init__(self, downsample=True, rescale=True, filters=64, upsample=False):
        super(CondResBlockNoLatent, self).__init__()

        self.filters = filters
        self.downsample = downsample

        if filters <= 128:
            self.bn1 = nn.GroupNorm(int(32  * filters / 128), filters, affine=True)
        else:
            self.bn1 = nn.GroupNorm(32, filters, affine=False)

        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        if filters <= 128:
            self.bn2 = nn.GroupNorm(int(32 * filters / 128), filters, affine=True)
        else:
            self.bn2 = nn.GroupNorm(32, filters, affine=True)

        self.upsample = upsample
        self.upsample_module = nn.Upsample(scale_factor=2)
        # Upscale to mask of image
        if downsample:
            if rescale:
                self.conv_downsample = nn.Conv2d(filters, 2 * filters, kernel_size=3, stride=1, padding=1)
            else:
                self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

            self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

        if upsample:
            self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_orig = x


        x = self.conv1(x)
        x = swish(x)

        x = self.conv2(x)
        x = swish(x)

        x_out = x_orig + x

        if self.upsample:
            x_out = self.upsample_module(x_out)
            x_out = swish(self.conv_downsample(x_out))

        if self.downsample:
            x_out = swish(self.conv_downsample(x_out))
            x_out = self.avg_pool(x_out)

        return x_out

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels, swish=1.0),
            nn.Identity(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels, swish=0.0 if use_scale_shift_norm else 1.0),
            nn.SiLU() if use_scale_shift_norm else nn.Identity(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            encoder_channels=None,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels, swish=0.0)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        if encoder_channels is not None:
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, encoder_out=None):
        b, c, *spatial = x.shape
        qkv = self.qkv(self.norm(x).view(b, c, -1))
        if encoder_out is not None:
            encoder_out = self.encoder_kv(encoder_out)
            h = self.attention(qkv, encoder_out)
        else:
            h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = th.cat([ek, k], dim=-1)
            v = th.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            encoder_channels=None,
            dataset=''
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.dataset = dataset
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes:
            layer_classes = self.num_classes.split(',')
            if self.dataset == 'clevr_pos': # 2
                self.label_emb = nn.Linear(int(layer_classes[0]), self.time_embed_dim)
                # unconditional label
                self.null_emb = nn.Embedding(1, self.time_embed_dim)
            elif self.dataset == 'clevr_rel': # 4,3,9,3,3,7
                attr_classes = [int(x) for x in layer_classes[:-1]]
                rel_class = int(layer_classes[-1])
                self.obj_emb = nn.ModuleList(
                    [nn.Embedding(num_classes, self.time_embed_dim // len(attr_classes))
                     for num_classes in attr_classes]
                )
                assert self.time_embed_dim // 3 == int(self.time_embed_dim / 3)
                self.fc = nn.Linear(self.time_embed_dim // len(attr_classes) * len(attr_classes),
                                    self.time_embed_dim // 3)
                self.rel_emb = nn.Embedding(rel_class, self.time_embed_dim // 3)
                self.null_emb = nn.Embedding(1, self.time_embed_dim)
            elif self.dataset == 'clevr': 
                # TODO add latent embedding
                pass
            else:
                raise NotImplementedError

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                encoder_channels=encoder_channels,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch, swish=1.0),
            nn.Identity(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )
        self.use_fp16 = use_fp16

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None, masks=None, layer_idx=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :param masks: an [N] Tensor of booleans
        :param layer_idx: integer indicates which attribute layer is used
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)) # embed timestep 

        if self.num_classes:
            # assert y.shape == (x.shape[0],)   # assert for discrete class labels
            # add label embedding, into timestep embedding
            if self.dataset == 'clevr_pos':
                label_emb = th.empty((y.shape[0], self.time_embed_dim), device=y.device)
                label_emb[masks] = self.label_emb(y[masks])
                # replace null labels with special embeddings
                label_emb[~masks] = self.null_emb.weight[0][None].repeat(label_emb[~masks].shape[0], 1)
                emb = emb + label_emb
            elif self.dataset == 'clevr': # TODO add latent embedding later
                pass
            elif self.dataset == 'clevr_rel':
                assert masks is not None, "masks are not provided for training relational clevr data."
                label_emb = th.empty((y.shape[0], emb.shape[1]), device=y.device)
                label_emb[~masks] = self.null_emb.weight[0][None].repeat(label_emb[~masks].shape[0], 1)
                # embed objects and relations
                obj_emb_1 = self.fc(th.cat([self.obj_emb[i](y[masks][:, i]) for i in range(5)], dim=1))
                obj_emb_2 = self.fc(th.cat([self.obj_emb[i - 5](y[masks][:, i]) for i in range(5, 10)], dim=1))
                rel_emb = self.rel_emb(y[masks][:, -1])
                label_emb[masks] = th.cat((obj_emb_1, obj_emb_2, rel_emb), dim=1)
                emb = emb + label_emb
            else:
                raise NotImplementedError

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


class DecompUNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            components=3, # TODO related to num_classes
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            encoder_channels=None,
            dataset=''
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.dataset = dataset
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4 # TODO what is model_channels? why * 4?
        self.time_embed_dim = time_embed_dim
        self.latent_dim = time_embed_dim # just set to be time_embed_dim, since adding them together
        self.components = components

        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes:
            layer_classes = self.num_classes.split(',')
            if self.dataset == 'clevr_pos': # 2; x and y coord embedded by linear layer of dim 2
                self.label_emb = nn.Linear(int(layer_classes[0]), self.time_embed_dim)
                # unconditional label
                self.null_emb = nn.Embedding(1, self.time_embed_dim)
            elif self.dataset == 'clevr_rel': # 4,3,9,3,3,7
                attr_classes = [int(x) for x in layer_classes[:-1]]
                rel_class = int(layer_classes[-1])
                self.obj_emb = nn.ModuleList(
                    [nn.Embedding(num_classes, self.time_embed_dim // len(attr_classes))
                     for num_classes in attr_classes]
                )
                assert self.time_embed_dim // 3 == int(self.time_embed_dim / 3)
                self.fc = nn.Linear(self.time_embed_dim // len(attr_classes) * len(attr_classes),
                                    self.time_embed_dim // 3)
                self.rel_emb = nn.Embedding(rel_class, self.time_embed_dim // 3)
                self.null_emb = nn.Embedding(1, self.time_embed_dim)
            elif self.dataset == 'clevr': 
                # TODO add latent embedding
                pass
            else:
                raise NotImplementedError

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                encoder_channels=encoder_channels,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch, swish=1.0),
            nn.Identity(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

        # TODO check what output means
        # goal: need to know number of desired latents n
        # input: tile image n times
        # output: mean of n predictions
        self.use_fp16 = use_fp16



        # TODO copy the encoder layers 
        filter_dim = 64 # TODO check this?
        latent_dim_expand = self.latent_dim * self.components # TODO params
        self.embed_conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)
        self.embed_layer1 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
        self.embed_layer2 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
        self.embed_layer3 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
        
        self.embed_fc1 = nn.Linear(filter_dim, filter_dim)
        self.embed_fc2 = nn.Linear(filter_dim, latent_dim_expand) # TODO dim issues; need to tile input to forward?

        # x = self.embed_conv1(im) 
        # x = F.relu(x)
        # x = self.embed_layer1(x) # CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
        # x = self.embed_layer2(x)
        # x = self.embed_layer3(x)


        # x = x.mean(dim=2).mean(dim=2)

        # x = x.view(x.size(0), -1)
        # output = self.embed_fc1(x)
        # x = F.relu(self.embed_fc1(x))
        # output = self.embed_fc2(x)


    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    # TODO add layers to model... right shapes...
    def embed_latent(self, im):
        """Generate latents from given image."""
        x = self.embed_conv1(im) 
        x = F.relu(x)
        x = self.embed_layer1(x) # CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
        x = self.embed_layer2(x)
        x = self.embed_layer3(x)


        x = x.mean(dim=2).mean(dim=2)

        x = x.view(x.size(0), -1)
        output = self.embed_fc1(x)
        x = F.relu(self.embed_fc1(x))
        output = self.embed_fc2(x)
        # import pdb; pdb.set_trace() # see what latent looks like
        return output

    def forward(self, x, timesteps, latent=None, latent_index=None, y=None, masks=None, layer_idx=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        # latents

        :param masks: an [N] Tensor of booleans
        :param layer_idx: integer indicates which attribute layer is used
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert latent is not None, "latent must be provided"
        # assert (y is not None) == (
        #         self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"

        # TODO tile x and emb, to have dim self.components that matches latent
        hs = []
        # sinusoidal embedding of timesteps
        # Parameters are shared across time, which is specified to the network using the Transformer sinusoidal position embedding
        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)) # embed timestep 
        if latent_index is None:
            s = latent.size()
            bs = s[0]
            latent = latent.view(s[0], self.components, -1)
            time_emb = time_emb[:, None, :].expand(-1, self.components, -1)
            x = x[:, None, :].expand(-1, self.components, -1, -1, -1)
            x = th.flatten(x, 0, 1)

            time_emb = th.flatten(time_emb, 0, 1)
            latent = th.flatten(latent, 0, 1)
            # import pdb
            # # pdb.set_trace()
            # print(time_emb.size()) # 48 x 768
            # print(latent.size())
        else:
            # take given slice
            latent = latent[:, self.latent_dim * latent_index: self.latent_dim * (latent_index + 1)]
        # time_emb = time_emb.tile(self.components)
        # if self.num_classes:
        #     # assert y.shape == (x.shape[0],)   # assert for discrete class labels
        #     # add label embedding, into timestep embedding
        #     if self.dataset == 'clevr_pos':
        #         label_emb = th.empty((y.shape[0], self.time_embed_dim), device=y.device)
        #         label_emb[masks] = self.label_emb(y[masks])
        #         # replace null labels with special embeddings
        #         label_emb[~masks] = self.null_emb.weight[0][None].repeat(label_emb[~masks].shape[0], 1)
        #         emb = emb + label_emb
        #     elif self.dataset == 'clevr': # TODO add latent embedding later
        #         pass
        #     elif self.dataset == 'clevr_rel':
        #         assert masks is not None, "masks are not provided for training relational clevr data."
        #         label_emb = th.empty((y.shape[0], emb.shape[1]), device=y.device)
        #         label_emb[~masks] = self.null_emb.weight[0][None].repeat(label_emb[~masks].shape[0], 1)
        #         # embed objects and relations
        #         obj_emb_1 = self.fc(th.cat([self.obj_emb[i](y[masks][:, i]) for i in range(5)], dim=1))
        #         obj_emb_2 = self.fc(th.cat([self.obj_emb[i - 5](y[masks][:, i]) for i in range(5, 10)], dim=1))
        #         rel_emb = self.rel_emb(y[masks][:, -1])
        #         label_emb[masks] = th.cat((obj_emb_1, obj_emb_2, rel_emb), dim=1)
        #         emb = emb + label_emb
        #     else:
        #         raise NotImplementedError
        # embed = time_emb + latent # vector of length latent dim x components 
        # latent embedding: we want latent encoder for this model. 
        
        # emb_chunked = th.chunk(embed, self.components)

        # embed = embed.reshape((-1, self.components)) # reshape to N x components x latent_dim
        # import pdb; pdb.set_trace()

        emb = time_emb + latent

        h = x.type(self.dtype)

        for module in self.input_blocks:
            # import pdb; pdb.set_trace()
            h = module(h, emb) # each module already takes in embedding from timestep
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        o = self.out(h)
        s = o.size()
        if latent_index is None:
            o = o.view(bs, -1, *s[1:]).mean(dim=1)
        # import pdb; pdb.set_trace()
        # print(o.size())
        # output += o
        return o


class SuperResUNetModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear", align_corners=False)
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class InpaintUNetModel(UNetModel):
    """
    A UNetModel which can perform inpainting.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2 + 1
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2 + 1
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, inpaint_image=None, inpaint_mask=None, **kwargs):
        if inpaint_image is None:
            inpaint_image = th.zeros_like(x)
        if inpaint_mask is None:
            inpaint_mask = th.zeros_like(x[:, :1])
        return super().forward(
            th.cat([x, inpaint_image * inpaint_mask, inpaint_mask], dim=1),
            timesteps,
            **kwargs,
        )


class SuperResInpaintUNetModel(UNetModel):
    """
    A UNetModel which can perform both upsampling and inpainting.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 3 + 1
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 3 + 1
        super().__init__(*args, **kwargs)

    def forward(
            self,
            x,
            timesteps,
            inpaint_image=None,
            inpaint_mask=None,
            low_res=None,
            **kwargs,
    ):
        if inpaint_image is None:
            inpaint_image = th.zeros_like(x)
        if inpaint_mask is None:
            inpaint_mask = th.zeros_like(x[:, :1])
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        return super().forward(
            th.cat([x, inpaint_image * inpaint_mask, inpaint_mask, upsampled], dim=1),
            timesteps,
            **kwargs,
        )
