"""
3d UNet modified from stable diffusion.
"""

import math
from typing import List, Optional

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


# ----------------InterTrackModule---------------- #

class InterTrackAttention(nn.Module):
    def __init__(
        self,
        d_intertrack_encoder,
        n_intertrack_head,
        d_intertrack_ff,
        num_intertrack_encoder_layers,

        track_num=4
    ):
        super().__init__()
        encoder = nn.TransformerEncoderLayer(
            d_model=d_intertrack_encoder,
            nhead=n_intertrack_head,
            dim_feedforward=d_intertrack_ff,
            batch_first=True,
            norm_first=True,
        )
        self.attention = nn.TransformerEncoder(
            encoder, num_layers=num_intertrack_encoder_layers
        )
        self.track_embedding = nn.Parameter(torch.randn(
            (1, track_num, d_intertrack_encoder)), requires_grad=True)

    def forward(self, input_tensor):
        """
        Input: a tensor of shape (N, L, C)
        Output: a tensor of shape (N, L, C)
        """

        output_tensor = input_tensor + \
            self.attention(input_tensor + self.track_embedding)
        return output_tensor


class InterTrackConv3d(nn.Module):

    def __init__(self, channels, track_num=4, kernel_size=3):
        super().__init__()
        self.conv1 = self._get_conv3d_block(channels, track_num, kernel_size)
        self.conv2 = self._get_conv3d_block(channels, track_num, kernel_size=5)
        self.conv3 = self._get_conv3d_block(channels, track_num, kernel_size=7)

    def _get_conv3d_block(self, channels, track_num, kernel_size):
        return nn.Conv3d(channels, channels*track_num,
                         (track_num, kernel_size, kernel_size),
                         stride=1, padding=(0, (kernel_size-1)//2, (kernel_size-1)//2)
                         )

    def forward(self, input_tensor: torch.Tensor):
        '''
        Input: a tensor of shape (B, track_num, channels, width, height)
        Output: a tensor of same shape.
        '''
        b, t, c, w, h = input_tensor.shape
        # reshape to B, channels, track_num, width, height
        x = input_tensor.permute(0, 2, 1, 3, 4)
        x = F.tanh(self.conv1(x)).reshape(b, c, t, w, h)
        x = F.tanh(self.conv2(x)).reshape(b, c, t, w, h)
        x = F.tanh(self.conv3(x)).reshape(b, t, c, w, h)
        return input_tensor + x


# ----------------IntraTrackModule---------------- #


class SpatialTransformer(nn.Module):
    """
    ## Spatial Transformer
    """

    def __init__(self, channels: int, n_heads: int, n_layers: int, d_cond: int):
        """
        :param channels: is the number of channels in the feature map
        :param n_heads: is the number of attention heads
        :param n_layers: is the number of transformer layers
        :param d_cond: is the size of the conditional embedding
        """
        super().__init__()
        # Initial group normalization
        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=channels, eps=1e-6, affine=True
        )
        # Initial $1 \times 1$ convolution
        self.proj_in = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    channels, n_heads, channels // n_heads, d_cond=d_cond
                )
                for _ in range(n_layers)
            ]
        )

        # Final $1 \times 1$ convolution
        self.proj_out = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        :param x: is the feature map of shape `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings of shape `[batch_size,  n_cond, d_cond]`
        """
        # Get shape `[batch_size, channels, height, width]`
        b, c, h, w = x.shape
        # For residual connection
        x_in = x
        # Normalize
        x = self.norm(x)
        # Initial $1 \times 1$ convolution
        x = self.proj_in(x)
        # Transpose and reshape from `[batch_size, channels, height, width]`
        # to `[batch_size, height * width, channels]`
        x = x.permute(0, 2, 3, 1).view(b, h * w, c)
        # Apply the transformer layers
        for block in self.transformer_blocks:
            x = block(x, cond)
        # Reshape and transpose from `[batch_size, height * width, channels]`
        # to `[batch_size, channels, height, width]`
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        # Final $1 \times 1$ convolution
        x = self.proj_out(x)
        # Add residual
        return x + x_in


class BasicTransformerBlock(nn.Module):
    """
    ### Transformer Layer
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, d_cond: int):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        :param d_cond: is the size of the conditional embeddings
        """
        super().__init__()
        # Self-attention layer and pre-norm layer
        self.attn1 = CrossAttention(d_model, d_model, n_heads, d_head)
        self.norm1 = nn.LayerNorm(d_model)
        # Cross attention layer and pre-norm layer
        self.attn2 = CrossAttention(d_model, d_cond, n_heads, d_head)
        self.norm2 = nn.LayerNorm(d_model)
        # Feed-forward network and pre-norm layer
        self.ff = FeedForward(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        :param x: are the input embeddings of shape `[batch_size, height * width, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size,  n_cond, d_cond]`
        """
        # Self attention
        x = self.attn1(self.norm1(x)) + x
        # Cross-attention with conditioning
        x = self.attn2(self.norm2(x), cond=cond) + x
        # Feed-forward network
        x = self.ff(self.norm3(x)) + x

        return x


class CrossAttention(nn.Module):
    """
    ### Cross Attention Layer

    This falls-back to self-attention when conditional embeddings are not specified.
    """

    use_flash_attention: bool = False

    def __init__(
        self,
        d_model: int,
        d_cond: int,
        n_heads: int,
        d_head: int,
        is_inplace: bool = True,
    ):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        :param d_cond: is the size of the conditional embeddings
        :param is_inplace: specifies whether to perform the attention softmax computation inplace to
            save memory
        """
        super().__init__()

        self.is_inplace = is_inplace
        self.n_heads = n_heads
        self.d_head = d_head

        # Attention scaling factor
        self.scale = d_head**-0.5

        # Query, key and value mappings
        d_attn = d_head * n_heads
        self.to_q = nn.Linear(d_model, d_attn, bias=False)
        self.to_k = nn.Linear(d_cond, d_attn, bias=False)
        self.to_v = nn.Linear(d_cond, d_attn, bias=False)

        # Final linear layer
        self.to_out = nn.Sequential(nn.Linear(d_attn, d_model))

        # Setup [flash attention](https://github.com/HazyResearch/flash-attention).
        # Flash attention is only used if it's installed
        # and `CrossAttention.use_flash_attention` is set to `True`.
        try:
            # You can install flash attention by cloning their Github repo,
            # [https://github.com/HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention)
            # and then running `python setup.py install`
            from flash_attn.flash_attention import FlashAttention

            self.flash = FlashAttention()
            # Set the scale for scaled dot-product attention.
            self.flash.softmax_scale = self.scale
        # Set to `None` if it's not installed
        except ImportError:
            self.flash = None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """
        :param x: are the input embeddings of shape `[batch_size, height * width, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size, n_cond, d_cond]`
        """

        # If `cond` is `None` we perform self attention
        has_cond = cond is not None
        if not has_cond:
            cond = x

        # Get query, key and value vectors
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)

        # Use flash attention if it's available and the head size is less than or equal to `128`
        if (
            CrossAttention.use_flash_attention
            and self.flash is not None
            and not has_cond
            and self.d_head <= 128
        ):
            return self.flash_attention(q, k, v)
        # Otherwise, fallback to normal attention
        else:
            return self.normal_attention(q, k, v)

    def flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Flash Attention

        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """

        # Get batch size and number of elements along sequence axis (`width * height`)
        batch_size, seq_len, _ = q.shape

        # Stack `q`, `k`, `v` vectors for flash attention, to get a single tensor of
        # shape `[batch_size, seq_len, 3, n_heads * d_head]`
        qkv = torch.stack((q, k, v), dim=2)
        # Split the heads
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_head)

        # Flash attention works for head sizes `32`, `64` and `128`, so we have to pad the heads to
        # fit this size.
        if self.d_head <= 32:
            pad = 32 - self.d_head
        elif self.d_head <= 64:
            pad = 64 - self.d_head
        elif self.d_head <= 128:
            pad = 128 - self.d_head
        else:
            raise ValueError(
                f"Head size ${self.d_head} too large for Flash Attention")

        # Pad the heads
        if pad:
            qkv = torch.cat(
                (qkv, qkv.new_zeros(batch_size, seq_len, 3, self.n_heads, pad)), dim=-1
            )

        # Compute attention
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        # This gives a tensor of shape `[batch_size, seq_len, n_heads, d_padded]`
        out, _ = self.flash(qkv)
        # Truncate the extra head size
        out = out[:, :, :, : self.d_head]
        # Reshape to `[batch_size, seq_len, n_heads * d_head]`
        out = out.reshape(batch_size, seq_len, self.n_heads * self.d_head)

        # Map to `[batch_size, height * width, d_model]` with a linear layer
        return self.to_out(out)

    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Normal Attention

        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """

        # Split them to heads of shape `[batch_size, seq_len, n_heads, d_head]`
        q = q.view(*q.shape[:2], self.n_heads, -1)
        k = k.view(*k.shape[:2], self.n_heads, -1)
        v = v.view(*v.shape[:2], self.n_heads, -1)

        # Calculate attention $\frac{Q K^\top}{\sqrt{d_{key}}}$
        attn = torch.einsum("bihd,bjhd->bhij", q, k) * self.scale

        # Compute softmax
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)$$
        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[half:] = attn[half:].softmax(dim=-1)
            attn[:half] = attn[:half].softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        # Compute attention output
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        out = torch.einsum("bhij,bjhd->bihd", attn, v)
        # Reshape to `[batch_size, height * width, n_heads * d_head]`
        out = out.reshape(*out.shape[:2], -1)
        # Map to `[batch_size, height * width, d_model]` with a linear layer
        return self.to_out(out)


class FeedForward(nn.Module):
    """
    ### Feed-Forward Network
    """

    def __init__(self, d_model: int, d_mult: int = 4):
        """
        :param d_model: is the input embedding size
        :param d_mult: is multiplicative factor for the hidden layer size
        """
        super().__init__()
        self.net = nn.Sequential(
            GeGLU(d_model, d_model * d_mult),
            nn.Dropout(0.0),
            nn.Linear(d_model * d_mult, d_model),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class GeGLU(nn.Module):
    """
    ### GeGLU Activation

    $$\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$$
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # Combined linear projections $xW + b$ and $xV + c$
        self.proj = nn.Linear(d_in, d_out * 2)

    def forward(self, x: torch.Tensor):
        # Get $xW + b$ and $xV + c$
        x, gate = self.proj(x).chunk(2, dim=-1)
        # $\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$
        return x * F.gelu(gate)


# -----------------UNet------------------ #


class UNetModel(nn.Module):
    """
    ## U-Net model
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        channels: int,
        n_res_blocks: int,
        attention_levels: List[int],
        channel_multipliers: List[int],
        n_heads: int,
        n_intertrack_head: int,
        num_intertrack_encoder_layers: int,
        intertrack_attention_levels: List[int],
        tf_layers: int = 1,
        d_cond: int = 768,
        use_conv3d=False,
    ):
        """
        :param in_channels: is the number of channels in the input feature map
        :param out_channels: is the number of channels in the output feature map
        :param channels: is the base channel count for the model
        :param n_res_blocks: number of residual blocks at each level
        :param attention_levels: are the levels at which attention should be performed
        :param channel_multipliers: are the multiplicative factors for number of channels for each level
        :param n_heads: the number of attention heads in the transformers
        """
        super().__init__()
        self.channels = channels

        # Number of levels
        levels = len(channel_multipliers)
        # Size time embeddings
        d_time_emb = channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(channels, d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb),
        )

        # Input half of the U-Net
        self.input_blocks = nn.ModuleList()
        # Initial $3 \times 3$ convolution that maps the input to `channels`.
        # The blocks are wrapped in `TimestepEmbedSequential` module because
        # different modules have different forward function signatures;
        # for example, convolution only accepts the feature map and
        # residual blocks accept the feature map and time embedding.
        # `TimestepEmbedSequential` calls them accordingly.
        self.input_blocks.append(
            TimestepEmbedSequential(
                nn.Conv2d(in_channels, channels, 3, padding=1))
        )
        # Number of channels at each block in the input half of U-Net
        input_block_channels = [channels]
        # Number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]
        # Prepare levels
        for i in range(levels):
            # Add the residual blocks and attentions
            for _ in range(n_res_blocks):
                # Residual block maps from previous number of channels to the number of
                # channels in the current level
                layers = [ResBlock(channels, d_time_emb,
                                   out_channels=channels_list[i])]
                channels = channels_list[i]
                # Add transformer
                if i in attention_levels:
                    layers.append(
                        SpatialTransformer(
                            channels, n_heads, tf_layers, d_cond)
                    )

                # Add intertrack attention

                if i in intertrack_attention_levels:
                    layers.append(
                        InterTrackAttention(
                            d_intertrack_encoder=channels,
                            n_intertrack_head=n_intertrack_head,
                            d_intertrack_ff=2 * channels,
                            num_intertrack_encoder_layers=num_intertrack_encoder_layers,
                        )
                    )
                if use_conv3d:
                    layers.append(InterTrackConv3d(
                        channels
                    ))

                # Add them to the input half of the U-Net and keep track of the number of channels of
                # its output
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(channels)
            # Down sample at all levels except last
            if i != levels - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(DownSample(channels)))
                input_block_channels.append(channels)

        # The middle of the U-Net
        self.middle_block = TimestepEmbedSequential(
            ResBlock(channels, d_time_emb),
            SpatialTransformer(channels, n_heads, tf_layers, d_cond),
            ResBlock(channels, d_time_emb),
        )

        # Second half of the U-Net
        self.output_blocks = nn.ModuleList([])
        # Prepare levels in reverse order
        for i in reversed(range(levels)):
            # Add the residual blocks and attentions
            for j in range(n_res_blocks + 1):
                # Residual block maps from previous number of channels plus the
                # skip connections from the input half of U-Net to the number of
                # channels in the current level.
                layers = [
                    ResBlock(
                        channels + input_block_channels.pop(),
                        d_time_emb,
                        out_channels=channels_list[i],
                    )
                ]
                channels = channels_list[i]
                # Add transformer
                if i in attention_levels:
                    layers.append(
                        SpatialTransformer(
                            channels, n_heads, tf_layers, d_cond)
                    )

                # Up-sample at every level after last residual block
                # except the last one.
                # Note that we are iterating in reverse; i.e. `i == 0` is the last.
                if i != 0 and j == n_res_blocks:
                    layers.append(UpSample(channels))

                if i in intertrack_attention_levels:
                    layers.append(
                        InterTrackAttention(
                            d_intertrack_encoder=channels,
                            n_intertrack_head=n_intertrack_head,
                            d_intertrack_ff=2 * channels,
                            num_intertrack_encoder_layers=num_intertrack_encoder_layers,
                        )
                    )
                if use_conv3d:
                    layers.append(InterTrackConv3d(channels))
                # Add to the output half of the U-Net
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # Final normalization and $3 \times 3$ convolution
        self.out = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

        self.polyffusion_weights_keys = None
        self.other_weights_keys = None

    def load_polyffusion_checkpoints(self, polyffusion_checkpoints: dict, freeze_polyffusion=True, zero=True):
        print("---------------loading polyffusion weights-------------------")

        if zero:
            # first zero out all parameters
            for params in self.parameters():
                params.data.fill_(0.0)
            print("params zeroed.")
        # then load polyffusion checkpoints
        unet_from_polyffusion_state_dict = {
            k.removeprefix("ldm.eps_model."): v
            for k, v in polyffusion_checkpoints.items()
            if k.removeprefix("ldm.eps_model.") in self.state_dict()
        }
        missing_keys, unexpected_keys = self.load_state_dict(
            unet_from_polyffusion_state_dict, strict=False
        )

        self.other_weights_keys = missing_keys
        self.polyffusion_weights_keys = [
            key for key in unet_from_polyffusion_state_dict.keys()]

        for key in missing_keys:
            print(key)
        print(
            "---------------polyffusion weights loaded with the above missing keys-------------------"
        )
        if freeze_polyffusion:
            self._freeze_polyffusion()
            print("Polyffusion weights are freezed.")
        print("It is expected that missing keys are all intertrack modules")

    def _freeze_polyffusion(self):
        for name, param in self.named_parameters():
            if name in self.polyffusion_weights_keys:
                param.requires_grad = False

    def _unfreeze_polyffusion(self):
        for name, param in self.named_parameters():
            if name in self.polyffusion_weights_keys:
                param.requires_grad = True

    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        """
        ## Create sinusoidal time step embeddings

        :param time_steps: are the time steps of shape `[batch_size]`
        :param max_period: controls the minimum frequency of the embeddings.
        """
        # $\frac{c}{2}$; half the channels are sin and the other half is cos,
        half = self.channels // 2
        # $\frac{1}{10000^{\frac{2i}{c}}}$
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=time_steps.device)
        # $\frac{t}{10000^{\frac{2i}{c}}}$
        args = time_steps[:, None].float() * frequencies[None]
        # $\cos\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$ and $\sin\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, cond: torch.Tensor):
        """
        :param x: is the input feature map of shape `[batch_size, track_num, channels, width, height]`
        :param time_steps: are the time steps of shape `[batch_size*track_num]`
        :param cond: conditioning of shape `[batch_size*track_num, n_cond, d_cond]`
        """
        # To store the input half outputs for skip connections
        x_input_block = []

        # Get time step embeddings
        t_emb = self.time_step_embedding(time_steps)
        t_emb = self.time_embed(t_emb)

        # Input half of the U-Net
        for module in self.input_blocks:
            x = module(x, t_emb, cond)
            x_input_block.append(x)
        # Middle of the U-Net
        x = self.middle_block(x, t_emb, cond)
        # Output half of the U-Net
        for module in self.output_blocks:

            x = th.cat([x, x_input_block.pop()], dim=2)

            x = module(x, t_emb, cond)
        b, t = x.shape[0], x.shape[1]
        x = rearrange(x, "b t c w h -> (b t) c w h")
        x = self.out(x)
        x = rearrange(x, "(b t) c w h -> b t c w h", b=b, t=t)
        # Final normalization and $3 \times 3$ convolution
        return x


class TimestepEmbedSequential(nn.Sequential):
    """
    ### Sequential block for modules with different inputs

    This sequential module can compose of different modules suck as `ResBlock`,
    `nn.Conv` and `SpatialTransformer` and calls them with the matching signatures
    """

    def forward(self, x, t_emb, cond=None):
        for layer in self:

            if isinstance(layer, InterTrackAttention):
                b, t, c, w, h = x.shape

                x = rearrange(x, "b t c w h -> (b w h) t c")
                x = layer(x)
                x = rearrange(x, "(b w h) t c -> b t c w h", b=b, w=w, h=h)

            elif isinstance(layer, InterTrackConv3d):

                x = layer(x)

            else:

                b, t = x.shape[0], x.shape[1]
                x = rearrange(x, "b t c w h -> (b t) c w h")
                if isinstance(layer, ResBlock):
                    x = layer(x, t_emb)

                elif isinstance(layer, SpatialTransformer):
                    x = layer(x, cond)

                else:
                    x = layer(x)
                x = rearrange(x, "(b t) c w h -> b t c w h", b=b, t=t)

        return x


class UpSample(nn.Module):
    """
    ### Up-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution mapping
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Up-sample by a factor of $2$
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Apply convolution
        return self.conv(x)


class DownSample(nn.Module):
    """
    ## Down-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution with stride length of $2$ to down-sample by a factor of $2$
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Apply convolution
        return self.op(x)


class ResBlock(nn.Module):
    """
    ## ResNet Block
    """

    def __init__(self, channels: int, d_t_emb: int, *, out_channels=None):
        """
        :param channels: the number of input channels
        :param d_t_emb: the size of timestep embeddings
        :param out_channels: is the number of out channels. defaults to `channels.
        """
        super().__init__()
        # `out_channels` not specified
        if out_channels is None:
            out_channels = channels

        # First normalization and convolution
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

        # Time step embeddings
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_t_emb, out_channels),
        )
        # Final convolution layer
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        # `channels` to `out_channels` mapping layer for residual connection
        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        :param t_emb: is the time step embeddings of shape `[batch_size, d_t_emb]`
        """
        # Initial convolution

        h = self.in_layers(x)
        # Time step embeddings
        t_emb = self.emb_layers(t_emb).type(h.dtype)
        # Add time step embeddings
        h = h + t_emb[:, :, None, None]
        # Final convolution
        h = self.out_layers(h)
        # Add skip connection
        return self.skip_connection(x) + h


class GroupNorm32(nn.GroupNorm):
    """
    ### Group normalization with float32 casting
    """

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    ### Group normalization

    This is a helper function, with fixed number of groups..
    """
    return GroupNorm32(32, channels)
