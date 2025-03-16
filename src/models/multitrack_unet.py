import math
from typing import List, Optional

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.unet_modules import (
    DownSample,
    InterTrackAttention,
    ResBlock,
    SpatialTransformer,
    UpSample,
    normalization,
    MultitrackSequential,
)


class MultiTrackUNet(nn.Module):
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
        # The blocks are wrapped in `MultitrackSequential` module because
        # different modules have different forward function signatures;
        # for example, convolution only accepts the feature map and
        # residual blocks accept the feature map and time embedding.
        # `MultitrackSequential` calls them accordingly.
        self.input_blocks.append(
            MultitrackSequential(nn.Conv2d(in_channels, channels, 3, padding=1))
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
                layers = [ResBlock(channels, d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]
                # Add transformer
                if i in attention_levels:
                    layers.append(
                        SpatialTransformer(channels, n_heads, tf_layers, d_cond)
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

                # Add them to the input half of the U-Net and keep track of the number of channels of
                # its output
                self.input_blocks.append(MultitrackSequential(*layers))
                input_block_channels.append(channels)
            # Down sample at all levels except last
            if i != levels - 1:
                self.input_blocks.append(MultitrackSequential(DownSample(channels)))
                input_block_channels.append(channels)

        # The middle of the U-Net
        self.middle_block = MultitrackSequential(
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
                        SpatialTransformer(channels, n_heads, tf_layers, d_cond)
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
                # Add to the output half of the U-Net
                self.output_blocks.append(MultitrackSequential(*layers))

        # Final normalization and $3 \times 3$ convolution
        self.out = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

        self.polyffusion_weights_keys = None
        self.other_weights_keys = None

    def load_polyffusion_checkpoints(
        self, polyffusion_checkpoints: dict, freeze_polyffusion=True, zero=True
    ):
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
            key for key in unet_from_polyffusion_state_dict.keys()
        ]

        for key in missing_keys:
            print(key)
        print(
            "---------------polyffusion weights loaded with the above missing keys-------------------"
        )
        if freeze_polyffusion:
            self._freeze_polyffusion()
            print("Polyffusion weights are freezed.")
        print("It is expected that missing keys are all intertrack attention modules")

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
