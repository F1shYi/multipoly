from models.unet_modules import DownSample, ResBlock, UpSample, normalization

import math
from typing import List, Optional

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class DrumUNet(nn.Module):
    """
    Takes as inputs multitrack 2-channel pianorolls `m_t`, `m_{t-1}` and drum matrix `d_t` at time step `t`,
    and predicts the noise to be denoised from `d_t` to get `d_{t-1}`.
    """

    def __init__(
        self,
        drum_in_channels: int,
        drum_out_channels: int,
        multitrack_in_channels: int,
        channels: int,
        n_res_blocks: int,
        channel_multipliers: List[int],
    ):
        """
        Args:
            drum_in_channels: is the number of channels in the input drum pianoroll.
            drum_out_channels: is the number of channels in the output drum pianoroll.
            multitrack_in_channels: is the number of channels in the input multitrack pianoroll. Default = `track_num * in_channels_per_track`
            channels: is the base channel count for the model
            n_res_blocks: number of residual blocks at each level
            channel_multipliers: are the multiplicative factors for number of channels for each level
        """

        self.channels = channels
        levels = len(channel_multipliers)

        d_time_emb = channels * 4
        self.time_embeds = [
            nn.Sequential(
                nn.Linear(channels, d_time_emb),
                nn.SiLU(),
                nn.Linear(d_time_emb, d_time_emb),
            ),
            nn.Sequential(
                nn.Linear(channels, d_time_emb),
                nn.SiLU(),
                nn.Linear(d_time_emb, d_time_emb),
            ),
            nn.Sequential(
                nn.Linear(channels, d_time_emb),
                nn.SiLU(),
                nn.Linear(d_time_emb, d_time_emb),
            ),
        ]

        self.in_projs = [
            nn.Conv2d(drum_in_channels, channels, 3, padding=1),
            nn.Conv2d(multitrack_in_channels, channels, 3, padding=1),
            nn.Conv2d(multitrack_in_channels, channels, 3, padding=1),
        ]

        self.input_blocks = [nn.ModuleList(), nn.ModuleList(), nn.ModuleList()]

        input_block_channels = [channels]
        channels_list = [channels * m for m in channel_multipliers]
        for i in range(levels):
            for _ in range(n_res_blocks):
                layers = [
                    ResBlock(channels, d_time_emb, out_channels=channels_list[i]),
                    ResBlock(channels, d_time_emb, out_channels=channels_list[i]),
                    ResBlock(channels, d_time_emb, out_channels=channels_list[i]),
                ]
                channels = channels_list[i]
                for ii in range(3):
                    self.input_blocks[ii].append(layers[ii])

                input_block_channels.append(channels)
            if i != levels - 1:
                for ii in range(3):
                    self.input_blocks[ii].append(DownSample(channels))

                input_block_channels.append(channels)

        self.middle_blocks = [
            nn.Sequential(
                ResBlock(channels, d_time_emb),
                ResBlock(channels, d_time_emb),
            ),
            nn.Sequential(
                ResBlock(channels, d_time_emb),
                ResBlock(channels, d_time_emb),
            ),
            nn.Sequential(
                ResBlock(channels, d_time_emb),
                ResBlock(channels, d_time_emb),
            ),
        ]

        self.output_blocks = [nn.ModuleList([]), nn.ModuleList([]), nn.ModuleList([])]
        for i in reversed(range(levels)):
            for j in range(n_res_blocks + 1):
                layers = [
                    [
                        ResBlock(
                            channels + input_block_channels.pop(),
                            d_time_emb,
                            out_channels=channels_list[i],
                        )
                    ],
                    [
                        ResBlock(
                            channels + input_block_channels.pop(),
                            d_time_emb,
                            out_channels=channels_list[i],
                        )
                    ],
                    [
                        ResBlock(
                            channels + input_block_channels.pop(),
                            d_time_emb,
                            out_channels=channels_list[i],
                        )
                    ],
                ]
                channels = channels_list[i]
                if i != 0 and j == n_res_blocks:
                    for layer in layers:
                        layer.append(UpSample(channels))

                for ii in range(3):
                    self.output_blocks[ii].append(layers[ii])

        self.out_proj = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, drum_out_channels, 3, padding=1),
        )

    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        """
        ## Create sinusoidal time step embeddings

        :param time_steps: are the time steps of shape `[batch_size]`
        :param max_period: controls the minimum frequency of the embeddings.
        """
        half = self.channels // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=time_steps.device)
        args = time_steps[:, None].float() * frequencies[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(
        self,
        m_t: torch.Tensor,
        m_t_minus_1: torch.Tensor,
        d_t: torch.Tensor,
        t: torch.Tensor,
    ):
        """Forward pass of the U-Net.

        Args:
            m_t (torch.Tensor): Input no-drum noisy multitrack at timestep `t` of shape `[batch_size, track_num, channels, width, heitht]`.
            m_t_minus_1 (torch.Tensor): Input no-drum noisy multitrack at timestep `t-1` of shape `[batch_size, track_num, channels, width, heitht]`.
            d_t (torch.Tensor): Input noisy drum of shape `[batch_size, 1, width, height]`.
            t (torch.Tensor): Input timestep of shape `[batch_size, ]`

        Returns:
            torch.Tensor: Output tensor of shape `[batch_size, out_channels, width, height]`.
        """

        hs = [
            self.in_projs[0](d_t),
            self.in_projs[1](m_t),
            self.in_projs[2](m_t_minus_1),
        ]
        ts = [
            self.time_step_embedding(t),
            self.time_step_embedding(t),
            self.time_step_embedding(t - 1),
        ]
        ts = [
            self.time_embeds[0](ts[0]),
            self.time_embeds[1](ts[1]),
            self.time_embeds[2](ts[2]),
        ]

        len_0 = len(self.input_blocks[0])
        len_1 = len(self.input_blocks[1])
        len_2 = len(self.input_blocks[2])

        assert len_0 == len_1 and len_0 == len_2

        for i in range(len_0):
            for j in range(3):
                layer = self.input_blocks[j][i]
                if isinstance(layer, ResBlock):
                    hs[j] = layer(hs[j], ts[j])
                elif isinstance(layer, DownSample):
                    hs[j] = layer(hs[j])
                else:
                    raise ValueError("Invalid layer!")
