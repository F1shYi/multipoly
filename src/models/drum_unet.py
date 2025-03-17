from models.unet_modules import (
    DownSample,
    ResBlock,
    UpSample,
    normalization,
    DrumSequential,
)

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

    def _get_time_embed(self, channels, d_time_emb) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(channels, d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb),
        )

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
        super().__init__()
        self.channels = channels

        levels = len(channel_multipliers)
        d_time_emb = channels * 4
        self.d_t_time_embed = self._get_time_embed(channels, d_time_emb)
        self.m_t_time_embed = self._get_time_embed(channels, d_time_emb)
        self.m_t_minus_1_time_embed = self._get_time_embed(channels, d_time_emb)

        self.d_t_input_blocks = nn.ModuleList()
        self.m_t_input_blocks = nn.ModuleList()
        self.m_t_minus_1_input_blocks = nn.ModuleList()

        self.d_t_input_blocks.append(
            DrumSequential(
                nn.Conv2d(drum_in_channels, channels, kernel_size=3, padding=1)
            )
        )
        self.m_t_input_blocks.append(
            DrumSequential(
                nn.Conv2d(multitrack_in_channels, channels, kernel_size=3, padding=1)
            )
        )
        self.m_t_minus_1_input_blocks.append(
            DrumSequential(
                nn.Conv2d(multitrack_in_channels, channels, kernel_size=3, padding=1)
            )
        )

        input_block_channels = [channels]
        channels_list = [channels * m for m in channel_multipliers]
        for i in range(levels):
            for _ in range(n_res_blocks):

                d_t_layers = [
                    ResBlock(channels, d_time_emb, out_channels=channels_list[i])
                ]
                m_t_layers = [
                    ResBlock(channels, d_time_emb, out_channels=channels_list[i])
                ]
                m_t_minus_1_layers = [
                    ResBlock(channels, d_time_emb, out_channels=channels_list[i])
                ]
                channels = channels_list[i]
                self.d_t_input_blocks.append(DrumSequential(*d_t_layers))
                self.m_t_input_blocks.append(DrumSequential(*m_t_layers))
                self.m_t_minus_1_input_blocks.append(
                    DrumSequential(*m_t_minus_1_layers)
                )
                input_block_channels.append(channels)
            if i != levels - 1:
                self.d_t_input_blocks.append(DrumSequential(DownSample(channels)))
                self.m_t_input_blocks.append(DrumSequential(DownSample(channels)))
                self.m_t_minus_1_input_blocks.append(
                    DrumSequential(DownSample(channels))
                )
                input_block_channels.append(channels)

        self.d_t_middle_block = DrumSequential(
            ResBlock(channels, d_time_emb),
            ResBlock(channels, d_time_emb),
        )
        self.m_t_middle_block = DrumSequential(
            ResBlock(channels, d_time_emb),
            ResBlock(channels, d_time_emb),
        )
        self.m_t_minus_1_middle_block = DrumSequential(
            ResBlock(channels, d_time_emb),
            ResBlock(channels, d_time_emb),
        )

        self.d_t_output_blocks = nn.ModuleList([])
        self.m_t_output_blocks = nn.ModuleList([])
        self.m_t_minus_1_output_blocks = nn.ModuleList([])

        for i in reversed(range(levels)):
            for j in range(n_res_blocks + 1):
                pop_channel = input_block_channels.pop()
                d_t_layers = [
                    ResBlock(
                        channels + pop_channel,
                        d_time_emb,
                        out_channels=channels_list[i],
                    )
                ]
                m_t_layers = [
                    ResBlock(
                        channels + pop_channel,
                        d_time_emb,
                        out_channels=channels_list[i],
                    )
                ]
                m_t_minus_1_layers = [
                    ResBlock(
                        channels + pop_channel,
                        d_time_emb,
                        out_channels=channels_list[i],
                    )
                ]
                channels = channels_list[i]

                if i != 0 and j == n_res_blocks:
                    d_t_layers.append(UpSample(channels))
                    m_t_layers.append(UpSample(channels))
                    m_t_minus_1_layers.append(UpSample(channels))

                self.d_t_output_blocks.append(DrumSequential(*d_t_layers))
                self.m_t_output_blocks.append(DrumSequential(*m_t_layers))
                self.m_t_minus_1_output_blocks.append(
                    DrumSequential(*m_t_minus_1_layers)
                )

        self.drum_out_proj = nn.Sequential(
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

    def _mask(self, d_t, m_t, m_t_minus_1):
        return d_t + F.tanh(m_t) * d_t + F.tanh(m_t_minus_1) * d_t

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

        batch_size, track_num, multitrack_channels, width, height = m_t.shape

        m_t = m_t.reshape(batch_size, track_num * multitrack_channels, width, height)
        m_t_minus_1 = m_t_minus_1.reshape(
            batch_size, track_num * multitrack_channels, width, height
        )

        d_t_input_block = []
        m_t_input_block = []
        m_t_minus_1_input_block = []

        d_t_t_emb = self.time_step_embedding(t)
        d_t_t_emb = self.d_t_time_embed(d_t_t_emb)
        m_t_t_emb = self.time_step_embedding(t)
        m_t_t_emb = self.m_t_time_embed(m_t_t_emb)
        m_t_minus_1_t_emb = self.time_step_embedding(t - 1)
        m_t_minus_1_t_emb = self.m_t_minus_1_time_embed(m_t_minus_1_t_emb)

        for d_t_module, m_t_module, m_t_minus_1_module in zip(
            self.d_t_input_blocks, self.m_t_input_blocks, self.m_t_minus_1_input_blocks
        ):
            d_t = d_t_module(d_t, d_t_t_emb)
            d_t_input_block.append(d_t)
            m_t = m_t_module(m_t, m_t_t_emb)
            m_t_input_block.append(m_t)
            m_t_minus_1 = m_t_minus_1_module(m_t_minus_1, m_t_minus_1_t_emb)
            m_t_minus_1_input_block.append(m_t_minus_1)
            d_t = self._mask(d_t, m_t, m_t_minus_1)

        d_t = self.d_t_middle_block(d_t, d_t_t_emb)
        m_t = self.m_t_middle_block(m_t, m_t_t_emb)
        m_t_minus_1 = self.m_t_minus_1_middle_block(m_t_minus_1, m_t_minus_1_t_emb)
        d_t = self._mask(d_t, m_t, m_t_minus_1)

        for d_t_module, m_t_module, m_t_minus_1_module in zip(
            self.d_t_output_blocks,
            self.m_t_output_blocks,
            self.m_t_minus_1_output_blocks,
        ):

            d_t = th.cat([d_t, d_t_input_block.pop()], dim=1)
            d_t = d_t_module(d_t, d_t_t_emb)
            m_t = th.cat([m_t, m_t_input_block.pop()], dim=1)
            m_t = m_t_module(m_t, m_t_t_emb)
            m_t_minus_1 = th.cat([m_t_minus_1, m_t_minus_1_input_block.pop()], dim=1)
            m_t_minus_1 = m_t_minus_1_module(m_t_minus_1, m_t_minus_1_t_emb)
            d_t = self._mask(d_t, m_t, m_t_minus_1)

        return self.drum_out_proj(d_t)
