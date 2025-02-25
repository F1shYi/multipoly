import torch
import torch.nn as nn
from einops import rearrange


class InterTrackAttention(nn.Module):
    def __init__(
        self,
        d_intertrack_encoder,
        n_intertrack_head,
        d_intertrack_ff,
        num_intertrack_encoder_layers,
    ):
        super().__init__()
        encoder = nn.TransformerEncoderLayer(
            d_model=d_intertrack_encoder,
            nhead=n_intertrack_head,
            dim_feedforward=d_intertrack_ff,
            batch_first=True,
        )
        self.attention = nn.TransformerEncoder(
            encoder, num_layers=num_intertrack_encoder_layers
        )

    def forward(self, input_tensor):
        """
        Input: a tensor of shape (batch_size, track_num, channels, width, height)
        Output: a tensor of shape (batch_size, track_num, channels, width, height)
        """

        # Rearrange the input tensor to (N, L, C)
        # where N = batch_size * width * height stands for the new batch size for the sequence
        # L = track_num stands for the sequence length
        # C = channels stands for the number of channels
        b, t, c, w, h = input_tensor.shape

        input_tensor = rearrange(input_tensor, "b t c w h -> (b w h) t c")

        # Apply the intertrack attention
        output_tensor = input_tensor + self.attention(input_tensor)

        # Rearrange the output tensor back to the original shape
        output_tensor = rearrange(
            output_tensor, "(b w h) t c -> b t c w h", b=b, w=w, h=h
        )

        return output_tensor
