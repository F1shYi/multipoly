import random
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet import UNetModel
from models.chord_encoder import ChordEncoder


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


class Diffusion(nn.Module):

    def __init__(
        self,
        unet_model: UNetModel,
        chord_encoder: ChordEncoder,
        n_steps: int,
        linear_start: float,
        linear_end: float,
    ):
        """
        :param unet_model: is the [U-Net](model/unet.html) that predicts noise
         $\epsilon_\text{cond}(x_t, c)$, in latent space
        :param chord_encoder: is the pretrained chord encoder.
        :param n_steps: is the number of diffusion steps $T$.
        :param linear_start: is the start of the $\beta$ schedule.
        :param linear_end: is the end of the $\beta$ schedule.
        """
        super().__init__()
        self.eps_model = unet_model
        self.chord_encoder = chord_encoder
        self.n_steps = n_steps
        beta = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_steps, dtype=torch.float64
            )
            ** 2
        )
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha = nn.Parameter(alpha.to(torch.float32), requires_grad=False)
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)
        self.sigma2 = self.beta

        for params in self.chord_encoder.parameters():
            params.requires_grad = False



    @property
    def device(self):
        """
        ### Get model device
        """
        return next(iter(self.eps_model.parameters())).device

    def _encoder_chord(self, chord):
        """
        :param chord: shape [B, 32, 36]
        :output cond: shape [B, 1, 512]
        """
        z = self.chord_encoder(chord).mean
        z = z.unsqueeze(1)
        return z

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor):
        return self.eps_model(x, t, context)

    def q_xt_x0(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        #### Get $q(x_t|x_0)$ distribution
        """

        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None
    ):
        """
        #### Sample from $q(x_t|x_0)$
        """

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if eps is None:
            eps = torch.randn_like(x0)

        # get $q(x_t|x_0)$
        mean, var = self.q_xt_x0(x0, t)
        # Sample from $q(x_t|x_0)$
        return mean + (var**0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        """
        #### Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
        """

        eps_theta = self.eps_model(xt, t)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = 1 / (alpha**0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.sigma2, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var**0.5) * eps

    def loss(
        self,
        x0: torch.Tensor,
        chord: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        uncond_prob: float = 0.2
    ):
        batch_size = x0.shape[0]
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps=noise)
        cond = self._encoder_chord(chord)

        if random.random() < uncond_prob:
            cond = -torch.ones_like(cond).to(cond.device)

        eps_theta = self.eps_model(xt, t, cond)

        return F.mse_loss(noise, eps_theta)


