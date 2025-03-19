import random
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet import UNetModel
from models.chord_encoder import ChordEncoder


def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1, 1)


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
        self.time_steps = np.asarray(list(range(self.n_steps)), dtype=np.int32)

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
        self.alpha_bar = nn.Parameter(
            alpha_bar.to(torch.float32), requires_grad=False)
        self.sigma2 = self.beta

        alpha_bar = self.alpha_bar
        alpha_bar_prev = torch.cat(
            [alpha_bar.new_tensor([1.0]), alpha_bar[:-1]])
        self.sqrt_alpha_bar = alpha_bar**0.5
        self.sqrt_1m_alpha_bar = (1.0 - alpha_bar) ** 0.5
        self.sqrt_recip_alpha_bar = alpha_bar**-0.5
        self.sqrt_recip_m1_alpha_bar = (1 / alpha_bar - 1) ** 0.5
        variance = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        self.log_var = torch.log(torch.clamp(variance, min=1e-20))
        self.mean_x0_coef = beta * (alpha_bar_prev**0.5) / (1.0 - alpha_bar)
        self.mean_xt_coef = (
            (1.0 - alpha_bar_prev) * ((1 - beta) ** 0.5) / (1.0 - alpha_bar)
        )

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

    def get_eps(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        *,
        uncond_scale: float,
        uncond_cond: Optional[torch.Tensor],
    ):
        if uncond_cond is None or uncond_scale == 1.0:
            return self.eps_model(x, t, c)
        elif uncond_scale == 0.0:  # unconditional
            return self.eps_model(x, t, uncond_cond)

        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        c_in = torch.cat([uncond_cond, c])

        e_t_uncond, e_t_cond = self.eps_model(x_in, t_in, c_in).chunk(2)
        e_t = e_t_uncond + uncond_scale * (e_t_cond - e_t_uncond)
        return e_t

    def q_xt_x0(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None
    ):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var**0.5) * eps

    def p_sample(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        step: int,
        repeat_noise: bool = False,
        temperature: float = 1.0,
        uncond_scale: float = 1.0,
        uncond_cond: Optional[torch.Tensor] = None,
    ):

        e_t = self.get_eps(
            x, t, c, uncond_scale=uncond_scale, uncond_cond=uncond_cond
        )

        bs, track_num = x.shape[0], x.shape[1]

        sqrt_recip_alpha_bar = x.new_full(
            (bs, track_num, 1, 1, 1), self.sqrt_recip_alpha_bar[step]
        )
        sqrt_recip_m1_alpha_bar = x.new_full(
            (bs, track_num, 1, 1, 1), self.sqrt_recip_m1_alpha_bar[step]
        )

        x0 = sqrt_recip_alpha_bar * x - sqrt_recip_m1_alpha_bar * e_t

        mean_x0_coef = x.new_full(
            (bs, track_num, 1, 1, 1), self.mean_x0_coef[step])
        mean_xt_coef = x.new_full(
            (bs, track_num, 1, 1, 1), self.mean_xt_coef[step])
        mean = mean_x0_coef * x0 + mean_xt_coef * x
        log_var = x.new_full((bs, track_num, 1, 1, 1), self.log_var[step])
        if step == 0:
            noise = 0
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]), device=x.device)
        else:
            noise = torch.randn(x.shape, device=x.device)
        noise = noise * temperature
        x_prev = mean + (0.5 * log_var).exp() * noise
        return x_prev, x0, e_t

    def loss(
        self,
        x0: torch.Tensor,
        chord: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ):
        batch_size, track_num = x0.shape[0], x0.shape[1]

        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps=noise)
        cond = self._encoder_chord(chord)

        if random.random() < 0.6:
            cond = (-torch.ones_like(cond)).to(self.device)

        t = torch.cat([t.reshape(-1, 1)]*track_num, dim=1).reshape(-1,)
        cond = torch.cat([cond]*track_num, dim=1).reshape(-1, 1, 512)

        eps_theta = self.eps_model(xt, t, cond)

        return F.mse_loss(noise, eps_theta)

    @torch.no_grad()
    def sample(
        self,
        shape: List[int],
        chords: Optional[torch.Tensor] = None,
        repeat_noise: bool = False,
        temperature: float = 1.0,
        x_last: Optional[torch.Tensor] = None,
        uncond_scale: float = 1.0,
        uncond_cond: Optional[torch.Tensor] = None,
        t_start: int = 0,
    ):
        """
        ### Sampling Loop

        :param shape: is the shape of the generated images in the
            form `[batch_size, track_num, channels, height, width]`
        :param chord: is the conditional chord with shape [batch_size, 32, 36]
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param x_last: is $x_T$. If not provided random noise will be used.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """

        bs, track_num = shape[0], shape[1]

        if uncond_cond is not None:
            uncond_cond = torch.cat(
                [uncond_cond]*track_num, dim=1).reshape(-1, 1, 512)

        cond = None
        if uncond_scale > 0.0:
            single_track_cond = self._encoder_chord(chords)
            cond = torch.cat([single_track_cond]*track_num,
                             dim=1).reshape(-1, 1, 512)
        else:
            cond = -torch.ones_like(uncond_cond)

        x = x_last if x_last is not None else torch.randn(
            shape, device=self.device)

        time_steps = np.flip(self.time_steps)[t_start:]

        from tqdm import tqdm
        for step in tqdm(time_steps):
            ts = x.new_full((bs*track_num,), step, dtype=torch.long)
            x, pred_x0, e_t = self.p_sample(
                x,
                cond,
                ts,
                step,
                repeat_noise=repeat_noise,
                temperature=temperature,
                uncond_scale=uncond_scale,
                uncond_cond=uncond_cond,
            )

        return x

    @torch.no_grad()
    def paint(
        self,
        shape: List[int],
        chords: Optional[torch.Tensor] = None,

        *,
        orig: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        uncond_scale: float = 1.0,
        uncond_cond: Optional[torch.Tensor] = None,
        repaint_n=1,
        t_start: int = 0,
    ):

        bs, track_num = shape[0], shape[1]

        cond = None
        if uncond_scale > 0.0:
            single_track_cond = self._encoder_chord(chords)
            cond = torch.cat([single_track_cond]*track_num,
                             dim=1).reshape(-1, 1, 512)
        if uncond_cond is not None:
            uncond_cond = torch.cat(
                [uncond_cond]*track_num, dim=1).reshape(-1, 1, 512)

        x = torch.randn(shape, device=self.device)

        time_steps = np.flip(self.time_steps)[t_start:]

        from tqdm import tqdm
        for step in tqdm(time_steps):
            x_t = x
            for u in range(repaint_n):
                noise = (
                    torch.randn_like(orig, device=orig.device)
                    if step > 0
                    else torch.zeros_like(orig, device=orig.device)
                )
                x_kn_tm1 = self.q_sample(orig, torch.tensor(
                    step, device=self.device, dtype=torch.long), eps=noise)
                ts = x_t.new_full((bs*track_num,), step, dtype=torch.long)

                x_unkn_tm1, _, _ = self.p_sample(
                    x_t,
                    cond,
                    ts,
                    step,
                    uncond_scale=uncond_scale,
                    uncond_cond=uncond_cond,
                )

                x = x_kn_tm1 * mask + x_unkn_tm1 * (1 - mask)
                if u < repaint_n - 1 and step > 0:
                    noise = torch.randn_like(orig, device=orig.device)
                    x_t = (
                        1 - self.beta[step - 1]
                    ) ** 0.5 * x + self.beta[step - 1] * noise

        return x
