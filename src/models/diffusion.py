import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import UNetModel


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


class LatentDiffusion(nn.Module):
    """
    ## Latent diffusion model

    This contains following components:

    * [AutoEncoder](model/autoencoder.html)
    * [U-Net](model/unet.html) with [attention](model/unet_attention.html)
    """

    eps_model: UNetModel

    def __init__(
        self,
        unet_model: UNetModel,
        n_steps: int,
        linear_start: float,
        linear_end: float,
    ):
        """
        :param unet_model: is the [U-Net](model/unet.html) that predicts noise
         $\epsilon_\text{cond}(x_t, c)$, in latent space
        :param latent_scaling_factor: is the scaling factor for the latent space. The encodings of
         the autoencoder are scaled by this before feeding into the U-Net.
        :param n_steps: is the number of diffusion steps $T$.
        :param linear_start: is the start of the $\beta$ schedule.
        :param linear_end: is the end of the $\beta$ schedule.
        """
        super().__init__()
        # Wrap the [U-Net](model/unet.html) to keep the same model structure as
        # [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion).
        self.eps_model = unet_model
        # Number of steps $T$
        self.n_steps = n_steps

        # $\beta$ schedule
        beta = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_steps, dtype=torch.float64
            )
            ** 2
        )
        # $\alpha_t = 1 - \beta_t$
        alpha = 1.0 - beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha = nn.Parameter(alpha.to(torch.float32), requires_grad=False)
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)
        self.sigma2 = self.beta

    @property
    def device(self):
        """
        ### Get model device
        """
        return next(iter(self.eps_model.parameters())).device

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor):
        """
        ### Predict noise

        Predict noise given the latent representation $x_t$, time step $t$, and the
        conditioning context $c$.

        $$\epsilon_\text{cond}(x_t, c)$$
        """
        return self.eps_model(x, t, context)

    def q_xt_x0(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        #### Get $q(x_t|x_0)$ distribution
        """

        # [gather](utils.html) $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        # $(1-\bar\alpha_t) \mathbf{I}$
        var = 1 - gather(self.alpha_bar, t)
        #
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

        # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
        eps_theta = self.eps_model(xt, t)
        # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha**0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var**0.5) * eps

    def loss(
        self,
        x0: torch.Tensor,
        cond: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        cond_concat: Optional[torch.Tensor] = None,
    ):
        batch_size = x0.shape[0]
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps=noise)
        if cond_concat is not None:
            xt_concat = torch.concat([xt, cond_concat], dim=1)
            eps_theta = self.eps_model(xt_concat, t, cond)
        else:
            eps_theta = self.eps_model(xt, t, cond)

        return F.mse_loss(noise, eps_theta)


class Polyffusion_SDF(nn.Module):
    def __init__(
        self,
        ldm: LatentDiffusion,
        cond_type,
        cond_mode="cond",
        chord_enc=None,
        chord_dec=None,
    ):
        """
        cond_type: {chord, texture}
        cond_mode: {cond, mix, uncond}
            mix: use a special condition for unconditional learning with probability of 0.2
        use_enc: whether to use pretrained chord encoder to generate encoded condition
        """
        super(Polyffusion_SDF, self).__init__()
        self.ldm = ldm
        self.cond_type = cond_type
        self.cond_mode = cond_mode
        self.chord_enc = chord_enc
        self.chord_dec = chord_dec

        # Freeze params for pretrained chord enc and dec
        if self.chord_enc is not None:
            for param in self.chord_enc.parameters():
                param.requires_grad = False
        if self.chord_dec is not None:
            for param in self.chord_dec.parameters():
                param.requires_grad = False

    @classmethod
    def load_trained(
        cls,
        ldm,
        chkpt_fpath,
        cond_type,
        cond_mode="cond",
        chord_enc=None,
        chord_dec=None,
    ):
        model = cls(
            ldm,
            cond_type,
            cond_mode,
            chord_enc,
            chord_dec,
        )
        trained_leaner = torch.load(chkpt_fpath)
        model.load_state_dict(trained_leaner["model"])
        return model

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        return self.ldm.p_sample(xt, t)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        return self.ldm.q_sample(x0, t)

    def _encode_chord(self, chord):
        if self.chord_enc is not None:
            # z_list = []
            # for chord_seg in chord.split(8, 1):  # (#B, 8, 36) * 4
            #     z_seg = self.chord_enc(chord_seg).mean
            #     z_list.append(z_seg)
            # z = torch.stack(z_list, dim=1)
            z = self.chord_enc(chord).mean
            z = z.unsqueeze(1)  # (#B, 1, 512)
            return z
        else:
            chord_flatten = torch.reshape(
                chord, (-1, 1, chord.shape[1] * chord.shape[2])
            )
            return chord_flatten

    def _decode_chord(self, z):
        if self.chord_dec is not None:
            # chord_list = []
            # for z_seg in z.split(1, 1):
            #     z_seg = z_seg.squeeze()
            #     # print(f"z_seg {z_seg.shape}")
            #     recon_root, recon_chroma, recon_bass = self.chord_dec(
            #         z_seg, inference=True, tfr=0.
            #     )
            #     recon_root = F.one_hot(recon_root.max(-1)[-1], num_classes=12)
            #     recon_chroma = recon_chroma.max(-1)[-1]
            #     recon_bass = F.one_hot(recon_bass.max(-1)[-1], num_classes=12)
            #     # print(recon_root.shape, recon_chroma.shape, recon_bass.shape)
            #     chord_seg = torch.cat([recon_root, recon_chroma, recon_bass], dim=-1)
            #     # print(f"chord seg {chord_seg.shape}")
            #     chord_list.append(chord_seg)
            # chord = torch.cat(chord_list, dim=1)
            # print(f"chord {chord.shape}")
            recon_root, recon_chroma, recon_bass = self.chord_dec(
                z, inference=True, tfr=0.0
            )
            recon_root = F.one_hot(recon_root.max(-1)[-1], num_classes=12)
            recon_chroma = recon_chroma.max(-1)[-1]
            recon_bass = F.one_hot(recon_bass.max(-1)[-1], num_classes=12)
            # print(recon_root.shape, recon_chroma.shape, recon_bass.shape)
            chord = torch.cat([recon_root, recon_chroma, recon_bass], dim=-1)
            return chord
        else:
            return z

    def get_loss_dict(self, batch, step):
        """
        z_y is the stuff the diffusion model needs to learn
        """
        prmat2c, pnotree, chord, prmat = batch
        # estx_to_midi_file(pnotree, "exp/pnotree.mid")
        # chd_to_midi_file(chord, "exp/chd_origin.mid")
        if self.cond_type == "chord":
            cond = self._encode_chord(chord)
        elif self.cond_type == "pnotree":
            cond = self._encode_pnotree(pnotree)
            # recon_pnotree = self._decode_pnotree(cond)
            # estx_to_midi_file(recon_pnotree, "exp/pnotree_decoded.mid")
            # exit(0)
        elif self.cond_type == "txt":
            cond = self._encode_txt(prmat)
        elif self.cond_type == "chord+txt":
            zchd = self._encode_chord(chord)
            ztxt = self._encode_txt(prmat)
            if self.cond_mode == "mix2":
                if random.random() < 0.2:
                    zchd = (-torch.ones_like(zchd)).to(prmat.device)  # a bunch of -1
                if random.random() < 0.2:
                    ztxt = (-torch.ones_like(ztxt)).to(prmat.device)  # a bunch of -1
            cond = torch.cat([zchd, ztxt], dim=-1)
        else:
            raise NotImplementedError
        # recon_chord = self._decode_chord(cond)
        # chd_to_midi_file(recon_chord, "exp/chd_recon.mid")
        # exit(0)

        if self.cond_mode == "uncond":
            cond = (-torch.ones_like(cond)).to(prmat.device)  # a bunch of -1
        elif self.cond_mode == "mix" or self.cond_mode == "mix2":
            if random.random() < 0.2:
                cond = (-torch.ones_like(cond)).to(prmat.device)  # a bunch of -1

        # if self.is_autoregressive:
        #     concat, x = prmat2c.split(64, -2)
        #     loss = self.ldm.loss(x, cond, concat=concat, concat_axis=-2)
        # else:

        if self.concat_blurry:
            blurry_img = get_blurry_image(prmat2c, ratio=self.concat_ratio)
            exit(0)
            loss = self.ldm.loss(prmat2c, cond, cond_concat=blurry_img)

        else:
            loss = self.ldm.loss(prmat2c, cond)
        return {"loss": loss}
