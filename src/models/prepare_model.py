from models.unet import UNetModel
from models.chord_encoder import ChordEncoder
from models.diffusion import Diffusion
from models.fixed_params import DIFFUSION_PARAMS_FIXED, POLYFFUSION_PARAMS_FIXED, CHORD_ENCODER_PARAMS_FIXED
import torch


def get_diffusion(polyffusion_ckpts_fpath, chord_encoder_ckpt_fpath, unet_trainable_params, use_polyffuion=True, freeze_polyffusion=True, zero=True, use_conv3d=False):

    unet = UNetModel(**POLYFFUSION_PARAMS_FIXED, **
                     unet_trainable_params, use_conv3d=use_conv3d)
    if use_polyffuion:
        polyffusion_checkpoint = torch.load(polyffusion_ckpts_fpath)["model"]
        unet.load_polyffusion_checkpoints(
            polyffusion_checkpoint, freeze_polyffusion, zero)

    chord_checkpoint = torch.load(chord_encoder_ckpt_fpath)["model"]
    chord_encoder = ChordEncoder(**CHORD_ENCODER_PARAMS_FIXED)
    CHORD_ENC_PREFIX = "chord_enc."
    chord_enc_state_dict = {key.removeprefix(
        CHORD_ENC_PREFIX): value for key, value in chord_checkpoint.items() if key.startswith(CHORD_ENC_PREFIX)}
    chord_encoder.load_state_dict(chord_enc_state_dict)

    diffusion = Diffusion(
        unet_model=unet, chord_encoder=chord_encoder, **DIFFUSION_PARAMS_FIXED)

    return diffusion


def get_diffusion_from_ckpts(unet_trainable_params, diffusion_ckpts_fpath, use_conv3d=False):
    unet = UNetModel(**POLYFFUSION_PARAMS_FIXED, **
                     unet_trainable_params, use_conv3d=use_conv3d)
    chord_encoder = ChordEncoder(**CHORD_ENCODER_PARAMS_FIXED)
    diffusion = Diffusion(
        unet_model=unet, chord_encoder=chord_encoder, **DIFFUSION_PARAMS_FIXED)
    diffusion_checkpoint = torch.load(diffusion_ckpts_fpath)[
        "model_state_dict"]
    diffusion.load_state_dict(diffusion_checkpoint, strict=False)
    return diffusion
