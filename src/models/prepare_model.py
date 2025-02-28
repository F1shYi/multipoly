from models.unet import UNetModel
from models.chord_encoder import ChordEncoder
from models.diffusion import Diffusion
from models.fixed_params import DIFFUSION_PARAMS_FIXED, POLYFFUSION_PARAMS_FIXED, CHORD_ENCODER_PARAMS_FIXED
import torch

def get_diffusion(polyffusion_ckpts_fpath, chord_encoder_ckpt_fpath, unet_trainable_params, freeze_polyffusion = True):

    unet = UNetModel(**POLYFFUSION_PARAMS_FIXED, **unet_trainable_params)
    polyffusion_checkpoint = torch.load(polyffusion_ckpts_fpath)["model"]
    unet.load_polyffusion_checkpoints(polyffusion_checkpoint, freeze_polyffusion)

    chord_checkpoint = torch.load(chord_encoder_ckpt_fpath)["model"]
    chord_encoder = ChordEncoder(**CHORD_ENCODER_PARAMS_FIXED)
    CHORD_ENC_PREFIX = "chord_enc."
    chord_enc_state_dict = {key.removeprefix(CHORD_ENC_PREFIX):value for key,value in chord_checkpoint.items() if key.startswith(CHORD_ENC_PREFIX)}
    chord_encoder.load_state_dict(chord_enc_state_dict)

    diffusion = Diffusion(unet_model=unet,chord_encoder=chord_encoder, **DIFFUSION_PARAMS_FIXED)

    return diffusion
