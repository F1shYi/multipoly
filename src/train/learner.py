from models.prepare_model import get_diffusion
from data.dataloader import get_train_val_dataloaders

if __name__ == "__main__":

    diffusion = get_diffusion("/root/autodl-tmp/multipoly/polyffusion_ckpts/ldm_chd8bar/sdf+pop909wm_mix16_chd8bar/01-11_102022/chkpts/weights_best.pt",
                              "/root/autodl-tmp/multipoly/pretrained/chd8bar/weights.pt",
                              {"n_intertrack_head":4, "num_intertrack_encoder_layers":1,"intertrack_attention_levels":[2]})
    train_dl, val_dl = get_train_val_dataloaders()
    
