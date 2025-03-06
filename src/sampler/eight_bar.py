from models.prepare_model import get_diffusion_from_ckpts
from data.utils import midi_to_one_hot_chd, chd_to_midi_file, prmat2c_to_midi_file
from data.dataloader import get_train_val_dataloaders
import os
import torch
import numpy as np
class EightBarSamplerFromChord:

    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = get_diffusion_from_ckpts(
            config["models"]["transformers"],
            config["paths"]["diffusion"],
        ).to(self.device)

        # self.chord = torch.tensor(
        #     midi_to_one_hot_chd(config["paths"]["chord"], "tmp.out"),
        #     dtype=torch.float,
        #     device=self.device).reshape(1,32,36)
        
        self.train_loader, self.val_loader = get_train_val_dataloaders(
            config["paths"]["dataset"],
            config["data"]["batch_size"],
            config["data"]["num_workers"],
            config["data"]["train_ratio"],
            pin_memory=True,
            return_split=False)

        for batch in self.train_loader:
            chord = batch[1]
            self.chord = chord[0].reshape(1,32,36).to(self.device)
            break
        
        self.output_folder = config["paths"]["output"]
        os.makedirs(self.output_folder, exist_ok=True)
        self.num_gen = config["num_gen"]

        chd_to_midi_file(self.chord.reshape(32,36), os.path.join(self.output_folder, "chord_cond.mid"))

    def sample(self):
        TRACK_NAME = ["bass", "guitar", "piano", "string"]

        for idx_gen in range(self.num_gen):
            print(f"generating song {idx_gen}")
            save_folder = os.path.join(self.output_folder, f"song_{idx_gen}")
            os.makedirs(save_folder, exist_ok=True)

            gen = self.diffusion.sample(
                shape=[1,4,2,128,128],
                chords=self.chord,
                uncond_scale=1.0)
            
            gen = gen.cpu().numpy()[0]
            for track_idx in range(4):
                gen_track = gen[track_idx]
                print(gen_track)
                print(np.sum(gen_track))
                save_fpath = os.path.join(save_folder, f"track_{TRACK_NAME[track_idx]}.mid")
                prmat2c_to_midi_file(gen_track, save_fpath)
            




        

    
    

