from models.prepare_model import get_diffusion_from_ckpts
from data.utils import midi_to_one_hot_chd, chd_to_midi_file, prmat2c_to_midi_file, multi_prmat2c_to_midi_file
#from data.dataloader import get_train_val_dataloaders
import os
import torch
import numpy as np

class WholeSongSampler:
    # TODO: Implement a whole song sampler.
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = get_diffusion_from_ckpts(
            config["models"]["transformers"],
            config["paths"]["diffusion"],
        ).to(self.device)
        self.output_folder = config["paths"]["output"]
        os.makedirs(self.output_folder, exist_ok=True)

        self.scale = config["samples"]["scale"]
        self.temperature = config["samples"]["temperature"]
        if self.scale == 0.0:
            self.chord = None
        else:
            self.chord = self._get_chord_from_config(config)

        self.uncond = -torch.ones(1,1,512).to(self.device)


        # self.chord = torch.tensor(
        #     midi_to_one_hot_chd(config["paths"]["chord"], "tmp.out"),
        #     dtype=torch.float,
        #     device=self.device).reshape(1,32,36)
        
        # self.train_loader, self.val_loader = get_train_val_dataloaders(
        #     config["paths"]["dataset"],
        #     config["data"]["batch_size"],
        #     config["data"]["num_workers"],
        #     config["data"]["train_ratio"],
        #     pin_memory=True,
        #     return_split=False)

        # for batch in self.train_loader:
        #     chord = batch[1]
        #     self.chord = chord[0].reshape(1,32,36).to(self.device)
        #     break
        # chd_to_midi_file(self.chord.reshape(32,36), os.path.join(self.output_folder, "chord_cond.mid"))

       
        
        self.num_gen = config["samples"]["num_gen"]

        
    def _get_chord_from_config(self, config) -> torch.Tensor:
        chord_midi_fpath = config["paths"]["chord"]
        chord = midi_to_one_hot_chd(chord_midi_fpath, chord_midi_fpath[:-4]+".out")
        n_step = chord.shape[0]
        if n_step < 32:
            raise ValueError("Chord too short!")
        elif n_step > 32:
            start = max((np.random.randint(0, n_step // 4) - 8) * 4,0)
            chord = chord[start:start + 32]
        chd_to_midi_file(chord, os.path.join(self.output_folder, "cond_chord.mid"))
        
        chord = chord.reshape(1,32,36)
        return torch.from_numpy(chord).to(self.device)

    def sample(self):
        TRACK_NAME = ["bass", "guitar", "piano", "string"]

        for idx_gen in range(self.num_gen):
            print(f"generating song {idx_gen}")
            save_folder = os.path.join(self.output_folder, f"song_{idx_gen}")
            os.makedirs(save_folder, exist_ok=True)

            gen = self.diffusion.sample(
                shape=[1,4,2,128,128],
                chords=self.chord,
                uncond_scale=self.scale,
                uncond_cond=self.uncond,
                temperature=self.temperature)
            
            gen = gen.cpu().numpy()[0]
            multi_prmat2c_to_midi_file(gen, os.path.join(save_folder, "multi.mid"))
            for track_idx in range(4):
                gen_track = gen[track_idx]
                save_fpath = os.path.join(save_folder, f"track_{TRACK_NAME[track_idx]}.mid")
                prmat2c_to_midi_file(gen_track, save_fpath)
            




        

    
    

