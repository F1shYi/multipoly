from models.prepare_model import get_diffusion_from_ckpts
from data.utils import midi_to_one_hot_chd, chd_to_midi_file, prmat2c_to_midi_file, multi_prmat2c_to_midi_file, midi_to_multi_prmat2c
# from data.dataloader import get_train_val_dataloaders
import os
import torch
import numpy as np


class EightBarSampler:

    def __init__(self, config):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = get_diffusion_from_ckpts(
            config["models"]["transformers"],
            config["paths"]["diffusion"],
        ).to(self.device)
        self.output_folder = config["paths"]["output"]
        os.makedirs(self.output_folder, exist_ok=True)
        self.uncond = -torch.ones(1, 1, 512).to(self.device)
        self.scales = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.temperature = config["samples"]["temperature"]
        self.num_gen = config["samples"]["num_gen"]

        self.chord, self.orig = self._get_chord_and_multiprmat2c_from_config(
            self, config)

    def _get_chord_and_multiprmat2c_from_config(self, config):
        midi_fpath = config["paths"]["cond"]
        chord = midi_to_one_hot_chd(
            midi_fpath, midi_fpath[:-4]+".out")
        chd_to_midi_file(chord, os.path.join(
            self.output_folder, "cond_chord.mid"))
        multi_prmat2c = midi_to_multi_prmat2c(midi_fpath)
        multi_prmat2c_to_midi_file(
            multi_prmat2c, os.path.join(self.output_folder, "orig_multi.mid"))
        TRACK_NAME = ["bass", "guitar", "piano", "string"]
        for track_idx in range(4):
            track = multi_prmat2c[track_idx]
            save_fpath = os.path.join(
                self.output_folder, f"orig_{TRACK_NAME[track_idx]}.mid")
            prmat2c_to_midi_file(track, save_fpath)
        chord = chord.reshape(1, 32, 36)
        return torch.from_numpy(chord).to(self.device), torch.from_numpy(multi_prmat2c).to(self.device)

    def sample(self):
        TRACK_NAME = ["bass", "guitar", "piano", "string"]

        for scale in self.scales:
            folder = os.path.join(self.output_folder, f"scale_{scale}")
            os.makedirs(folder, exist_ok=True)

            for idx_gen in range(self.num_gen):

                save_folder = os.path.join(
                    folder, f"song_{idx_gen}")
                os.makedirs(save_folder, exist_ok=True)

                gen = self.diffusion.sample(
                    shape=[1, 4, 2, 128, 128],
                    chords=self.chord,
                    uncond_scale=scale,
                    uncond_cond=self.uncond,
                    temperature=self.temperature)

                gen = gen.cpu().numpy()[0]
                multi_prmat2c_to_midi_file(
                    gen, os.path.join(save_folder, "multi.mid"))
                for track_idx in range(4):
                    gen_track = gen[track_idx]
                    save_fpath = os.path.join(
                        save_folder, f"track_{TRACK_NAME[track_idx]}.mid")
                    prmat2c_to_midi_file(gen_track, save_fpath)

    def paint(self):
        NAME = ["bass", "guitar", "piano", "string"]
        for scale in self.scales:
            folder = os.path.join(self.output_folder, f"scale_{scale}")
            os.makedirs(folder, exist_ok=True)
            for idx_gen in range(self.num_gen):
                for keep_idx in [0, 1, 2, 3]:
                    gen_folder = os.path.join(
                        folder, f"{NAME[keep_idx]}_song_{idx_gen}")

                    orig = self.orig
                    mask = torch.zeros_like(orig)

                    mask[keep_idx] = torch.ones_like(mask[keep_idx])

                    orig = orig.reshape(1, *orig.shape).to(self.device)
                    mask = mask.reshape(1, *mask.shape).to(self.device)
                    gen = self.diffusion.paint(
                        shape=[1, 4, 2, 128, 128],
                        orig=orig,
                        mask=mask,
                        chords=self.chord,
                        uncond_scale=scale,
                        uncond_cond=self.uncond
                    )
                    gen = gen.detach().cpu().numpy()[0]
                    multi_prmat2c_to_midi_file(
                        gen, os.path.join(gen_folder, f"multi.mid"))
                    for track_idx, track in enumerate(gen):
                        midi_fpath = os.path.join(
                            gen_folder, f"track_{NAME[track_idx]}.mid")
                        prmat2c_to_midi_file(track, midi_fpath)

    def run(self):
        self.sample()
        self.paint()
