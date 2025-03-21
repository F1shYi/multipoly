from models.prepare_model import get_diffusion_from_ckpts
from data.utils import (
    midi_to_one_hot_chd,
    chd_to_midi_file,
    prmat2c_to_midi_file,
    multi_prmat2c_to_midi_file,
    whole_song_midi_to_multi_prmat2c,
)

import os
import torch
import numpy as np


class WholeSongSampler:

    def __init__(self, config):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = get_diffusion_from_ckpts(
            config["models"]["transformers"],
            config["paths"]["diffusion"],
            use_conv3d=config["models"]["use_conv3d"]
        ).to(self.device)
        self.output_folder = config["paths"]["output"]
        os.makedirs(self.output_folder, exist_ok=True)
        self.chord_scales = config["samples"]["chord_scales"]
        self.paint_scales = config["samples"]["paint_scales"]
        self.temperature = config["samples"]["temperature"]
        self.num_gen = config["samples"]["num_gen"]

        self.chord, self.orig = self._get_chord_and_multiprmat2c_from_config(
            config)
        print(self.chord.shape, self.orig.shape)
        self.bar_num = self.chord.shape[0] // 4
        print(f"Getting condition with {self.bar_num} bars")

    def _get_chord_and_multiprmat2c_from_config(self, config):
        midi_fpath = config["paths"]["cond"]
        chord = midi_to_one_hot_chd(midi_fpath, midi_fpath[:-4] + ".out")
        chd_to_midi_file(chord, os.path.join(
            self.output_folder, "cond_chord.mid"))

        multi_prmat2c = whole_song_midi_to_multi_prmat2c(midi_fpath)
        multi_prmat2c_to_midi_file(
            multi_prmat2c, os.path.join(self.output_folder, "orig_multi.mid")
        )
        TRACK_NAME = ["bass", "guitar", "piano", "string"]
        for track_idx in range(4):
            track = multi_prmat2c[track_idx]
            save_fpath = os.path.join(
                self.output_folder, f"orig_{TRACK_NAME[track_idx]}.mid"
            )
            prmat2c_to_midi_file(track, save_fpath)
        assert chord.shape[0] // 4 == multi_prmat2c.shape[2] // 16
        if chord.shape[0] % 32 != 0:
            chord = chord[: -(chord.shape[0] % 32)]
            multi_prmat2c = multi_prmat2c[:, :,
                                          : -(multi_prmat2c.shape[2] % 128)]
        return torch.from_numpy(chord).to(self.device), torch.from_numpy(
            multi_prmat2c
        ).to(self.device)

    def sample_non_ar_given_chord_only(self):
        TRACK_NAME = ["bass", "guitar", "piano", "string"]
        chord = self.chord.reshape(-1, 32, 36)
        eight_bar_nums = chord.shape[0]

        for scale in self.chord_scales:
            folder = os.path.join(self.output_folder, f"scale_{scale}")
            os.makedirs(folder, exist_ok=True)
            folder = os.path.join(folder, "chord_only_non_ar")
            os.makedirs(folder, exist_ok=True)

            for idx_gen in range(self.num_gen):
                save_folder = os.path.join(folder, f"song_{idx_gen}")
                os.makedirs(save_folder, exist_ok=True)
                gens = []
                for bar in range(eight_bar_nums):
                    gen = self.diffusion.sample(
                        shape=[1, 4, 2, 128, 128],
                        chords=chord[bar].reshape(1, 32, 36),
                        uncond_scale=scale,
                        uncond_cond=-torch.ones(1, 1, 512).to(self.device),
                        temperature=self.temperature,
                    )
                    gens.append(gen[0].cpu().numpy())

                gen = np.concatenate(gens, axis=2)
                multi_prmat2c_to_midi_file(
                    gen, os.path.join(save_folder, "multi.mid"))
                for track_idx in range(4):
                    gen_track = gen[track_idx]
                    save_fpath = os.path.join(
                        save_folder, f"track_{TRACK_NAME[track_idx]}.mid"
                    )
                    prmat2c_to_midi_file(gen_track, save_fpath)

    def sample_ar_given_chord_only(self):
        TRACK_NAME = ["bass", "guitar", "piano", "string"]
        for scale in self.chord_scales:
            folder = os.path.join(self.output_folder, f"scale_{scale}")
            os.makedirs(folder, exist_ok=True)
            folder = os.path.join(folder, "chord_only_ar")
            os.makedirs(folder, exist_ok=True)

            for idx_gen in range(self.num_gen):
                save_folder = os.path.join(folder, f"song_{idx_gen}")
                os.makedirs(save_folder, exist_ok=True)
                final_gen = self._sample_ar_given_chord_only_per_scale(scale)
                multi_prmat2c_to_midi_file(
                    final_gen, os.path.join(save_folder, "multi.mid")
                )
                for track_idx in range(4):
                    gen_track = final_gen[track_idx]
                    save_fpath = os.path.join(
                        save_folder, f"track_{TRACK_NAME[track_idx]}.mid"
                    )
                    prmat2c_to_midi_file(gen_track, save_fpath)

    def sample_non_ar_given_chord_and_track(self):
        TRACK_NAME = ["bass", "guitar", "piano", "string"]
        chord = self.chord.reshape(-1, 32, 36)
        eight_bar_nums = chord.shape[0]

        for scale in self.paint_scales:
            folder = os.path.join(self.output_folder, f"scale_{scale}")
            os.makedirs(folder, exist_ok=True)
            folder = os.path.join(folder, "chord_and_track_non_ar")
            os.makedirs(folder, exist_ok=True)
            keep_indices = []
            for keep_idx in range(4):
                track_slice = self.orig[keep_idx]
                if torch.sum(track_slice) <= 10.0:
                    continue
                keep_indices.append(keep_idx)

            for keep_idx in keep_indices:
                print(
                    f"Valid tracks to keep as condition: {TRACK_NAME[keep_idx]}")

            for keep_idx in keep_indices:
                inst_folder = os.path.join(
                    folder, f"given_{TRACK_NAME[keep_idx]}")
                os.makedirs(inst_folder, exist_ok=True)
                mask = torch.zeros([1, 4, 2, 128, 128]).to(self.device)
                mask[:, keep_idx] = 1.0
                for idx_gen in range(self.num_gen):
                    save_folder = os.path.join(inst_folder, f"song_{idx_gen}")
                    os.makedirs(save_folder, exist_ok=True)
                    gens = []
                    for bar in range(eight_bar_nums):
                        gen = self.diffusion.paint(
                            shape=[1, 4, 2, 128, 128],
                            chords=chord[bar].reshape(1, 32, 36),
                            uncond_scale=scale,
                            uncond_cond=-torch.ones(1, 1, 512).to(self.device),
                            orig=self.orig[
                                :, :, bar * 8 * 16: bar * 8 * 16 + 128, :
                            ].reshape(1, 4, 2, 128, 128),
                            mask=mask,
                        )
                        gens.append(gen[0].cpu().numpy())

                    gen = np.concatenate(gens, axis=2)

                    multi_prmat2c_to_midi_file(
                        gen, os.path.join(save_folder, "multi.mid")
                    )
                    for track_idx in range(4):
                        gen_track = gen[track_idx]
                        save_fpath = os.path.join(
                            save_folder, f"track_{TRACK_NAME[track_idx]}.mid"
                        )
                        prmat2c_to_midi_file(gen_track, save_fpath)

    def sample_ar_given_chord_and_track(self):
        TRACK_NAME = ["bass", "guitar", "piano", "string"]

        for scale in self.paint_scales:
            folder = os.path.join(self.output_folder, f"scale_{scale}")
            os.makedirs(folder, exist_ok=True)
            folder = os.path.join(folder, "chord_and_track_ar")
            os.makedirs(folder, exist_ok=True)
            keep_indices = []
            for keep_idx in range(4):
                track_slice = self.orig[keep_idx]
                if torch.sum(track_slice) <= 10.0:
                    continue
                keep_indices.append(keep_idx)

            for keep_idx in keep_indices:
                print(
                    f"Valid tracks to keep as condition: {TRACK_NAME[keep_idx]}")

            for keep_idx in keep_indices:
                inst_folder = os.path.join(
                    folder, f"given_{TRACK_NAME[keep_idx]}")
                os.makedirs(inst_folder, exist_ok=True)
                for idx_gen in range(self.num_gen):
                    save_folder = os.path.join(inst_folder, f"song_{idx_gen}")
                    os.makedirs(save_folder, exist_ok=True)
                    gen = self._sample_ar_given_chord_and_track_per_scale_per_keepidx(
                        scale, keep_idx
                    )
                    multi_prmat2c_to_midi_file(
                        gen, os.path.join(save_folder, "multi.mid")
                    )
                    for track_idx in range(4):
                        gen_track = gen[track_idx]
                        save_fpath = os.path.join(
                            save_folder, f"track_{TRACK_NAME[track_idx]}.mid"
                        )
                        prmat2c_to_midi_file(gen_track, save_fpath)

    def _sample_ar_given_chord_only_per_scale(self, scale):

        final_gen = np.zeros([4, 2, 16 * self.bar_num, 128])
        print(f"Generating samples of shape {final_gen.shape}...")

        # Generate the 8 bars at the beginning
        chord = self.chord[0:32, :].reshape(-1, 32, 36)
        gen = self.diffusion.sample(
            shape=[1, 4, 2, 128, 128],
            chords=chord,
            uncond_scale=scale,
            uncond_cond=-torch.ones(1, 1, 512).to(self.device),
            temperature=self.temperature,
        )
        final_gen[:, :, 0:128, :] = gen.cpu().numpy()[0]

        # Autoregressive generation
        for start_bar in range(4, self.bar_num - 7, 4):
            chord = self.chord[start_bar * 4: start_bar * 4 + 32, :].reshape(
                -1, 32, 36
            )
            orig = torch.zeros([1, 4, 2, 128, 128]).to(self.device)
            orig[:, :, :, 0:64, :] = gen[:, :, :, 64:, :]
            mask = torch.zeros_like(orig)
            mask[:, :, :, 0:64, :] = torch.ones_like(orig[:, :, :, 0:64, :])
            mask = mask.to(self.device)

            gen = self.diffusion.paint(
                shape=[1, 4, 2, 128, 128],
                chords=chord,
                uncond_scale=scale,
                uncond_cond=-torch.ones(1, 1, 512).to(self.device),
                orig=orig,
                mask=mask,
            )
            final_gen[:, :, (start_bar + 4) * 16: (start_bar + 8) * 16, :] = (
                gen[:, :, :, 64:, :].cpu().numpy()[0]
            )

        return final_gen

    def _sample_ar_given_chord_and_track_per_scale_per_keepidx(self, scale, keepidx):
        final_gen = np.zeros([4, 2, 16 * self.bar_num, 128])
        print(f"Generating samples of shape {final_gen.shape}...")

        # Generate the 8 bars at the beginning
        chord = self.chord[0:32, :].reshape(-1, 32, 36)
        orig = self.orig[:, :, 0:128, :].reshape(1, 4, 2, 128, 128)
        mask = torch.zeros_like(orig)
        mask[:, keepidx, :, :, :] = torch.ones_like(mask[:, keepidx, :, :, :])
        mask = mask.to(self.device)

        gen = self.diffusion.paint(
            shape=[1, 4, 2, 128, 128],
            chords=chord,
            uncond_scale=scale,
            uncond_cond=-torch.ones(1, 1, 512).to(self.device),
            orig=orig,
            mask=mask,
        )
        final_gen[:, :, 0:128, :] = gen.cpu().numpy()[0]

        # Autoregressive generation
        for start_bar in range(4, self.bar_num - 7, 4):
            chord = self.chord[start_bar * 4: start_bar * 4 + 32, :].reshape(
                -1, 32, 36
            )
            orig = torch.zeros([1, 4, 2, 128, 128]).to(self.device)
            orig[:, :, :, 0:64, :] = gen[:, :, :, 64:, :]
            orig[:, keepidx, :, :, :] = self.orig[
                keepidx, :, start_bar * 16: start_bar * 16 + 128, :
            ]
            mask = torch.zeros_like(orig)
            mask[:, :, :, 0:64, :] = torch.ones_like(mask[:, :, :, 0:64, :])
            mask[:, keepidx, :, :, :] = torch.ones_like(
                mask[:, keepidx, :, :, :])
            mask = mask.to(self.device)

            gen = self.diffusion.paint(
                shape=[1, 4, 2, 128, 128],
                chords=chord,
                uncond_scale=scale,
                uncond_cond=-torch.ones(1, 1, 512).to(self.device),
                orig=orig,
                mask=mask,
            )
            final_gen[:, :, (start_bar + 4) * 16: (start_bar + 8) * 16, :] = (
                gen[:, :, :, 64:, :].cpu().numpy()[0]
            )
        return final_gen

    def run(
        self,
        non_ar_chord=True,
        ar_chord=True,
        non_ar_chord_track=True,
        ar_chord_track=True,
    ):
        # 1. non-ar chord-only
        if non_ar_chord:
            self.sample_non_ar_given_chord_only()
        # 2. ar chord-only
        if ar_chord:
            self.sample_ar_given_chord_only()
        # 3. non-ar chord + track
        if non_ar_chord_track:
            self.sample_non_ar_given_chord_and_track()
        # 4. ar chord + track
        if ar_chord_track:
            self.sample_ar_given_chord_and_track()
