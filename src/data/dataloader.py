import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from .dataset import LMDDataset,DataSampleNpz
from .utils import (
    chd_pitch_shift,
    chd_to_midi_file,
    chd_to_onehot,
    onehot_to_chd,
    pr_mat_pitch_shift,
    prmat2c_to_midi_file,
)

def collate_fn(batch, shift):
    def sample_shift():
        return np.random.choice(np.arange(-6, 6), 1)[0]

    prmat2cs = []
    chords = []
    for b in batch:
        seg_prmat2c = b[0]
        seg_chord = b[1]
        seg_chord = np.array(onehot_to_chd(seg_chord), dtype=np.int32)

        if shift:
            shift_pitch = sample_shift()
            seg_prmat2c = pr_mat_pitch_shift(seg_prmat2c, shift_pitch)
            seg_chord = np.array(chd_pitch_shift(seg_chord, shift_pitch),dtype=np.int32)
    
        seg_chord = chd_to_onehot(seg_chord)

        prmat2cs.append(seg_prmat2c)
        chords.append(seg_chord)
        
    prmat2cs = torch.Tensor(np.array(prmat2cs, np.float32)).float()
    chords = torch.Tensor(np.array(chords, np.float32)).float()
    
    return prmat2cs, chords


def get_train_val_dataloaders(
    data_folder:str, batch_size:int, num_workers=0,train_ratio=0.9, pin_memory=False, 
):
    
    all_fpaths = [os.path.join(data_folder,fpath) for fpath in os.listdir(data_folder) if fpath.endswith(".npz")]
    all_fpaths = np.array(all_fpaths)
    np.random.shuffle(all_fpaths)
    train_num = int(len(all_fpaths)*train_ratio)
    val_num = len(all_fpaths) - train_num

    train_fpaths = all_fpaths[:train_num]
    val_fpaths = all_fpaths[train_num+1:]

    train_samples = [DataSampleNpz(data_fpath) for data_fpath in train_fpaths]
    val_samples = [DataSampleNpz(data_fpath) for data_fpath in val_fpaths]

    train_dataset = LMDDataset(train_samples)
    val_dataset = LMDDataset(val_samples)

    train_dl = DataLoader(
        train_dataset,
        batch_size,
        True,
        collate_fn=lambda x: collate_fn(x, shift=True),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size,
        False,
        collate_fn=lambda x: collate_fn(x, shift=False),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    print(
        f"Dataloader ready: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}, train_segments={len(train_dataset)}, val_segments={len(val_dataset)}"
    )
    return train_dl, val_dl


