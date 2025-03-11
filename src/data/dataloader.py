import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from .dataset import LMDDataset,DataSampleNpz,EightBarSegmentDataset
from .utils import (
    onehot_chd_pitch_shift,
    chd_to_midi_file,
    pr_mat_pitch_shift,
    prmat2c_to_midi_file,
)

def collate_fn(batch, shift):
    
    prmat2cs = []
    chords = []
    for b in batch:
        seg_prmat2c = b[0]
        seg_chord = b[1]
        if shift:
            shift_pitch = np.random.choice(np.arange(-6, 6), 1)[0]
            seg_prmat2c = pr_mat_pitch_shift(seg_prmat2c, shift_pitch)
            seg_chord = onehot_chd_pitch_shift(seg_chord, shift_pitch)
    
        prmat2cs.append(seg_prmat2c)
        chords.append(seg_chord)
        
    ret_prmat2cs = torch.Tensor(np.array(prmat2cs, np.float32)).float()
    ret_chords = torch.Tensor(np.array(chords, np.float32)).float()
    
    return ret_prmat2cs, ret_chords


def get_train_val_datas(
    train_folder:str,val_folder:str, train_bs:int, val_bs:int, num_workers=0, pin_memory=False
):
    train_ds = EightBarSegmentDataset(train_folder)
    val_ds = EightBarSegmentDataset(val_folder)
    train_dl = DataLoader(train_ds, batch_size=train_bs, shuffle=True,
                        collate_fn=lambda x: collate_fn(x, shift=True),
                        pin_memory=pin_memory, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=val_bs, shuffle=True,
                        collate_fn=lambda x: collate_fn(x, shift=False),
                        pin_memory=pin_memory, num_workers=num_workers)
    return train_ds, val_ds, train_dl, val_dl

   

