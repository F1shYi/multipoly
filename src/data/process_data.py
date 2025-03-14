# Goal
# Given a datafolder which contains many MIDIs, 
# generate a bunch of MIDIs, each corresponding to
# an 8-bar segment, where none of the considered tracks is empty. 

import muspy
import numpy as np
from typing import List
from data.utils import midi_to_one_hot_chd, chd_to_midi_file, multi_prmat2c_to_midi_file


def get_track_idx(is_drum = None, program_num = None, track_name = None):
    if is_drum == True:
        return 4
    if program_num == 32 or track_name == "Bass":
        return 0
    if program_num == 24 or track_name == "Guitar":
        return 1
    if program_num == 0 or track_name == "Piano":
        return 2
    if program_num == 48 or track_name == "Strings":
        return 3
    raise ValueError("Invalid Program Number or Track Name!")
   
def get_program_num(track_idx):
    """
    Args:
        track_idx: int
    Returns:
        Tuple(is_drum, program_num)
    """
    TRACKS = [
        (False, 32),
        (False, 24),
        (False, 0),
        (False, 48),
        (True, 0)
    ]
    return TRACKS[track_idx]

def music_to_a_list_of_8bar_segments(music: muspy.Music, threshold: int) -> List[muspy.Music]:
    """
    Convert a piece of music into a list of 8-bar segments.
    Each segment contains 5 tracks: Drum, Bass, Guitar, Piano and String.
    
    Args:
        music (muspy.Music): The 4/4 music after adjusting resolution to 4.
        threshold (int): Only the 8-bar segments with more than or equal to `threshold` bars where all tracks exist will be returned.
    Returns:
        List[muspy.Music]: a list of 8-bar segments.
    """
    
    assert music.resolution == 4
    assert len(music.time_signatures) == 1
    assert music.time_signatures[0].denominator == 4
    assert music.time_signatures[0].numerator == 4
    assert threshold >= 1 and threshold <= 8

    if len(music.tracks) != 5:
        return []
    
    end_timestep = music.get_end_time()
    bar_num = (end_timestep // 16) + 2

    # note_cumulative_sum[track_idx, bar_idx] indicates 
    # the number of notes in track `track_idx`
    # starting before the `bar_idx + 1`-th bar.
    note_cumulative_sum = np.zeros((5, bar_num),dtype=np.int32)
    notes_per_track = [[],[],[],[],[]]
    all_notes = 0
    non_empty_bars = [i for i in range(bar_num)]
    non_empty_nums = np.zeros((bar_num,),dtype=np.int32)
    for track in music.tracks:
        track_idx = get_track_idx(is_drum=track.is_drum, program_num=track.program)
        all_notes += len(track)
        for note in track.notes:
            notes_per_track[track_idx].append((
                note.start,
                note.duration,
                note.pitch
            ))
        notes_per_track[track_idx].sort(key=lambda x: (x[0], x[1], x[2]))
        notes = notes_per_track[track_idx]
        cumulative_sum = 0
        for current_bar in range(bar_num):
            while cumulative_sum < len(notes) and notes[cumulative_sum][0] < (current_bar) * 16:
                cumulative_sum += 1
            note_cumulative_sum[track_idx][current_bar] = cumulative_sum 
        note_num_each_bar = np.diff(note_cumulative_sum[track_idx])
        non_empty_bar = np.argwhere(note_num_each_bar > 0).reshape(-1,)
        non_empty_bars = ([int(i) for i in non_empty_bar if int(i) in non_empty_bars])
    for bar_idx in non_empty_bars:
        for i in range(0,8):
            if bar_idx - i >= 0:
                non_empty_nums[bar_idx - i] += 1
    

    valid_start_bar_idx = np.argwhere(non_empty_nums >= threshold)
    music_segments = [muspy.Music(resolution=4,tempos=[muspy.Tempo(0,120)],time_signatures=[muspy.TimeSignature(0,4,4)]) for _ in range(len(valid_start_bar_idx))]
    for segment_idx, bar_idx in enumerate(valid_start_bar_idx):
        for track_idx in range(5):
            is_drum, program_num = get_program_num(track_idx)
            track = muspy.Track(is_drum=is_drum, program=program_num)
            notes_start_idx = (note_cumulative_sum[track_idx][bar_idx].item())
            notes_end_idx = (note_cumulative_sum[track_idx][bar_idx + 8].item())
            notes = notes_per_track[track_idx][notes_start_idx:notes_end_idx]
            track.notes = [
                muspy.Note(
                    time=note[0] - bar_idx.item()*16,
                    duration=min(note[1], (bar_idx.item()+8)*16 - note[0]),
                    pitch=note[2],
                ) for note in notes]
            music_segments[segment_idx].tracks.append(track)
    
    return music_segments
    

def midi_to_train_val_segs():
    midi_folder = "/root/autodl-tmp/multipoly/data/lmd/lpd_5_midi/"
    train_folder = "/root/autodl-tmp/multipoly/data/train_segs"
    val_folder = "/root/autodl-tmp/multipoly/data/val_segs"

    

    train_seg_num = 0
    val_seg_num = 0
    from tqdm import tqdm
    import os

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    all_midi_fpaths = [os.path.join(midi_folder, path) for path in os.listdir(midi_folder) if path.endswith(".mid")]

    import random

    for midi_fpath in tqdm(all_midi_fpaths):
        music = muspy.read(midi_fpath)
        music_segments = music_to_a_list_of_8bar_segments(music, threshold=8)
        
        if random.random() < 0.1:
            write_mode = "val"
        else:
            write_mode = "train"
        

        for segment in (music_segments):

            if write_mode == "val":
                write_fpath = os.path.join(val_folder, f"seg_{val_seg_num}.mid")
                segment.write_midi(write_fpath)
                val_seg_num += 1
            else:
                write_fpath = os.path.join(train_folder, f"seg_{train_seg_num}.mid")
                segment.write_midi(write_fpath)
                train_seg_num += 1
                
            
    print(f"Finishes processing with {train_seg_num} training segments and {val_seg_num} valid segments.")


def midi_to_npz():
    train_midi_folder = "/root/autodl-tmp/multipoly/data/train_segs_filtered"
    val_midi_folder = "/root/autodl-tmp/multipoly/data/val_segs_filtered"
    train_folder = "/root/autodl-tmp/multipoly/data/train"
    val_folder = "/root/autodl-tmp/multipoly/data/val"
    import os
    import random
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    train_midi_fpaths = [os.path.join(train_midi_folder, path) for path in os.listdir(train_midi_folder) if path.endswith(".mid")]
    val_midi_fpaths = [os.path.join(val_midi_folder, path) for path in os.listdir(val_midi_folder) if path.endswith(".mid")]
    
    from tqdm import tqdm
    for train_idx, train_midi_fpath in tqdm(enumerate(train_midi_fpaths)):
        music = muspy.read_midi(train_midi_fpath)
        multi_prmat_2c = np.zeros((4,2,128,128),dtype=np.float32)
        for track in music.tracks:
            if track.is_drum:
                continue
            if track.program == 32:
                track_idx = 0
            if track.program == 24:
                track_idx = 1
            if track.program == 0:
                track_idx = 2
            if track.program == 48:
                track_idx = 3
            for note in track.notes:
                start_time = note.start
                duration = note.duration
                pitch = note.pitch
                multi_prmat_2c[track_idx, 0, start_time, pitch] = 1.0
                for d in range(1, duration):
                    if start_time + d < 128:
                        multi_prmat_2c[track_idx, 1, start_time + d, pitch] = 1.0
        
        chord = midi_to_one_hot_chd(train_midi_fpath, train_midi_fpath[:-4]+".out")
        np.savez(os.path.join(train_folder,f"{train_idx}.npz"),**{
            "multi_prmat_2c":multi_prmat_2c,
            "onehot_chord":chord
        })
    print(f"Training NPZs saved succeeded at {train_folder}")

    for val_idx, val_midi_fpath in tqdm(enumerate(val_midi_fpaths)):
        music = muspy.read_midi(val_midi_fpath)
        multi_prmat_2c = np.zeros((4,2,128,128),dtype=np.float32)
        for track in music.tracks:
            if track.is_drum:
                continue
            if track.program == 32:
                track_idx = 0
            if track.program == 24:
                track_idx = 1
            if track.program == 0:
                track_idx = 2
            if track.program == 48:
                track_idx = 3
            for note in track.notes:
                start_time = note.start
                duration = note.duration
                pitch = note.pitch
                multi_prmat_2c[track_idx, 0, start_time, pitch] = 1.0
                for d in range(1, duration):
                    if start_time + d < 128:
                        multi_prmat_2c[track_idx, 1, start_time + d, pitch] = 1.0
        
        chord = midi_to_one_hot_chd(val_midi_fpath, val_midi_fpath[:-4]+".out")
        np.savez(os.path.join(val_folder,f"{val_idx}.npz"),**{
            "multi_prmat_2c":multi_prmat_2c,
            "onehot_chord":chord
        })
    print(f"Validating NPZs saved succeeded at {val_folder}")


def filter_midi():
    IN_FOLDER = "/root/autodl-tmp/multipoly/data/val_segs"
    OUT_FOLDER = "/root/autodl-tmp/multipoly/data/val_segs_filtered"
    
    from tqdm import tqdm
    import os
    os.makedirs(OUT_FOLDER, exist_ok=True)
    all_midi_fpaths = [os.path.join(IN_FOLDER, path) for path in os.listdir(IN_FOLDER) if path.endswith(".mid")]
    cnt = 0
    for midi_fpath in tqdm(all_midi_fpaths):
        music = muspy.read_midi(midi_fpath)
        ok_music = True
        for track in music.tracks:
            # piano avg notes count: 93.3577
            # guitar avg notes count: 115.5076
            # bass avg notes count: 38.8402
            # drums avg notes count: 166.7206
            # strings avg notes count: 4999.5

            if track.is_drum:
                continue

            if track.program == 32: # bass
                if len(track) < 40:
                    ok_music = False
                    break
   
            if track.program == 24: # guitar
                if len(track) < 120:
                    ok_music = False
                    break

            if track.program == 0: # piano
                if len(track) < 100:
                    ok_music = False
                    break

            if track.program == 48: # string
                if len(track) > 2500:
                    ok_music = False
                    break
    
        if ok_music == False:
            continue
        else:
            save_fpath = os.path.join(OUT_FOLDER,f"filtered_{cnt}.mid")
            cnt += 1
            music.write_midi(save_fpath)
    print(f"After filtering: {cnt}")
            

def check_valid(folder):
    import os
    from tqdm import tqdm
    all_npz_fpaths = [os.path.join(folder, path) for path in os.listdir(folder) if path.endswith(".npz")]
    invalid_npz_fpaths = []
    for fpath in all_npz_fpaths:
        data = np.load(fpath, allow_pickle=True)
        prmat = data["multi_prmat_2c"]
        chord = data["onehot_chord"]
        if prmat.shape != (4,2,128,128) or chord.shape != (32,36):
            invalid_npz_fpaths.append(fpath)
    print(len(invalid_npz_fpaths))
    for invalid_npz_fpath in invalid_npz_fpaths:
        os.remove(invalid_npz_fpath)
       

if __name__ == "__main__":
    check_valid("/root/autodl-tmp/multipoly/data/val")