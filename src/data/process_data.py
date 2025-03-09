# Goal
# Given a datafolder which contains many MIDIs, 
# generate a bunch of MIDIs, each corresponding to
# an 8-bar segment, where none of the considered tracks is empty. 

import muspy
import numpy as np
from typing import List
from data.utils import midi_to_one_hot_chd


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
    

def midi2seg():
    midi_folder = "/root/autodl-tmp/multipoly/data/lmd/lpd_5_midi/"
    write_folder1 = "/root/autodl-tmp/multipoly/data/segments"
    # write_folder2 = "/root/autodl-fs/segments"

    

    seg_num = 0
    from tqdm import tqdm
    import os

    os.makedirs(write_folder1, exist_ok=True)
    # os.makedirs(write_folder2, exist_ok=True)
    all_midi_fpaths = [os.path.join(midi_folder, path) for path in os.listdir(midi_folder) if path.endswith(".mid")]

    for midi_fpath in tqdm(all_midi_fpaths):
        music = muspy.read(midi_fpath)
        music_segments = music_to_a_list_of_8bar_segments(music, threshold=8)
        for segment in (music_segments):
            write_fpath1 = os.path.join(write_folder1, f"seg_{seg_num}.mid")
            # write_fpath2 = os.path.join(write_folder2, f"set_{seg_num}.mid")
        
            segment.write_midi(write_fpath1)
            # segment.write_midi(write_fpath2)
            seg_num += 1
    print(f"Finishes processing with {seg_num} segments.")


def midi_to_npz():
    midi_folder = "/root/autodl-tmp/multipoly/data/segments"
    train_folder = "/root/autodl-tmp/multipoly/data/train"
    val_folder = "/root/autodl-tmp/multipoly/data/val"
    import os
    import random
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    all_midi_fpaths = [os.path.join(midi_folder, path) for path in os.listdir(midi_folder) if path.endswith(".mid")]
    random.shuffle(all_midi_fpaths)
    
    train_length = int(len(all_midi_fpaths)*0.8)

    train_midi_fpaths = all_midi_fpaths[:train_length]
    val_midi_fpaths = all_midi_fpaths[train_length:]
    from tqdm import tqdm
    for train_idx, train_midi_fpath in tqdm(enumerate(train_midi_fpaths)):
        music = muspy.read_midi(train_midi_fpath)
        multi_prmat_2c = np.zeros((4,2,128,128))
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
        multi_prmat_2c = np.zeros((4,2,128,128))
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
        np.savez(os.path.join(val_folder,f"{val_idx}.npz"),{
            "multi_prmat_2c":multi_prmat_2c,
            "onehot_chord":chord
        })
    print(f"Validating NPZs saved succeeded at {val_folder}")

if __name__ == "__main__":
    midi_to_npz()