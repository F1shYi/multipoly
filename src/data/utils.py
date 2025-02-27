import numpy as np
import muspy
from chord_extractor import extract_chords_from_midi_file
import csv
import mir_eval
import pretty_midi as pm

def get_chord_matrix(fpath):
    """
    chord matrix [M * 14], each line represent the chord of a beat
    same format as mir_eval.chord.encode():
        root_number(1), semitone_bitmap(12), bass_number(1)
    inputs are generated from junyan's algorithm
    """
    ONE_BEAT = 0.5
    file = csv.reader(open(fpath), delimiter="\t")
    beat_cnt = 0
    chords = []
    for line in file:
        start = float(line[0]) / ONE_BEAT
        end = float(line[1]) / ONE_BEAT
        chord = line[2]

        while beat_cnt < int(round(end)):
            beat_cnt += 1
            # see https://craffel.github.io/mir_eval/#mir_eval.chord.encode
            chd_enc = mir_eval.chord.encode(chord)

            root = chd_enc[0]
            # make chroma and bass absolute
            chroma_bitmap = chd_enc[1]
            chroma_bitmap = np.roll(chroma_bitmap, root)
            bass = (chd_enc[2] + root) % 12

            chord_line = [root]
            for _ in chroma_bitmap:
                chord_line.append(_)
            chord_line.append(bass)

            chords.append(chord_line)
    return chords



def chd_pitch_shift(chd, shift):
    chd = chd.copy()
    chd[:, 0] = (chd[:, 0] + shift) % 12
    chd[:, 1:13] = np.roll(chd[:, 1:13], shift, axis=-1)
    chd[:, -1] = (chd[:, -1] + shift) % 12
    return chd

def pr_mat_pitch_shift(pr_mat, shift):
    pr_mat = pr_mat.copy()
    pr_mat = np.roll(pr_mat, shift, -1)
    return pr_mat

def chd_to_onehot(chd):
    n_step = chd.shape[0]
    onehot_chd = np.zeros((n_step, 36), dtype=np.float32)
    onehot_chd[np.arange(n_step), chd[:, 0]] = 1
    onehot_chd[:, 12:24] = chd[:, 1:13]
    onehot_chd[np.arange(n_step), 24 + chd[:, -1]] = 1
    return onehot_chd


def onehot_to_chd(onehot):
    n_step = onehot.shape[0]
    chd = np.zeros((n_step, 14), dtype=np.float32)
    chd[:, 0] = np.argmax(onehot[:, 0:12], axis=1)
    chd[:, 1:13] = onehot[:, 12:24]
    chd[:, 13] = np.argmax(onehot[:, 24:36], axis=1)
    return chd

def get_note_matrix(music: muspy.Music, track_dict):

    notes = []
    for inst in music.tracks:
        if inst.is_drum:
            continue
        for note in inst.notes:
            onset = int(note.time)
            duration = int(note.duration)
            if duration == 0:
                duration = 1
            # this is compulsory because there may be notes
            # with zero duration after adjusting resolution
            notes.append(
                [
                    onset,
                    note.pitch,
                    duration,
                    track_dict[inst.name],
                ]
            )
    
    notes.sort(key=lambda x: (x[0], x[1], x[2]))
    return notes



def get_start_table(notes, db_pos):
    """
    i-th row indicates the starting row of the "notes" array at i-th beat.
    """
    row_cnt = 0
    start_table = {}
    for db in db_pos:
        while row_cnt < len(notes) and notes[row_cnt][0] < db:
            row_cnt += 1
        start_table[db] = row_cnt

    return start_table

def get_downbeat_pos_and_filter(music: muspy.Music):
    """
    simply get the downbeat position of the given midi file
    and whether each downbeat is complete
    "complete" means at least one 4/4 measures after it.
    E.g.,
    [1, 2, 3, 4, 1, 2, 3, 4] is complete.
    [1, 2, 3, 1, 2, 3, 4], [1, 2, 3, 4, 1, 2, 3] are not.
    """
    BIN = 4
    music.infer_barlines_and_beats()
    barlines = music.barlines
    for b in barlines:
        if not float(b.time).is_integer():
            return None, None

    db_pos = [int(b.time) for b in barlines]
    db_pos_diff = np.diff(db_pos).tolist()
    db_pos_diff.append(db_pos_diff[len(db_pos_diff) - 1])
    assert len(db_pos_diff) == len(db_pos)
    db_pos_filter = []
    for i in range(len(db_pos)):
        if db_pos_diff[i] not in {2 * BIN, 4 * BIN, 8 * BIN}:
            db_pos_filter.append(False)
            continue
        length = db_pos_diff[i]
        left = 8 * BIN - length
        idx = i + 1
        bad = False
        while left > 0 and idx < len(db_pos):
            if db_pos_diff[idx] != length:
                bad = True
                break
            left -= length
            idx += 1
        if bad:
            db_pos_filter.append(False)
        else:
            db_pos_filter.append(True)

    return db_pos, db_pos_filter




def chd_to_midi_file(chords, output_fpath, one_beat=0.5):
    """
    retrieve midi from chords
    """
    if "Tensor" in str(type(chords)):
        chords = chords.cpu().detach().numpy()
    midi = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    piano = pm.Instrument(program=piano_program)
    t = 0.0
    for chord in chords:
        
        if chord.shape[0] == 14:
            root = int(chord[0])
            chroma = chord[1:13].astype(int)
            bass = int(chord[13])
        elif chord.shape[0] == 36:
            root = int(chord[0:12].argmax())
            chroma = chord[12:24].astype(int)
            bass = int(chord[24:].argmax())

        chroma = np.roll(chroma, -bass)
        c3 = 48
        for i, n in enumerate(chroma):
            if n == 1:
                note = pm.Note(
                    velocity=80,
                    pitch=c3 + i + bass,
                    start=t * one_beat,
                    end=(t + 1) * one_beat,
                )
                piano.notes.append(note)
        t += 1

    midi.instruments.append(piano)
    midi.write(output_fpath)



def midi_to_npz(midi_fpath, chd_fpath, track_dict = {"Bass":0,"Guitar":1,"Piano":2,"Strings":3}):
    '''
    npz: dict with keys "notes", "chords", "start_table", "db_pos", "db_pos_filter"

    notes: N*4 matrix, where N is the number of total notes, each note represented by (onset, pitch, duration, track)
    chords: M*36 matrix, where M is the number of total beats. Chord represented in one-hot form.

    '''
    music = muspy.read(midi_fpath)
    np_notes = np.array(get_note_matrix(music, track_dict))
    extract_chords_from_midi_file(midi_fpath, chd_fpath)
    chord = np.array(get_chord_matrix(chd_fpath))
    np_chords = chd_to_onehot(chord)
    db_pos, db_pos_filter = get_downbeat_pos_and_filter(music)
    start_table = get_start_table(np_notes, db_pos)

    npz_file = {
            "notes": np_notes,
            "start_table": np.array(start_table),
            "db_pos": np.array(db_pos),
            "db_pos_filter": np.array(db_pos_filter),
            "chord": np_chords,
        }
    return npz_file


def nmat_to_multi_prmat2c(nmat, n_step,track_num):
    pr_mat = np.zeros((track_num, 2, n_step, 128), dtype=np.float32)
    
    for o, p, d, track_idx in nmat:
        if o < n_step:
            pr_mat[track_idx, 0, o, p] = 1.0
            for dd in range(1, d):
                if o + dd < n_step:
                    pr_mat[track_idx, 1, o + dd, p] = 1.0

    return pr_mat

    
def prmat2c_to_midi_file(
    prmat2c, fpath
):
    '''
    single track prmat2c to midi
    '''
    # prmat2c: (2, step, 128)
    if "Tensor" in str(type(prmat2c)):
        prmat2c = prmat2c.cpu().detach().numpy()
    print(f"prmat2c : {prmat2c.shape}")
    midi = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    origin = pm.Instrument(program=piano_program)
    n_step = prmat2c.shape[2]
    onset = prmat2c[0]
    sustain = prmat2c[1]
    for step_ind, step in enumerate(onset):
        for key, on in enumerate(step):
            on = int(round(on))
            if on > 0:
                dur = 1
                while step_ind + dur < n_step:
                    if not (int(round(sustain[step_ind + dur, key])) > 0):
                        break
                    dur += 1
                note = pm.Note(
                    velocity=80,
                    pitch=key,
                    start=step_ind * 1 / 8,
                    end=(step_ind + dur) * 1 / 8,
                )
                origin.notes.append(note)
    
    midi.instruments.append(origin)
    midi.write(fpath)


