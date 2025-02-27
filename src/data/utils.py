"""
Helper functions for processing midi data
Some notations used in this code:
music: muspy.Music
pr2c: np.array. Binary pianoroll with 2 channels, normally with shape [2, height(pitch), width(timestep)].
    Using 0/1 to indicate whether or not there exists a note's onset or sustain in the first and second channel respectively.
pr1c: np.array. Binary pianoroll with only 1 channels using 0/1 indicating whether or not there exists a note
    , normally with shape [height, width].
midi_fpath: path to midi file.
image_fpath: path to image file.
"""

import muspy
import pypianoroll as pr
import numpy as np

RESOLUTION = 4


def music_from_midi_fpath(midi_fpath):
    music = muspy.read(midi_fpath)
    music.adjust_resolution(4)
    return music

def pr2c_from_music(music:muspy.Music):
    pass
