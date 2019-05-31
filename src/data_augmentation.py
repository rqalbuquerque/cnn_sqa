"""Data Augmentation definitions for speech quality assessment.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import numpy as np
import soundfile as sf


def flip(data):
    """Flip the speech data along ot time axis.

    Args:
      data: sample audio
      output: output flipped audio
    """
    return np.flip(data)


def random_circular_shift(data):
    """Random circular shift the speech data along ot time axis.

    Args:
      data: sample audio
      n: points of random shift
      output: output shifted sample
    """
    [index] = np.random.randint(int(0.1*len(data)), int(0.9*len(data)), 1)
    return np.concatenate((data[index:], data[0:index]))


def apply(data, mode):
    if mode == 'flip':
        return flip(data)
    elif mode == 'random_circular_shift':
        return random_circular_shift(data)


def load_and_apply(input_dir, output_dir, ext):
        # Get all subfolders
    database = [x[0] for x in os.walk(input_dir)]

    # Look through all the subfolders to find audio samples
    for subfolder in database[1:]:
        for filepath in glob.glob(os.path.join(subfolder, '*.' + ext)):
            data, rate = sf.read(filepath)
            name = os.path.splitext(filepath)[0]
            sf.write(name + "_flipped.wav", flip(data), rate)


if __name__ == '__main__':
    input_dir = '../database/speech_dataset'
    output_dir = '../database/speech_dataset'
    flip(input_dir, output_dir, 'wav')
