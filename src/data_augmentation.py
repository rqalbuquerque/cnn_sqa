"""Data Augmentation definitions for speech quality assessment.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import numpy as np
import soundfile as sf

"""Flip the speech files along ot time axis.

Args:
  input_dir: Path of the data samples.
  output_dir: Path of the new data samples.
"""
def flip(input_dir, output_dir, ext):
	# Get all subfolders
    database = [x[0] for x in os.walk(input_dir)]

    # Look through all the subfolders to find audio samples
    for subfolder in database[1:]:
      for filepath in glob.glob(os.path.join(subfolder, '*.' + ext)):
        data, rate = sf.read(filepath)
        name = os.path.splitext(filepath)[0]
        sf.write(name + "_flipped.wav", np.flip(data), rate)

if __name__ == '__main__':
	input_dir= '../database/speech_dataset'
	output_dir= '../database/speech_dataset'
	flip(input_dir, output_dir, 'wav')