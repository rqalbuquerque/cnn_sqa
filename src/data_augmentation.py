"""Data Augmentation definitions for speech quality assessment.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import numpy as np
import argparse

import utils
import input_data

def flip(data):
    """Flip the speech data along ot time axis.

    Args:
      data: sample audio
      output: output flipped audio
    """
    return np.flip(data).tolist()


def rcs(data):
    """Random circular shift the speech data along ot time axis.

    Args:
      data: sample audio
      n: points of random shift
      output: output shifted sample
    """
    [index] = np.random.randint(int(0.1*len(data)), int(0.9*len(data)), 1)
    return np.concatenate((data[index:], data[0:index])).tolist()


def process(data, mode):
  if mode == 'flip':
    return flip(data)
  elif mode == 'rcs':
    return rcs(data)
  raise "Invalid data augmentation algorithm!"


def process(input_dir, output_dir, mode):
  samples = utils.find_by_extension(input_dir, 'wav')
  for sample in samples:
    data = input_data.load_wav_file(input_dir + sample)
    aug_data = process(data, mode)
    input_data.save_wav_file(output_dir + '/' + sample, aug_data, 16000)
    print('Processed sample: ' + sample)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input_dir',
      type=str,
      default='/home/rqa/rqa/Workspace/Databases/Mixed/speech_noise_dataset_suppl23_voip-tcd/speech_dataset_expanded/',
      help='Location of source dir.')
  parser.add_argument(
      '--output_dir',
      type=str,
      default='/home/rqa/rqa/Workspace/Databases/Mixed/speech_noise_dataset_suppl23_voip-tcd/speech_dataset_expanded_augmented/rcs/',
      help='Location of output dir.')
  parser.add_argument(
      '--mode',
      type=str,
      default='rcs',
      help='Data augmentation algorithm.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Sample rate of loaded audio files.')
  
  flags, unparsed = parser.parse_known_args()
  process(
      flags.input_dir,
      flags.output_dir,
      flags.mode
  )
