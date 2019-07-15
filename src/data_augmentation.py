"""Data Augmentation definitions for speech quality assessment.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import numpy as np

def flip(data):
    """Flip the speech data along ot time axis.

    Args:
      data: sample audio
      output: output flipped audio
    """
    return np.flip(data).tolist()


def random_circular_shift(data):
    """Random circular shift the speech data along ot time axis.

    Args:
      data: sample audio
      n: points of random shift
      output: output shifted sample
    """
    [index] = np.random.randint(int(0.1*len(data)), int(0.9*len(data)), 1)
    return np.concatenate((data[index:], data[0:index])).tolist()


def apply(data, mode):
  if mode == 'flip':
    return flip(data)
  elif mode == 'random_circular_shift':
    return random_circular_shift(data)
  raise "Invalid data augmentation algorithm!"
