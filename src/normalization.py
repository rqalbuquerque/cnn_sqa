"""Normalize data for speech quality assessment.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

def normalize(data, mode, args):
  if mode == 'min_max':
    return norm_by_min_max(data, args[0], args[1])
  elif mode == 'min':
    return norm_by_min(data, args[0])

def norm_by_min_max(data, min_v, max_v):
  return (data - min_v)/(max_v - min_v)

def norm_by_min(data, min_v):
  return data - min_v