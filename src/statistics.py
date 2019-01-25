"""Generate statistics from data.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

"""Handles normalization with big dataset object."""
class BatchGenerator(object):

  def __init__(self, input_iterator, set_name, batch=0):
    self.iterator = input_iterator
    self.set_name = set_name
    self.batch_size = batch

  def gen_statistics(self, mode, sess):
    if mode == 'min_max':
      return self.get_min_max(sess)
    else:
      return None, None

  def get_min_max(self, sess):
    set_size = self.iterator.set_size(self.set_name)
    min_v = sys.float_info.max
    max_v = -sys.float_info.max
    data_place = tf.placeholder(tf.float32, [None, None], name='input_data')
    min_op = tf.reduce_min(data_place)
    max_op = tf.reduce_max(data_place)

    for i in range(0, set_size, self.batch_size):
      data = self.iterator.get_indexed_samples(self.batch_size, i, self.set_name, sess)
      aux_min, aux_max = sess.run([min_op, max_op], feed_dict={data_place: data})
      max_v = aux_max if aux_max > max_v else max_v
      min_v = aux_min if aux_min < min_v else min_v

    return min_v, max_v