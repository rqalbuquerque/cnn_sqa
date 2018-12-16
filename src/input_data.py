"""Input data setting definitions for speech quality assessment.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import os.path
import random
import re
import sys

import numpy as np
from six.moves import xrange

import data_augmentation

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
RANDOM_SEED = 59185


"""Determines which data partition the file should belong to.

Args:
  filename: File path of the data sample.
  validation_percentage: How much of the data set to use for validation.
  testing_percentage: How much of the data set to use for testing.

Returns:
  String, one of 'training', 'validation', or 'testing'.
"""
def which_set(filename, validation_percentage, testing_percentage):
  base_name = os.path.basename(filename)
  hash_name_hashed = hashlib.sha1(compat.as_bytes(base_name)).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS))
 
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
 
  return result


def apply_data_augmentation(data, mode):
  if mode == 'flip':
    return data_augmentation.flip(data)
  elif mode == 'random_circular_shift':
    return data_augmentation.random_circular_shift(data)

"""Handles loading, partitioning, and preparing audio training data."""
class AudioProcessor(object):

  def __init__(self, data_dir, validation_percentage, testing_percentage, model_settings):
    self.prepare_data_index(data_dir, validation_percentage, testing_percentage)
    self.prepare_processing_input_graph(model_settings)
    self.prepare_processing_feature_graph(model_settings)

  """Prepares a list of the samples organized by set.

  The training loop needs a list of all the available data.
  This function analyzes the folders below the `data_dir`.
  For each sub-folder is necessary a set of wav files and one
  file with scores called 'scores.txt' in the format.:

    fileName1.wav scoreFile1 ...
    fileName2.wav scoreFile2 ...
    .
    .
    .
    fileNameN.wav scoreFileN ...

  Args:
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    Dictionary containing a list of files.

  Raises:
    Exception: If expected files are not found.
  """
  def prepare_data_index(self, data_dir, validation_percentage, testing_percentage):
    # Make sure the shuffling.
    self.data_index = {'validation': [], 'testing': [], 'training': []}
    # Get all subfolders
    database = [x[0] for x in os.walk(data_dir)]
    database = database[1:] if len(database) > 1 else database
    # Look through all the subfolders
    for folder in database:
      scores_path = os.path.join(folder, '*scores*.txt')
      scores_file = gfile.Glob(scores_path)[0]
      # Interates over file to get file names and score
      with open(scores_file) as file:
        for line in file.readlines():
          info = line.split(' ')
          wav_path = folder + '/' + info[0]
          score = float(info[1])
          set_index = which_set(wav_path, validation_percentage, testing_percentage)
          self.data_index[set_index].append({'score': score, 'file': wav_path})
    # Make sure the ordering is random.
    for set_index in ['validation', 'testing', 'training']:
      random.shuffle(self.data_index[set_index])

  def set_size(self, set_index):
    return len(self.data_index[set_index]) if set_index in {'training', 'validation', 'testing'} else -1 

  def set_indexes(self, set_index):
    return self.data_index[set_index] if set_index in {'training', 'validation', 'testing'} else [] 

  def calc_log_mel_spec(self, spectrogram, sample_rate):
    # Warp the linear scale spectrograms into the mel-scale.
      num_spectrogram_bins = spectrogram.shape[-1].value
      
      lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
      linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
      
      mel_spectrogram = tf.tensordot(
        tf.abs(spectrogram), linear_to_mel_weight_matrix, 1)
      mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

      # Compute a stabilized log to get log-magnitude mel-scale spectrogram.
      return tf.log(mel_spectrogram + 1e-6)

  def calc_mfcc(self, spectrogram, dct_coefficient_count, sample_rate):
    log_mel_spectrogram = self.calc_log_mel_spec(spectrogram, sample_rate)

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    return tf.contrib.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrogram)[..., :dct_coefficient_count]


  """Builds a TensorFlow graph to apply the input distortions.

  Creates a graph that loads a WAVE file, decodes it, calculates a spectrogram, 
  and then builds an feature selected fingerprint from that.

  This must be called with an active TensorFlow session running, and it
  creates multiple placeholder inputs, and one output:

    - wav_filename_placeholder_: Filename of the WAV to load.
    - feature_: Calculated feature

  Args:
    model_settings: Information about the current model being trained.
  """
  def prepare_processing_input_graph(self, model_settings):
    with tf.name_scope('input'):
      self.wav_filename_placeholder_ = tf.placeholder(tf.string, [], 'file_name')
      self.waveforme = self.prepare_load_wav_graph("load_old", self.wav_filename_placeholder_, model_settings)

  def prepare_processing_feature_graph(self, model_settings):
    with tf.name_scope('feature'):
      self.waveforme_placeholder_ = tf.placeholder(tf.float32, [model_settings['desired_samples'], 1], 'waveform')
      self.spectrogram = self.prepare_spectrogram_graph("spec_old", self.waveforme_placeholder_, model_settings)
      self.feature = self.prepare_feature_graph(self.spectrogram, model_settings)

  def prepare_load_wav_graph(self, mode, wav_filename_placeholder, model_settings):
    if mode == "load_old":
      wav_loader = io_ops.read_file(wav_filename_placeholder)
      wav_decoder = contrib_audio.decode_wav(
          wav_loader, 
          desired_channels=1, 
          desired_samples=model_settings['desired_samples'])
      return wav_decoder.audio
    elif mode == "load_new":
      wav_loader = tf.read_file(wav_filename_placeholder)
      wav_decoder = tf.contrib.ffmpeg.decode_audio(
          wav_loader,
          file_format="wav",
          samples_per_second=model_settings['sample_rate'],
          channel_count=1)
      return tf.transpose(wav_decoder)

  def prepare_spectrogram_graph(self, mode, waveforme, model_settings):
    if mode == "spec_old":
      spectrogram = contrib_audio.audio_spectrogram(
          waveforme, 
          window_size=model_settings['window_size_samples'], 
          stride=model_settings['window_stride_samples'], 
          magnitude_squared=False)
      return spectrogram
    elif mode == "spec_new":
      spectrogram = tf.contrib.signal.stft(
          waveforme, 
          frame_length=model_settings['window_size_samples'], 
          frame_step=model_settings['window_stride_samples'], 
          fft_length=model_settings['window_size_samples'])
      return spectrogram

  def prepare_feature_graph(self, spectrogram, model_settings):
    if model_settings["feature_used"] == "spectrogram":
      frames_count = model_settings["spectrogram_length"]
      coeffic_count = model_settings['dct_coefficient_count']
      feature = tf.slice(
          spectrogram, 
          [0,0,0],
          [-1,frames_count,coeffic_count])
    elif model_settings["feature_used"] == "mfcc": 
      feature = contrib_audio.mfcc(
          tf.real(spectrogram),
          model_settings['sample_rate'],
          dct_coefficient_count=model_settings['dct_coefficient_count'])
    elif model_settings["feature_used"] == "mfcc2":
      feature = self.calc_mfcc(
        spectrogram, model_settings['dct_coefficient_count'], model_settings['sample_rate'])
    elif model_settings["feature_used"] == "log_mel_spectrogram":
      feature = self.calc_log_mel_spec(
        spectrogram, model_settings['sample_rate'])
    return feature

  """Gather samples from the data set, applying transformations as needed.

  When the mode is 'training', a random selection of samples will be returned,
  otherwise the first N clips in the partition will be used. This ensures that
  validation always uses the same samples, reducing noise in the metrics.

  Args:
    qty: Desired number of samples to return. -1 means the entire
      contents of this partition.
    offset: Where to start when fetching deterministically.
    model_settings: Information about the current model being trained.
    mode: Which partition to use, must be 'training', 'validation', or
      'testing'.
    sess: TensorFlow session that was active when processor was created.

  Returns:
    List of sample data for the transformed samples, and list of labels in
    one-hot form.
  """
  # Pick one of the partitions to choose samples from.
  def get_data(self, qty, offset, model_settings, mode, sess):
    candidates, total = self.data_index[mode], self.set_size(mode)
    sample_count = total if qty < 1 else max(1, min(qty, total - offset))
    
    # Data augmentation algorithms
    variations_count = (len(model_settings['data_aug_algorithms']) + 1) if (mode == 'training') else 1

    # Data and scores will be populated and returned.
    data = np.zeros((sample_count*variations_count, model_settings['fingerprint_size']))
    scores = np.zeros((sample_count*variations_count, 1))

    # Use the processing graph created earlier to repeatedly to generate the
    # final output sample data we'll use in training.
    for i in xrange(offset, offset + sample_count):
      # Pick which audio sample to use.
      sample_index = np.random.randint(total) if (mode == 'training') else i
      original_sample = candidates[sample_index]

      input_dict = {
          self.wav_filename_placeholder_: original_sample['file'] 
      }

      # Run the graph to produce the original waveform.
      original_waveform = sess.run(self.waveforme, feed_dict=input_dict).flatten()
      original_score = original_sample['score']

      # generate data augmentation variations
      variations = (original_waveform,)
      for j in xrange(0,variations_count-1):
        variations += (apply_data_augmentation(original_waveform, model_settings['data_aug_algorithms'][j]),)

      # Run the graph to produce the output feature.
      for j in xrange(0,variations_count):
        input_dict = {
          self.waveforme_placeholder_: np.transpose([variations[j]])
        } 

        data[i + j - offset, :] = sess.run(self.feature, feed_dict=input_dict).flatten()
        scores[i + j - offset] = original_score

    return data, scores