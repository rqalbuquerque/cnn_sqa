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

import librosa
import scipy
import numpy as np
from six.moves import xrange

import data_augmentation

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

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


"""Handles loading, partitioning, and preparing audio training data."""
class AudioProcessor(object):

  def __init__(self, data_dir, validation_percentage, testing_percentage, model_settings):
    self.prepare_data_index(data_dir, validation_percentage, testing_percentage)
    self.prepare_processing_input(model_settings)
    self.prepare_data_info(model_settings)
          
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

  def prepare_data_info(self, model_settings):
    self.input_processing_lib = model_settings['input_processing_lib']
    self.fingerprint_size = model_settings['fingerprint_size']
    self.data_aug_algorithms = model_settings['data_aug_algorithms']

  def prepare_processing_input(self, model_settings):
    if model_settings['input_processing_lib'] == 'tensorflow':
      self.prepare_processing_input_graph(model_settings)
      self.prepare_processing_feature_graph(model_settings)
    elif model_settings['input_processing_lib'] == 'librosa':
      self.prepare_processing_input_librosa(model_settings)
      self.prepare_processing_feature_librosa(model_settings)

  """Builds a TensorFlow graph to apply the input distortions.

  Creates a graph that loads a WAVE file, decodes it, calculates a spectrogram, 
  and then builds an feature selected fingerprint from that.

  This must be called with an active TensorFlow session running, and it
  creates multiple placeholder inputs, and one output:

    - wav_filename_placeholder: Filename of the WAV to load.
    - waveforme_placeholder: Waveform of the WAV to loaded.
    - feature: Calculated feature
  """
  def prepare_processing_input_graph(self, model_settings):
    with tf.name_scope('input'):
      self.wav_filename_placeholder = tf.placeholder(tf.string, [], 'file_name')
      self.waveforme = self.prepare_load_wav_graph(self.wav_filename_placeholder, model_settings)

  def prepare_processing_feature_graph(self, model_settings):
    with tf.name_scope('feature'):
      self.waveforme_placeholder = tf.placeholder(tf.float32, [model_settings['desired_samples'], 1], 'waveform')
      self.spectrogram = self.prepare_spectrogram_graph(self.waveforme_placeholder, model_settings)
      self.feature = self.prepare_feature_graph(self.spectrogram, model_settings)

  def prepare_load_wav_graph(self, wav_filename_placeholder, model_settings):
    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = contrib_audio.decode_wav(
        wav_loader, 
        desired_channels=1, 
        desired_samples=model_settings['desired_samples'])
    return wav_decoder.audio

  def prepare_spectrogram_graph(self, waveforme, model_settings):
    spectrogram = contrib_audio.audio_spectrogram(
        waveforme, 
        window_size=model_settings['window_size_samples'], 
        stride=model_settings['window_stride_samples'], 
        magnitude_squared=False)
    return spectrogram

  def prepare_feature_graph(self, spectrogram, model_settings):
    if model_settings['feature'] == 'spectrogram':
      frames_count = model_settings['spectrogram_length']
      coefficient_count = model_settings['dct_coefficient_count']
      feature = tf.slice(
          spectrogram, [0,0,0], [-1,frames_count,coefficient_count])
    elif model_settings['feature'] == 'mfcc': 
      feature = contrib_audio.mfcc(
          tf.abs(spectrogram),
          model_settings['sample_rate'],
          dct_coefficient_count=model_settings['dct_coefficient_count'])
    return tf.transpose(feature, [0,2,1])

  """Load wav and generates features using Librosa.

  Directly load a .wav file and generates the choosed feature
  """
  def prepare_processing_input_librosa(self, model_settings):
    self.sr = model_settings['sample_rate']
    self.duration = model_settings['desired_samples']/self.sr

  def prepare_processing_feature_librosa(self, model_settings):
    self.feature = model_settings['feature']
    self.hop_length = model_settings['window_stride_samples'] 
    self.n_fft = model_settings['window_size_samples'] 
    self.n_coeffs = model_settings['dct_coefficient_count']
    self.window = scipy.signal.windows.hann

  # Tensorflow func definitions
  def load_by_tensorflow(self, filename, sess):
    return sess.run(
      self.waveforme, feed_dict={ self.wav_filename_placeholder: filename })

  def feature_by_tensorflow(self, waveforme, sess):
    return sess.run(
      self.feature, feed_dict={ self.waveforme_placeholder: waveforme }).flatten()

  # Librosa func definitions
  def load_by_librosa(self, filename):
    with tf.name_scope('input'):
      data, sr = librosa.load(filename, sr=self.sr, duration=self.duration)
      return data

  def feature_by_librosa(self, data):
    with tf.name_scope('feature'):
      if self.feature == 'amplitude':
        feat = np.abs(librosa.stft(
          y=data, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window))
        return feat[0:self.n_coeffs,2:-2].flatten()
      elif self.feature == 'amplitude_to_db':
        feat = np.abs(librosa.stft(
          y=data, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window))
        feat = librosa.amplitude_to_db(feat, ref=np.max)
        return feat[0:self.n_coeffs,2:-2].flatten()
      elif self.feature == 'mel_spectrogram_power_1':
        feat = librosa.power_to_db(
          librosa.feature.melspectrogram(
            y=data, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.hop_length, fmax=8000, power=1), ref=np.max)
        return feat[0:self.n_coeffs,2:-2].flatten()
      elif self.feature == 'mel_spectrogram_power_2':
        feat = librosa.power_to_db(
          librosa.feature.melspectrogram(
            y=data, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.hop_length, fmax=8000, power=2), ref=np.max)
        return feat[0:self.n_coeffs,2:-2].flatten()
      elif self.feature == 'mfcc':
        feat = librosa.feature.mfcc(
          y=data, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft, n_mfcc=self.n_coeffs)
        return feat.flatten()

  def load_waveform(self, filename, lib, sess):
    waveforme = []

    if lib == 'librosa':
      waveforme = self.load_by_librosa(filename)
    elif lib == 'tensorflow':
      waveforme = self.load_by_tensorflow(filename, sess)

    return waveforme

  def gen_feature(self, waveforme, lib, sess):
    feature = []

    if lib == 'librosa':
      feature = self.feature_by_librosa(waveforme)
    elif lib == 'tensorflow':
      feature = self.feature_by_tensorflow(waveforme, sess)

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

  def get_data(self, qty, offset, mode, sess):
    candidates, total = self.data_index[mode], self.set_size(mode)
    sample_count = total if qty < 1 else max(1, min(qty, total - offset))
    
    # Data augmentation algorithms
    variations_count = (len(self.data_aug_algorithms) + 1) if (mode == 'training') else 1

    # Data and scores will be populated and returned.
    data = np.zeros((sample_count*variations_count, self.fingerprint_size))
    scores = np.zeros((sample_count*variations_count, 1))

    # Use the processing graph created earlier to repeatedly to generate the
    # final output sample data we'll use in training.
    for i in xrange(offset, offset + sample_count):
      # Pick which audio sample to use.
      sample_index = np.random.randint(total) if (mode == 'training') else i
      original_sample = candidates[sample_index]

      # load waveform
      original_waveform = self.load_waveform(original_sample['file'], self.input_processing_lib, sess)
      original_score = original_sample['score']

      # Generates data augmentation variations
      variations = (original_waveform,)
      for j in xrange(0, variations_count-1):
        variations += (data_augmentation.apply(original_waveform, self.data_aug_algorithms[j]),)

      # Produce the output feature
      for j in xrange(0, variations_count):
        data[i + j - offset, :] = self.gen_feature(variations[j], self.input_processing_lib, sess)
        scores[i + j - offset] = original_score

    return data, scores