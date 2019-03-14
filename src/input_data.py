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
    self.feature_name = model_settings['feature']
    self.n_coeffs = model_settings['n_coeffs']
    self.n_frames = model_settings['n_frames']

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
    - waveform_placeholder: Waveform of the WAV to loaded.
    - feature: Calculated feature
  """
  def prepare_processing_input_graph(self, model_settings):
    with tf.name_scope('input'):
      self.wav_filename_placeholder = tf.placeholder(tf.string, [], 'file_name')
      self.waveform = self.prepare_load_wav_graph(self.wav_filename_placeholder, model_settings)

  def prepare_processing_feature_graph(self, model_settings):
    with tf.name_scope('feature'):
      self.waveform_placeholder = tf.placeholder(tf.float32, [model_settings['desired_samples'], 1], 'waveform')
      self.spectrogram = self.prepare_spectrogram_graph(self.waveform_placeholder, model_settings)
      self.feature = self.prepare_feature_graph(self.spectrogram, model_settings)

  def prepare_load_wav_graph(self, wav_filename_placeholder, model_settings):
    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = contrib_audio.decode_wav(
        wav_loader, 
        desired_channels=1, 
        desired_samples=model_settings['desired_samples'])
    return wav_decoder.audio

  def prepare_spectrogram_graph(self, waveform, model_settings):
    spectrogram = contrib_audio.audio_spectrogram(
        waveform, 
        window_size=model_settings['window_size_samples'], 
        stride=model_settings['window_stride_samples'], 
        magnitude_squared=True)
    return spectrogram

  def prepare_feature_graph(self, spectrogram, model_settings):
    if model_settings['feature'] == 'spectrogram':
      frames_count = model_settings['spectrogram_length']
      coefficient_count = model_settings['n_coeffs']
      feat = tf.slice(
          spectrogram, [0,0,0], [-1,frames_count,coefficient_count])
    elif model_settings['feature'] == 'mfcc': 
      feat = contrib_audio.mfcc(
          tf.abs(spectrogram),
          model_settings['sample_rate'],
          n_coeffs=model_settings['n_coeffs'])
    elif model_settings['feature'] == 'new_mfcc':
      abs_spec = tf.abs(spectrogram)
      linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        model_settings['n_coeffs'], abs_spec.shape[-1].value, model_settings['sample_rate'], 20.0, 4000.0)
      mel_spectrograms = tf.tensordot(
        abs_spec, linear_to_mel_weight_matrix, 1)
      mel_spectrograms.set_shape(abs_spec.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))
      log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)
      feat = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)
    return tf.transpose(feat, [0,2,1])

  """Load wav and generates features using Librosa.

  Directly load a .wav file and generates the choosed feature
  """
  def prepare_processing_input_librosa(self, model_settings):
    self.sr = model_settings['sample_rate']
    self.duration = model_settings['clip_duration_ms']/1000 #seconds

  def prepare_processing_feature_librosa(self, model_settings):
    self.hop_length = model_settings['window_stride_samples'] 
    self.n_fft = model_settings['window_size_samples'] 
    self.window = scipy.signal.windows.hann

  # Tensorflow func definitions
  def load_by_tensorflow(self, filename, sess):
    return sess.run(
      self.waveform, feed_dict={ self.wav_filename_placeholder: filename })

  def feature_by_tensorflow(self, waveform, sess):
    return sess.run(
      self.feature, feed_dict={ self.waveform_placeholder: waveform }).flatten()

  # Librosa func definitions
  def load_by_librosa(self, filename):
    with tf.name_scope('input'):
      data, _ = librosa.load(filename, sr=self.sr, duration=self.duration)
      return data

  def feature_by_librosa(self, data):
    with tf.name_scope('feature'):
      if self.feature_name == 'amplitude':
        feat = np.abs(librosa.stft(
          y=data, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window))
        return feat[0:self.n_coeffs,2:-2].flatten()
      elif self.feature_name == 'amplitude_to_db':
        feat = np.abs(librosa.stft(
          y=data, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window))
        feat = librosa.amplitude_to_db(feat, ref=np.max)
        return feat[0:self.n_coeffs,2:-2].flatten()
      elif self.feature_name == 'mel_spectrogram_power_1':
        feat = librosa.power_to_db(
          librosa.feature.melspectrogram(
            y=data, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.hop_length, fmax=8000, power=1), ref=np.max)
        return feat[0:self.n_coeffs,2:-2].flatten()
      elif self.feature_name == 'mel_spectrogram_power_2':
        feat = librosa.power_to_db(
          librosa.feature.melspectrogram(
            y=data, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.hop_length, fmax=8000, power=2), ref=np.max)
        return feat[0:self.n_coeffs,2:-2].flatten()
      elif self.feature_name == 'mfcc':
        feat = librosa.feature.mfcc(
          y=data, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft, n_mfcc=self.n_coeffs)
        return feat.flatten()

  def load_waveform(self, filename, lib, sess):
    if lib == 'librosa':
      return self.load_by_librosa(filename)
    elif lib == 'tensorflow':
      return self.load_by_tensorflow(filename, sess)

  def gen_feature(self, waveform, lib, sess):
    if lib == 'librosa':
      return self.feature_by_librosa(waveform)
    elif lib == 'tensorflow':
      return self.feature_by_tensorflow(waveform, sess)

  def load_feature_by_mat(self, fileName):
    mat_dict = scipy.io.loadmat(fileName)
    data = mat_dict[self.feature_name]
    data = data[0:self.n_coeffs, 0:self.n_frames]
    data = np.pad(data, ((0, self.n_coeffs - data.shape[0]),(0, self.n_frames - data.shape[1])), 'constant')
    return data.flatten()


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

  def get_data_by_wav(self, qty, offset, mode, sess):
    candidates, total = self.data_index[mode], self.set_size(mode)
    sample_count = total if qty < 1 else max(1, min(qty, total - offset))
    
    # Data augmentation algorithms
    variations_count = (len(self.data_aug_algorithms) + 1) if (mode == 'training') else 1

    # Data and scores will be populated and returned.
    data = np.zeros((sample_count*variations_count, self.fingerprint_size))
    scores = np.zeros((sample_count*variations_count, 1))
    names = ["" for x in range(sample_count*variations_count)]

    # Use the processing graph created earlier to repeatedly to generate the
    # final output sample data we'll use in training.
    for i in xrange(offset, offset + sample_count):
      # Pick which audio sample to use.
      sample_index = np.random.randint(total) if (mode == 'training') else i
      original_sample = candidates[sample_index]

      # load waveform
      original_waveform = self.load_waveform(original_sample['file'], self.input_processing_lib, sess)

      # Generates data augmentation variations
      variations = (original_waveform,)
      for j in xrange(0, variations_count-1):
        variations += (data_augmentation.apply(original_waveform, self.data_aug_algorithms[j]),)

      # Produce the output feature
      for j in xrange(0, variations_count):
        data[i + j - offset, :] = self.gen_feature(variations[j], self.input_processing_lib, sess)
        scores[i + j - offset] = original_sample['score']
        names[i + j - offset] = original_sample['file']

    return names, data, scores

  def get_data_by_path(self, filepath, sess):
    waveform = self.load_waveform(filepath, self.input_processing_lib, sess)
    return self.gen_feature(waveform, self.input_processing_lib, sess)

  def get_data_by_mat(self, qty, offset, mode):
    candidates, total = self.data_index[mode], self.set_size(mode)
    sample_count = total if qty < 1 else max(1, min(qty, total - offset))

    # Data and scores will be populated and returned.
    data = np.zeros((sample_count, self.fingerprint_size))
    scores = np.zeros((sample_count, 1))
    names = ["" for x in range(sample_count)]

    # Use the processing graph created earlier to repeatedly to generate the
    # final output sample data we'll use in training.
    for i in xrange(offset, offset + sample_count):
      # Pick which audio sample to use.
      sample_index = np.random.randint(total) if (mode == 'training') else i
      original_sample = candidates[sample_index]

      # load waveform
      data[i - offset, :] = self.load_feature_by_mat(original_sample['file'])
      scores[i - offset] = original_sample['score']
      names[i - offset] = original_sample['file']

    return names, data, scores

  def get_data(self, qty, offset, mode, sess):
    if self.input_processing_lib == 'scipy':
      return self.get_data_by_mat(qty, offset, mode)
    else:
      return self.get_data_by_wav(qty, offset, mode, sess)
