"""Input data setting definitions for speech quality assessment.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import math
import os.path
import random
import re
import sys
import tarfile
import sys
import glob

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import librosa
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
RANDOM_SEED = 59185


def which_set(filename, validation_percentage, testing_percentage):
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result


def load_wav_file(filename):
  """Loads an audio file and returns a float PCM-encoded array of samples.

  Args:
    filename: Path to the .wav file to load.

  Returns:
    Numpy array holding the sample data as floats between -1.0 and 1.0.
  """
  with tf.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
    return sess.run(
        wav_decoder,
        feed_dict={wav_filename_placeholder: filename}).audio.flatten()


def save_wav_file(filename, wav_data, sample_rate):
  """Saves audio sample data to a .wav audio file.

  Args:
    filename: Path to save the file to.
    wav_data: 2D array of float PCM-encoded audio data.
    sample_rate: Samples per second to encode in the file.
  """
  with tf.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.placeholder(tf.string, [])
    sample_rate_placeholder = tf.placeholder(tf.int32, [])
    wav_data_placeholder = tf.placeholder(tf.float32, [None, 1])
    wav_encoder = contrib_audio.encode_wav(wav_data_placeholder,
                                           sample_rate_placeholder)
    wav_saver = io_ops.write_file(wav_filename_placeholder, wav_encoder)
    sess.run(
        wav_saver,
        feed_dict={
            wav_filename_placeholder: filename,
            sample_rate_placeholder: sample_rate,
            wav_data_placeholder: np.reshape(wav_data, (-1, 1))
        })

def save_spectrogram_images(inputdir, outputdir):
  if os.path.exists(inputdir):
    if not os.path.exists(outputdir):
      os.makedirs(outputdir)

    with tf.Session() as sess:
      for filename in glob.glob(os.path.join(inputdir, '*.wav')):
        print(filename)
        name =  os.path.splitext(os.path.basename(filename))[0]
        # Run the computation graph and save the png encoded image to a file
        wav_file, graph = spectrogram_image_graph()
        image = sess.run(graph, feed_dict={wav_file: filename})

        with open(outputdir + "/" + name + ".png", 'wb') as f:
            f.write(image)

def spectrogram_image_graph():
    # Wav file name
  wav_file = tf.placeholder(tf.string)

  # Read the wav file
  wav_loader = io_ops.read_file(wav_file)

  # Decode the wav mono into a 2D tensor with time in dimension 0
  # and channel along dimension 1
  waveform = contrib_audio.decode_wav(wav_loader, desired_channels=1, desired_samples=(9*16000))

  # Compute the spectrogram
  spectrogram = contrib_audio.audio_spectrogram(
          waveform.audio,
          window_size=1024,
          stride=128)

  feature = contrib_audio.mfcc(
      spectrogram,
      waveform.sample_rate,
      dct_coefficient_count=40)

  # Custom brightness
  mul = tf.multiply(feature, 100)

  # Normalize pixels
  min_const = tf.constant(255.)
  minimum =  tf.minimum(mul, min_const)

  # Expand dims so we get the proper shape
  expand_dims = tf.expand_dims(minimum, -1)

  # Resize the spectrogram to input size of the model
  resize = tf.image.resize_bilinear(expand_dims, [1024, 1024])

  # Remove the trailing dimension
  squeeze = tf.squeeze(resize, 0)

  # Tensorflow spectrogram has time along y axis and frequencies along x axis
  # so we fix that
  flip = tf.image.flip_left_right(squeeze)
  transpose = tf.image.transpose_image(flip)

  # Convert image to 3 channels, it's still a grayscale image however
  grayscale = tf.image.grayscale_to_rgb(transpose)

  # Cast to uint8 and encode as png
  cast = tf.cast(grayscale, tf.uint8)
  return wav_file, tf.image.encode_png(cast)


class AudioProcessor(object):
  """Handles loading, partitioning, and preparing audio training data."""

  #def __init__(self, data_url, data_dir, validation_percentage, testing_percentage, model_settings):
  def __init__(self, data_dir, validation_percentage, testing_percentage, feature_type, model_settings):
    self.data_dir = data_dir
    # self.maybe_download_and_extract_dataset(data_url, data_dir)
    self.prepare_data_index(validation_percentage, testing_percentage)
    self.prepare_processing_graph(model_settings)

  def maybe_download_and_extract_dataset(self, data_url, dest_directory):
    """Download and extract data set tar file.

    If the data set we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a
    directory.
    If the data_url is none, don't download anything and expect the data
    directory to contain the correct files already.

    Args:
      data_url: Web location of the tar file containing the data set.
      dest_directory: File path to extract data to.
    """
    if not data_url:
      return
    if not os.path.exists(dest_directory):
      os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

      def _progress(count, block_size, total_size):
        sys.stdout.write(
            '\r>> Downloading %s %.1f%%' %
            (filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

      try:
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
      except:
        tf.logging.error('Failed to download URL: %s to folder: %s', data_url,
                         filepath)
        tf.logging.error('Please make sure you have enough free space and'
                         ' an internet connection')
        raise
      print()
      statinfo = os.stat(filepath)
      tf.logging.info('Successfully downloaded %s (%d bytes)', filename,
                      statinfo.st_size)

    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
   

  def prepare_data_index(self, validation_percentage, testing_percentage):
    """Prepares a list of the samples organized by set.

    The training loop needs a list of all the available data.
    This function analyzes the folders below the `data_dir`.
    For each sub-folder is necessary the set of wav files and one
    file with scores called 'scores.txt' with format.:

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
      Dictionary containing a list of file.

    Raises:
      Exception: If expected files are not found.
    """

    # Make sure the shuffling.
    self.data_index = {'validation': [], 'testing': [], 'training': []}
    # Get all subfolders
    database = [x[0] for x in os.walk(self.data_dir)]
    if len(database) > 1:
      del database[0]
    # Look through all the subfolders to find audio samples
    for folder in database:
      score_paths = os.path.join(folder, '*scores*.txt')
      score_files = gfile.Glob(score_paths)
      # Interates over file to get file names and score
      with open(score_files[0]) as file:
        for line in file.readlines():
          info = line.split(' ')
          wav_path = folder + '/' + info[0]
          score = float(info[1])
          set_index = which_set(wav_path, validation_percentage, testing_percentage)
          self.data_index[set_index].append({'score': score, 'file': wav_path})
    # Make sure the ordering is random.
    for set_index in ['validation', 'testing', 'training']:
      random.shuffle(self.data_index[set_index])

  def get_set_sizes(self):
    return len(self.data_index['training']), len(self.data_index['validation']), len(self.data_index['testing'])

  def get_testing_indexes(self):
    return self.data_index['testing']

  def prepare_processing_graph(self, model_settings):
    """Builds a TensorFlow graph to apply the input distortions.

    Creates a graph that loads a WAVE file, decodes it, shifts it in time, 
    lculates a spectrogram, and then builds an MFCC fingerprint from that.

    This must be called with an active TensorFlow session running, and it
    creates multiple placeholder inputs, and one output:

      - wav_filename_placeholder_: Filename of the WAV to load.
      - time_shift_padding_placeholder_: Where to pad the clip.
      - background_data_placeholder_: PCM sample data for background noise.
      - mfcc_: Output 2D fingerprint of processed audio.

    Args:
      model_settings: Information about the current model being trained.
    """

    desired_samples = model_settings['desired_samples']
    self.wav_filename_placeholder_ = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
    wav_decoder = contrib_audio.decode_wav(
        wav_loader, desired_channels=1, desired_samples=desired_samples)

    self.spectrogram = contrib_audio.audio_spectrogram(
        wav_decoder.audio,
        window_size=model_settings['window_size_samples'],
        stride=model_settings['window_stride_samples'],
        magnitude_squared=True)

    self.feature = []
    if model_settings['feature_used'] == 'spectrogram':
      coeffic_count = int(model_settings['dct_coefficient_count']/4)
      frames_count = self.spectrogram.shape[1]
      self.feature = tf.slice(self.spectrogram,
                              [0,0,0],
                              [-1,frames_count,coeffic_count])
    elif model_settings['feature_used'] == 'mfcc':
      frames_count = self.spectrogram.shape[1]
      coeffic_count = 
      feature = tf.slice(self.spectrogram,
                              [0,0,0],
                              [-1,frames_count,coeffic_count])

      self.feature = contrib_audio.mfcc(
          self.spectrogram,
          wav_decoder.sample_rate,
          dct_coefficient_count=model_settings['dct_coefficient_count'])
      
  def set_size(self, mode):
    """Calculates the number of samples in the dataset partition.

    Args:
      mode: Which partition, must be 'training', 'validation', or 'testing'.

    Returns:
      Number of samples in the partition.
    """
    return len(self.data_index[mode])

  # def get_data(self, how_many, offset, model_settings, time_shift, mode, sess):
  def get_data(self, how_many, offset, model_settings, mode, sess):
    """Gather samples from the data set, applying transformations as needed.

    When the mode is 'training', a random selection of samples will be returned,
    otherwise the first N clips in the partition will be used. This ensures that
    validation always uses the same samples, reducing noise in the metrics.

    Args:
      how_many: Desired number of samples to return. -1 means the entire
        contents of this partition.
      offset: Where to start when fetching deterministically.
      model_settings: Information about the current model being trained.
      time_shift: How much to randomly shift the clips by in time.
      mode: Which partition to use, must be 'training', 'validation', or
        'testing'.
      sess: TensorFlow session that was active when processor was created.

    Returns:
      List of sample data for the transformed samples, and list of labels in
      one-hot form.
    """
    # Pick one of the partitions to choose samples from.
    candidates = self.data_index[mode]
    if how_many == -1:
      sample_count = len(candidates)
    else:
      sample_count = max(0, min(how_many, len(candidates) - offset))
    
    # Data and scores will be populated and returned.
    data = np.zeros((sample_count, model_settings['fingerprint_size']))
    scores = np.zeros((sample_count, 1))
    desired_samples = model_settings['desired_samples']
    pick_deterministically = (mode != 'training')

    # Use the processing graph we created earlier to repeatedly to generate the
    # final output sample data we'll use in training.
    for i in xrange(offset, offset + sample_count):
      # Pick which audio sample to use.
      if how_many == -1 or pick_deterministically:
        sample_index = i
      else:
        sample_index = np.random.randint(len(candidates))
      sample = candidates[sample_index]
     
      # If we're time shifting, set up the offset for this sample.
      # if time_shift > 0:
      #   time_shift_amount = np.random.randint(-time_shift, time_shift)
      # else:
      #   time_shift_amount = 0
      # if time_shift_amount > 0:
      #   time_shift_padding = [[time_shift_amount, 0], [0, 0]]
      #   time_shift_offset = [0, 0]
      # else:
      #   time_shift_padding = [[0, -time_shift_amount], [0, 0]]
      #   time_shift_offset = [-time_shift_amount, 0]
      # input_dict = {
      #     self.wav_filename_placeholder_: sample['file'],
      #     self.time_shift_padding_placeholder_: time_shift_padding,
      #     self.time_shift_offset_placeholder_: time_shift_offset,
      # }

      input_dict = {
          self.wav_filename_placeholder_: sample['file']
      }

      # Run the graph to produce the output audio.
      data[i - offset, :] = sess.run(self.feature, feed_dict=input_dict).flatten()
      scores[i - offset] = sample['score']
    return data, scores

