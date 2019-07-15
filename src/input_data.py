"""Input data processing definitions for speech.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
from six.moves import xrange
from sklearn.model_selection import train_test_split

from src import utils
from src import data_augmentation

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops

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
    return sess.run(wav_decoder, feed_dict={wav_filename_placeholder: filename}).audio.flatten()

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

def set_partitions(samples, val_percentage, test_percentage):
  """Creates cross validation partitions from list of samples.

  Args:
    samples: List of samples represented as dictionaries.
    val_percentage: Validation set length in percentage.

  Returns:
    A dictionary with the partitions (training, validation and testing
    populated with samples.
  """
  data_index = {'training': [], 'validation': [], 'testing': []}
  data_index['training'], val_test_samples = train_test_split(
      samples, test_size=val_percentage + test_percentage)
  data_index['validation'], data_index['testing'] = train_test_split(
      val_test_samples, test_size=test_percentage)
  return data_index

class AudioProcessor:
    """Handles loading and preparing audio data with TensorFlow."""

    def __init__(self, params):
      self.prepare_data_info(params)
      self.prepare_processing_load_graph()
      self.prepare_processing_feature_graph()

    def prepare_data_info(self, params):
        self.fingerprint_size = params['fingerprint_size']
        self.n_coeffs = params['n_coeffs']
        self.desired_samples = params['desired_samples']
        self.window_size = params['window_size_samples']
        self.window_stride = params['window_stride_samples']
        self.sample_rate = params['sample_rate']

    def prepare_processing_load_graph(self):
        """Builds a TensorFlow graph to load wav files.

        Creates a graph that loads a WAVE file and decodes it.

        This must be called with an active TensorFlow session running, and it
        creates multiple placeholder inputs, and one output:

          - wav_filename_placeholder: Filename of the WAV to load.
          - waveform_placeholder: Waveform of the WAV to loaded.
        """
        with tf.name_scope('input'):
            self.wav_filename_placeholder = tf.placeholder(tf.string, [], 'file_name')
            self.waveform = contrib_audio.decode_wav(
                io_ops.read_file(self.wav_filename_placeholder),
                desired_channels=1,
                desired_samples=self.desired_samples
              ).audio

    def prepare_processing_feature_graph(self):
        """Builds a TensorFlow graph to generate features from waveforms.

        Creates a graph that generate spectrogram and tranform to choosed feature.

        This must be called with an active TensorFlow session running, and it
        creates multiple placeholder inputs, and one output:

          - waveform_placeholder: Waveforme loaded.
          - spectrogram: Generated Spectrogram.
          - feature: Calculated feature.
        """
        with tf.name_scope('feature'):
            self.waveform_placeholder = tf.placeholder(tf.float32, [self.desired_samples, 1], 'waveform')
            self.spectrogram = self.prepare_spectrogram_graph(self.waveform_placeholder)
            self.feature = self.prepare_feature_graph(self.spectrogram)

    def prepare_spectrogram_graph(self, waveform_placeholder):
        return contrib_audio.audio_spectrogram(
            waveform_placeholder, window_size=self.window_size, stride=self.window_stride)

    def prepare_feature_graph(self, spectrogram):
        abs_spec = tf.abs(spectrogram)
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            self.n_coeffs, abs_spec.shape[-1].value, self.sample_rate, 20.0, 4000.0)
        mel_spectrograms = tf.tensordot(abs_spec, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(   
            abs_spec.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)
        feat = tf.contrib.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
        return tf.transpose(feat, [0, 2, 1])

    def load(self, filename, sess):
        return sess.run(self.waveform, feed_dict={self.wav_filename_placeholder: filename})

    def gen_feature(self, waveform, sess):
        return sess.run(self.feature, feed_dict={self.waveform_placeholder: waveform}).flatten()


    def prepare_data_index(self, data_dir, extension, apply_shuffle=False):
        """Prepares a list of the samples.

        Args:
          data_dir: Directory with files.
          extension: Extension of files to looking for.
          apply_shuffle: Flag for enable/disable shuffle files.

        Returns:
          Dictionary containing a list of files.
        """
        # get all files
        file_names = utils.find_by_extension(data_dir, extension)
        self.data_index = [{'file': data_dir + "/" + name} for name in file_names]
        random.shuffle(self.data_index) if apply_shuffle else None

    def prepare_data_index_from_csv_file(self, csv_file, val_percentage, test_percentage):
        """Index data using csv files and applies partitions.

        Args:
          csv_file: CSV file with file names and scores.
          val_percentage: Validation set length in percentage.
          test_percentage: Testing set length in percentage.

        Returns:
          A dictionary with the partitions (training, validation and testing
          populated with samples.
        """
        try:
          samples = utils.read_csv_as_dict(csv_file, ',')
          samples = [{'file': s['file'], 'score': float(s['score'])} for s in samples]
          self.data_index = set_partitions(samples, val_percentage, test_percentage)
        except:
          print("Invalid CSV file, index not prepared!")

    def get_size(self):
        return len(self.data_index)

    def get_index(self):
        return self.data_index

    def get_size_by_partition(self, partition):
        return len(self.data_index[partition]) if partition in {'training', 'validation'} else -1

    def get_index_by_partition(self, partition):
        return self.data_index[partition] if partition in {'training', 'validation'} else -1        

    def get_data(self, qty, offset, sess):
        """Gather samples from the data set, applying transformations as needed.

        Args:
          qty: Desired number of samples to return. -1 means the entire contents of this partition.
          offset: Where to start when fetching deterministically.
          sess: TensorFlow session that was active when processor was created.

        Returns:
          List of sample data for the transformed samples, and list of labels in
          one-hot form.
        """
        # Pick one of the partitions to choose samples from.
        candidates, total = self.data_index, len(self.data_index)
        sample_count = total if qty < 1 else max(1, min(qty, total - offset))

        # Data and scores will be populated and returned.
        data = np.zeros((sample_count, self.fingerprint_size))
        names = ["" for x in range(sample_count)]

        # Use the processing graph created earlier to repeatedly to generate the
        # final output sample data.
        for i in xrange(offset, offset + sample_count):
            # load waveform
            original_waveform = self.load(candidates[i]['file'], sess)
            # Produce the output feature
            data[i - offset, :] = self.gen_feature(original_waveform, sess)
            names[i - offset] = candidates[i]['file']

        return names, data

    def get_data_by_partition(self, qty, offset, partition, sess):
        """Gather samples from the data set.

        When the mode is 'training', a random selection of samples will be returned,
        otherwise the first N clips in the partition will be used. This ensures that
        validation always uses the same samples, reducing noise in the metrics.

        Args:
          qty: Desired number of samples to return. -1 means the entire
            contents of this partition.
          offset: Where to start when fetching deterministically.
          params: Information about the current model being trained.
          partition: Which partition to use, must be 'training', 'validation'
          sess: TensorFlow session that was active when processor was created.

        Returns:
          List of sample data for the transformed samples, and list of labels in
          one-hot form.
        """
        # Pick one of the partitions to choose samples from.
        candidates, total = self.data_index[partition], self.get_size_by_partition(partition)
        if qty < 1:
          sample_count, offset = total, 0
        else:
          sample_count = max(1, min(qty, total - offset))

        # Data and scores will be populated and returned.
        data = np.zeros((sample_count, self.fingerprint_size))
        scores = np.zeros((sample_count, 1))
        names = ["" for x in range(sample_count)]

        # Use the processing graph created earlier to repeatedly to generate the
        # final output sample data we'll use in training.
        for i in xrange(offset, offset + sample_count):
            # Pick which audio sample to use.
            sample_index = i if (qty < 1 or partition != 'training') else np.random.randint(total)
            original_sample = candidates[sample_index]

            # load waveform
            original_waveform = self.load(original_sample['file'], sess)
            # gen feature
            data[i - offset, :] = self.gen_feature(original_waveform, sess)
            scores[i - offset] = original_sample['score']
            names[i - offset] = original_sample['file']

        return names, data, scores

