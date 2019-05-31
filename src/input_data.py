"""Input data processing definitions for speech.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import os.path
import random

import librosa
import scipy
import numpy as np
from abc import ABCMeta, abstractmethod
from six.moves import xrange

import utils
import data_augmentation

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M


def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.

    Args:
      filename: File path of the data sample.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    hash_name_hashed = hashlib.sha1(compat.as_bytes(base_name)).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) % (
        MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS))

    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'

    return result


def cross_val_partitions(samples, validation_percentage, testing_percentage):
    """Creates cross validation partitions from list of samples.

    Args:
      samples: List of samples represented as dictionaries.
      validation_percentage: Validation set length in percentage.
      testing_percentage: Testing set length in percentage.

    Returns:
      A dictionary with the partitions (training, validation and testing
      populated with samples.
    """
    data_index = {'validation': [], 'testing': [], 'training': []}
    # index
    for sample in samples:
        index = which_set(
            sample['name'], validation_percentage, testing_percentage)
        data_index[index].append(sample)
    # shuffle
    for set_index in ['validation', 'testing', 'training']:
        random.shuffle(data_index[set_index])
    return data_index


class AudioProcessor:
    """Abstract calss to handle preparing audio data.

    This is the base class to create audio processors with data 
    pre-indexed. This class is based on four stages to get audio
    data: index -> load -> feature -> get

    To create a new audio processor is necessary override two methods:
      load() -> used to load files
      gen_feature -> used to generate features

    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.data_index = []

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def gen_feature(self):
        pass

    def index_from_dir(self, data_dir, extension, apply_shuffle=False):
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

        self.data_index = []
        for file_name in file_names:
            self.data_index.append({'file': file_name})

        random.shuffle(self.data_index) if apply_shuffle else self.data_index

    def index_from_csv_dir(self, csv_dir, validation_percentage, testing_percentage):
        """Index data using csv files and applies partitions.

        Args:
          csv_dir: Directory with csv files describing data attributes.
          validation_percentage: Validation set length in percentage.
          testing_percentage: Testing set length in percentage.

        Returns:
          A dictionary with the partitions (training, validation and testing
          populated with samples.
        """
        score_paths = utils.find_by_extension(csv_dir, "scores.csv")
        samples = reduce((lambda x, y: utils.read_csv_as_dict(
            x, ',') + utils.read_csv_as_dict(y, ',')), score_paths)
        self.data_index = cross_val_partitions(
            samples, validation_percentage, testing_percentage)

    def index_from_txt_dir(self, scores_dir, validation_percentage, testing_percentage):
        """Prepares a list of samples organized by set.

        This function analyzes the folders below the `scores_dir`.
        For each sub-folder is necessary a set of `data_ext` files and one
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
        # Make sure the shuffling.
        self.data_index = {'validation': [], 'testing': [], 'training': []}
        # get all score files
        score_paths = utils.find_by_extension(scores_dir, "scores.txt")
        # set index by file nameprepare_processing_feature_graph
        for score_path in score_paths:
            prefix = os.path.dirname(score_path)
            with open(score_path) as file:
                for line in file.readlines():
                    info = line.split(' ')
                    wav_path = prefix + '/' + info[0]
                    score = float(info[1])
                    set_index = which_set(
                        wav_path, validation_percentage, testing_percentage)
                    self.data_index[set_index].append(
                        {'score': score, 'file': wav_path})
        # shuffle
        for set_index in ['validation', 'testing', 'training']:
            random.shuffle(self.data_index[set_index])

    def get_partition_size(self, partition):
        return len(self.data_index[partition]) if partition in {'training', 'validation', 'testing'} else -1
    def get_size(self):
        return len(self.data_index)

    def get_data(self, qty, offset):
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
            original_waveform = self.load(candidates[i]['file'])
            # Produce the output feature
            data[i - offset, :] = self.gen_feature(original_waveform)
            names[i - offset] = candidates[i]['file']

        return names, data

    def get_data_by_partition(self, qty, offset, partition):
        """Gather samples from the data set, applying transformations as needed.

        When the mode is 'training', a random selection of samples will be returned,
        otherwise the first N clips in the partition will be used. This ensures that
        validation always uses the same samples, reducing noise in the metrics.

        Args:
          qty: Desired number of samples to return. -1 means the entire
            contents of this partition.
          offset: Where to start when fetching deterministically.
          params: Information about the current model being trained.
          partition: Which partition to use, must be 'training', 'validation', or
            'testing'.
          sess: TensorFlow session that was active when processor was created.

        Returns:
          List of sample data for the transformed samples, and list of labels in
          one-hot form.
        """
        # Pick one of the partitions to choose samples from.
        candidates, total = self.data_index[partition], self.get_partition_size(
            partition)
        sample_count = total if qty < 1 else max(1, min(qty, total - offset))

        # Data augmentation algorithms
        variations_count = (len(self.data_aug_algorithms) +
                            1) if (partition == 'training') else 1

        # Data and scores will be populated and returned.
        data = np.zeros((sample_count*variations_count, self.fingerprint_size))
        scores = np.zeros((sample_count*variations_count, 1))
        names = ["" for x in range(sample_count*variations_count)]

        # Use the processing graph created earlier to repeatedly to generate the
        # final output sample data we'll use in training.
        for i in xrange(offset, offset + sample_count):
            # Pick which audio sample to use.
            sample_index = np.random.randint(total) if (
                partition == 'training') else i
            original_sample = candidates[sample_index]

            # load waveform
            original_waveform = self.load(original_sample['file'])

            # Generates data augmentation variations
            variations = (original_waveform,)
            for j in xrange(0, variations_count-1):
                variations += (data_augmentation.apply(original_waveform,
                                                       self.data_aug_algorithms[j]),)

            # Produce the output feature
            for j in xrange(0, variations_count):
                data[i + j - offset, :] = self.gen_feature(variations[j])
                scores[i + j - offset] = original_sample['score']
                names[i + j - offset] = original_sample['file']

        return names, data, scores


class TFAudioProcessor(AudioProcessor):
    """Handles loading and preparing audio data with TensorFlow."""

    def __init__(self, params, sess):
        super(TFAudioProcessor, self).__init__()
        self.prepare_data_info(params, sess)
        self.prepare_processing_load_graph()
        self.prepare_processing_feature_graph()

    def prepare_data_info(self, params, sess):
        self.fingerprint_size = params['fingerprint_size']
        self.feature_name = params['feature']
        self.n_coeffs = params['n_coeffs']
        self.desired_samples = params['desired_samples']
        self.window_size = params['window_size_samples']
        self.window_stride = params['window_stride_samples']
        self.sample_rate = params['sample_rate']
        self.data_aug_algorithms = params['data_aug_algorithms']
        self.n_frames = params['n_frames']
        self.sess = sess

    def prepare_processing_load_graph(self):
        """Builds a TensorFlow graph to load wav files.

        Creates a graph that loads a WAVE file and decodes it.

        This must be called with an active TensorFlow session running, and it
        creates multiple placeholder inputs, and one output:

          - wav_filename_placeholder: Filename of the WAV to load.
          - waveform_placeholder: Waveform of the WAV to loaded.
        """
        with tf.name_scope('input'):
            self.wav_filename_placeholder = tf.placeholder(
                tf.string, [], 'file_name')
            self.waveform = self.prepare_load_wav_graph(
                self.wav_filename_placeholder)

    def prepare_load_wav_graph(self, wav_filename_placeholder):
        wav_loader = io_ops.read_file(wav_filename_placeholder)
        wav_decoder = contrib_audio.decode_wav(
            wav_loader,
            desired_channels=1,
            desired_samples=self.desired_samples)
        return wav_decoder.audio

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
            self.waveform_placeholder = tf.placeholder(
                tf.float32, [self.desired_samples, 1], 'waveform')
            self.spectrogram = self.prepare_spectrogram_graph(
                self.waveform_placeholder)
            self.feature = self.prepare_feature_graph(self.spectrogram)

    def prepare_spectrogram_graph(self, waveform):
        spectrogram = contrib_audio.audio_spectrogram(
            waveform,
            window_size=self.window_size,
            stride=self.window_stride,
            magnitude_squared=True)
        return spectrogram

    def prepare_feature_graph(self, spectrogram):
        abs_spec = tf.abs(spectrogram)
        if self.feature_name == 'spectrogram':
            feat = tf.slice(abs_spec, [0, 0, 0],
                            [-1, self.n_frames, self.n_coeffs])
        elif self.feature_name == 'new_mfcc':
            linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
                self.n_coeffs, abs_spec.shape[-1].value, self.sample_rate, 20.0, 4000.0)
            mel_spectrograms = tf.tensordot(
                abs_spec, linear_to_mel_weight_matrix, 1)
            mel_spectrograms.set_shape(   
                abs_spec.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
            log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)
            feat = tf.contrib.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
        return tf.transpose(feat, [0, 2, 1])

    def load(self, filename):
        return self.sess.run(self.waveform, feed_dict={self.wav_filename_placeholder: filename})

    def gen_feature(self, waveform):
        return self.sess.run(self.feature, feed_dict={self.waveform_placeholder: waveform}).flatten()


class LibrosaAudioProcessor(AudioProcessor):
    """Handles loading and preparing audio data with Librosa."""

    def __init__(self, params):
        super().__init__()
        self.prepare_data_info(params)

    def prepare_data_info(self, params):
        self.fingerprint_size = params['fingerprint_size']
        self.feature_name = params['feature']
        self.n_coeffs = params['n_coeffs']
        self.desired_samples = params['desired_samples']
        self.spectrogram_length = params['spectrogram_length']
        self.sample_rate = params['sample_rate']
        self.duration = model_params['clip_duration_ms']/1000  # seconds
        self.hop_length = model_params['window_stride_samples']
        self.n_fft = model_params['window_size_samples']
        self.window = scipy.signal.windows.hann

    def load(self, filename):
        data, _ = librosa.load(
            filename, sr=self.sample_rate, duration=self.duration)
        return data

    def gen_feature(self, data):
        feat = []
        with tf.name_scope('feature'):
            if self.feature_name == 'amplitude':
                feat = np.abs(librosa.stft(y=data, n_fft=self.n_fft,
                                           hop_length=self.hop_length, window=self.window))
                feat = feat[0:self.n_coeffs, 2:-2].flatten()
            elif self.feature_name == 'amplitude_to_db':
                feat = np.abs(librosa.stft(y=data, n_fft=self.n_fft,
                                           hop_length=self.hop_length, window=self.window))
                feat = librosa.amplitude_to_db(feat, ref=np.max)
                feat = feat[0:self.n_coeffs, 2:-2].flatten()
            elif self.feature_name == 'mel_spectrogram_power_1':
                feat = librosa.power_to_db(
                    librosa.feature.melspectrogram(
                        y=data, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.hop_length, fmax=8000, power=1), ref=np.max)
                feat = feat[0:self.n_coeffs, 2:-2].flatten()
            elif self.feature_name == 'mel_spectrogram_power_2':
                feat = librosa.power_to_db(
                    librosa.feature.melspectrogram(
                        y=data, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.hop_length, fmax=8000, power=2), ref=np.max)
                feat = feat[0:self.n_coeffs, 2:-2].flatten()
            elif self.feature_name == 'mfcc':
                feat = librosa.feature.mfcc(
                    y=data, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft, n_mfcc=self.n_coeffs)
                feat = feat.flatten()
        return feat


class ScipyAudioProcessor(AudioProcessor):
    """Handles loading and preparing audio data with Scipy."""

    def prepare_data_info(self, params):
        self.feature_name = params['feature']

    def load(self, fileName):
        return scipy.io.loadmat(fileName)

    def gen_feature(self, mat_dict):
        data = mat_dict[self.feature_name]
        data = data[0:self.n_coeffs, 0:self.n_frames]
        data = np.pad(data, ((
            0, self.n_coeffs - data.shape[0]), (0, self.n_frames - data.shape[1])), 'constant')
        return data.flatten()
