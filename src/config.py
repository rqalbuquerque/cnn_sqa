"""Config definitions for speech quality assessment.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

def set_flags(configs={}):
  parser = argparse.ArgumentParser()

# config DIRs
  parser.add_argument(
      '--data_dir',
      type=str,
      default=configs.get('data_dir', '../database/speech_dataset'),
      help=""" Where to download the speech training data to. """)
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default=configs.get('summaries_dir', '../result/retrain_logs'),
      help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
      '--train_dir',
      type=str,
      default=configs.get('train_dir', '../result/speech_quality_evaluation_train'),
      help='Directory to write event logs and checkpoint.')

# config Learning
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=configs.get('testing_percentage', 10),
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=configs.get('validation_percentage', 10),
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=configs.get('batch_size', 30),
      help='How many items to train with at once')
  parser.add_argument(
      '--training_steps',
      type=str,
      default=configs.get('training_steps', '5000,1000'),
      help='How many training loops to run')  
  parser.add_argument(
      '--learning_rate',
      type=str,
      default=configs.get('learning_rate', '0.01,0.001'),
      help='How large a learning rate to use when training.')

# config Signal
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=configs.get('sample_rate', 16000),
      help='Expected sample rate of the wavs')
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=configs.get('clip_duration_ms', 9000),
      help='Expected duration in milliseconds of the wavs')

# config Spectrogram
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=configs.get('window_size_ms', 32.0),
      help='How long each spectrogram timeslice is')
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=configs.get('window_stride_ms', 8.0),
      help='How long each spectrogram timeslice is')
  parser.add_argument(
      '--feature_used',
      type=str,
      default=configs.get('feature_used', 'mfcc'),
      help='How long each spectrogram timeslice is')
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=configs.get('dct_coefficient_count', 40),
      help='How many bins to use for the MFCC fingerprint')

# config CNN
  parser.add_argument(
      '--model_architecture',
      type=str,
      default=configs.get('model_architecture', 'conv2'),
      help='What model architecture to use')
  parser.add_argument(
      '--conv_layers',
      type=int,
      default=configs.get('conv_layers', 3),
      help='What model architecture to use')
  parser.add_argument(
      '--filter_width',
      type=int,
      default=configs.get('filter_width', 5),
      help='What model architecture to use')
  parser.add_argument(
      '--filter_count',
      type=int,
      default=configs.get('filter_count', 50),
      help='What model architecture to use')
  parser.add_argument(
      '--stride',
      type=int,
      default=configs.get('stride', 1),
      help='What model architecture to use')
  parser.add_argument(
      '--pooling',
      type=str,
      default=configs.get('pooling', 'avg'),
      help='Number of units in hidden layer 1.')
  
# config FC
  parser.add_argument(
      '--fc_layers',
      type=int,
      default=configs.get('fc_layers', 1),
      help='Number of units in hidden layer 1.')
  parser.add_argument(
      '--hidden_units',
      type=int,
      default=configs.get('hidden_units', 400),
      help='Number of units in hidden layer 1.')

  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=configs.get('eval_step_interval', 250),
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--save_step_interval',
      type=int,
      default=configs.get('save_step_interval', 1000),
      help='Save model checkpoint every save_steps.')
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default=configs.get('start_checkpoint', ''),
      help='If specified, restore this pretrained model before any training.')

# config nans checking
  parser.add_argument(
      '--check_nans',
      type=bool,
      default=configs.get('check_nans', False),
      help='Whether to check for invalid numbers during processing')

  return parser.parse_known_args()
