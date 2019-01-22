"""Config definitions for speech quality assessment.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json

def save_configs(flags, out_dir):
  with open(out_dir + '/' + 'configs.json', "w") as f:
    json.dump(flags, f, indent=2)

def read_config(config_path): 
  with open(config_path) as f:
    return json.load(f)

def set_flags(configs={}):
  parser = argparse.ArgumentParser(description='Process model args.')

# config DIRs
  parser.add_argument(
      '--data_dir',
      type=str,
      default=configs.get('data_dir', '../database/speech_dataset_test'),
      help=""" Where to download the speech training data to. """)
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default=configs.get('summaries_dir', '../logs_test/summary'),
      help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
      '--train_dir',
      type=str,
      default=configs.get('train_dir', '../logs_test/event'),
      help='Directory to write event logs and checkpoint.')

# config summary
  parser.add_argument(
      '--enable_hist_summary',
      type=str,
      default=configs.get('enable_hist_summary', False),
      help='Directory to write event logs and checkpoint.')

# config profile
  parser.add_argument(
      '--enable_profile',
      type=str,
      default=configs.get('enable_profile', False),
      help='Directory to write event logs and checkpoint.')

# config Learning
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=configs.get('testing_percentage', 5),
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=configs.get('validation_percentage', 5),
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=configs.get('batch_size', 5),
      help='How many items to train with at once')
  parser.add_argument(
      '--training_steps',
      type=str,
      default=configs.get('training_steps', [30,5]),
      help='How many training loops to run')  
  parser.add_argument(
      '--learning_rate',
      type=list,
      default=configs.get('learning_rate', [0.01,0.001]),
      help='How large a learning rate to use when training.')

# config Signal
  parser.add_argument(
      '--input_processing_lib',
      type=str,
      default=configs.get('input_processing_lib', 'tensorflow'),
      help='Processing library of audio samples')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=configs.get('sample_rate', 16000),
      help='Expected sample rate of the wavs')
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=configs.get('clip_duration_ms', 8000),
      help='Expected duration in milliseconds of the wavs')

# config Data Augmentation
  parser.add_argument(
      '--data_aug_algorithms',
      type=list,
      default=configs.get('data_aug_algorithms', []),
      help='Expected sample rate of the wavs')

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
      '--feature',
      type=str,
      default=configs.get('feature', 'mfcc'),
      help='How feature use')
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=configs.get('dct_coefficient_count', 40),
      help='How many bins to use for the feature fingerprint')

# config CNN
  parser.add_argument(
      '--model_architecture',
      type=str,
      default=configs.get('model_architecture', 'conv'),
      help='What model architecture to use')
  parser.add_argument(
      '--conv_layers',
      type=int,
      default=configs.get('conv_layers', 3),
      help='How many convolutional layers to use')
  parser.add_argument(
      '--filter_width',
      type=list,
      default=configs.get('filter_width', [3,5,7]),
      help='What filter width to use')
  parser.add_argument(
      '--filter_height',
      type=list,
      default=configs.get('filter_height', [3,5,7]),
      help='What filter height to use')
  parser.add_argument(
      '--filter_count',
      type=list,
      default=configs.get('filter_count', [3,3,3]),
      help='What filter count to use')
  parser.add_argument(
      '--stride',
      type=list,
      default=configs.get('stride', [2,3,4]),
      help='What long stride to use')
  parser.add_argument(
      '--apply_batch_norm',
      type=bool,
      default=configs.get('apply_batch_norm', True),
      help='Decide to apply batch normalization')
  parser.add_argument(
      '--pooling',
      type=str,
      default=configs.get('pooling', ''),
      help='What pooling type to use.')
  
# config FC
  parser.add_argument(
      '--fc_layers',
      type=int,
      default=configs.get('fc_layers', 1),
      help='Number of fully connected layers to use.')
  parser.add_argument(
      '--hidden_units',
      type=str,
      default=configs.get('hidden_units', [50]),
      help='Number of units in hidden layers.')

  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=configs.get('eval_step_interval', 2),
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--save_step_interval',
      type=int,
      default=configs.get('save_step_interval', 2000),
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
