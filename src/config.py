"""Config definitions for speech quality assessment.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json

def save_configs(out_dir, flags):
  with open(out_dir + '/' + 'config.json', "w") as f:
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
      default=configs.get(
        'data_dir', 
        '/home/rqa/renato/git/A-Convolutional-Neural-Network-Approach-for-Speech-Quality-Assessment/database/matlab_features/vad'),
      help=""" Where to download the speech training data to. """)
  parser.add_argument(
      '--output_dir',
      type=str,
      default=configs.get(
        'output_dir', 
        '/home/rqa/renato/git/A-Convolutional-Neural-Network-Approach-for-Speech-Quality-Assessment/logs_test'),
      help='Where to save summary logs for TensorBoard.')
 
# config summary
  parser.add_argument(
      '--enable_hist_summary',
      type=str,
      default=configs.get('enable_hist_summary', False),
      help='Directory to write histogram summary.')

# config profile
  parser.add_argument(
      '--enable_profile',
      type=str,
      default=configs.get('enable_profile', False),
      help='Directory to write profile results.')

# checkpoint saving
  parser.add_argument(
      '--enable_checkpoint_save',
      type=bool,
      default=configs.get('enable_checkpoint_save', True),
      help='Flag to enable/disable checkpoint saving.')

# config Learning
  parser.add_argument(
      '--optimizer',
      type=str,
      default=configs.get('optimizer', 'gradient_descent'),
      help='What optimizer to use.')
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=configs.get('testing_percentage', 70),
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
      help='How many items to radomly select to apply mini-batch train')
  parser.add_argument(
      '--training_steps',
      type=str,
      default=configs.get('training_steps', [20,10,10,10,10]),
      help='How many training loops to run')  
  parser.add_argument(
      '--learning_rate',
      type=list,
      default=configs.get('learning_rate', [0.01,0.007,0.005,0.003,0.001]),
      help='How large a learning rate to use when training.')

# config Signal
  parser.add_argument(
      '--input_processing_lib',
      type=str,
      default=configs.get('input_processing_lib', 'scipy'),
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

# config Data Manipulation
  parser.add_argument(
      '--data_aug_algorithms',
      type=list,
      default=configs.get('data_aug_algorithms', []),
      help='Expected sample rate of the wavs')

# config Feature
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=configs.get('window_size_ms', 25.6),
      help='How long each spectrogram timeslice is')
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=configs.get('window_stride_ms', 10.0),
      help='How long each spectrogram timeslice is')
  parser.add_argument(
      '--feature',
      type=str,
      default=configs.get('feature', 'pncc_40_coefficients'),
      help='How feature use')
  parser.add_argument(
      '--n_coeffs',
      type=int,
      default=configs.get('n_coeffs', 40),
      help='How many bins to use for the feature fingerprint')

# config CNN
  parser.add_argument(
      '--model_architecture',
      type=str,
      default=configs.get('model_architecture', 'slim_conv'),
      help='What model architecture to use')
  parser.add_argument(
      '--conv_layers',
      type=int,
      default=configs.get('conv_layers', 5),
      help='How many convolutional layers to use')
  parser.add_argument(
      '--filter_width',
      type=list,
      default=configs.get('filter_width', [2,3,5,7,9]),
      help='What filter width to use')
  parser.add_argument(
      '--filter_height',
      type=list,
      default=configs.get('filter_height', [2,3,5,7,9]),
      help='What filter height to use')
  parser.add_argument(
      '--filter_count',
      type=list,
      default=configs.get('filter_count', [5,5,5,5,5]),
      help='What filter count to use')
  parser.add_argument(
      '--stride',
      type=list,
      default=configs.get('stride', [1,2,2,2,2]),
      help='What long stride to use')
  parser.add_argument(
      '--apply_batch_norm',
      type=bool,
      default=configs.get('apply_batch_norm', True),
      help='Decide to apply batch normalization')
  parser.add_argument(
      '--apply_dropout',
      type=bool,
      default=configs.get('apply_dropout', True),
      help='Decide to apply dropout')
  parser.add_argument(
      '--activation',
      type=str,
      default=configs.get('activation', 'relu'),
      help='What activation function type to use.')
  parser.add_argument(
      '--apply_regularization',
      type=bool,
      default=configs.get('apply_regularization', False),
      help='Decide to apply dropout')
  parser.add_argument(
      '--kernel_regularizer',
      type=str,
      default=configs.get('kernel_regularizer', ''),
      help='What kernel regularizer function to use')
  
# config FC
  parser.add_argument(
      '--fc_layers',
      type=int,
      default=configs.get('fc_layers', 2),
      help='Number of fully connected layers to use.')
  parser.add_argument(
      '--hidden_units',
      type=str,
      default=configs.get('hidden_units', [25,25]),
      help='Number of units in hidden layers.')

  parser.add_argument(
      '--summary_step_interval',
      type=int,
      default=configs.get('summary_step_interval', 5),
      help='How often to summary the training results.')
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=configs.get('eval_step_interval', 10),
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default=configs.get('start_checkpoint', ''),
      help='If specified, restore this pretrained model before any training.')

# testing
  parser.add_argument(
      '--evaluate_testing',
      type=str,
      default=configs.get('evaluate_testing', False),
      help='Apply or not testing step.')

# config nans checking
  parser.add_argument(
      '--check_nans',
      type=bool,
      default=configs.get('check_nans', False),
      help='Whether to check for invalid numbers during processing')

  return parser.parse_known_args()
