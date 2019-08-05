"""Config definitions for speech quality assessment.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json

def save(out_dir, flags):
    with open(out_dir + '/' + 'config.json', "w") as f:
        json.dump(flags, f, indent=2)

def read(path):
    with open(path) as f:
        return json.load(f)

def set(configs={}):
    parser = argparse.ArgumentParser(description='Process model args.')

    # config DIRs
    parser.add_argument(
        '--data_dir',
        type=str,
        default=configs.get(
            'data_dir',
            '/home/rqa/renato/git/cnn_sqa/databases/speech_noise_dataset_suppl23/suppl23_mix_speech_noise/'),
        help='Where to get the speech training data.')
    parser.add_argument(
        '--data_file',
        type=str,
        default=configs.get(
            'data_file',
            '/home/rqa/renato/git/cnn_sqa/databases/speech_noise_dataset_suppl23/suppl23_mix_speech_noise/scores.csv'),
        help='Where to read data samples.')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=configs.get(
            'output_dir',
            '/home/rqa/renato/git/cnn_sqa/logs/test'),
        help='Where to save summary logs for TensorBoard.')

    # checkpoint saving
    parser.add_argument(
        '--enable_checkpoint_save',
        type=bool,
        default=configs.get('enable_checkpoint_save', True),
        help='Flag to enable/disable checkpoint saving.')

    # config Learning
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=configs.get('validation_percentage', 0.5),
        help='What percentage of wavs to use as a validation set.')
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=configs.get('testing_percentage', 95),
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=configs.get('batch_size', 5),
        help='How many items to radomly select to apply mini-batch train')
    parser.add_argument(
        '--training_steps',
        type=str,
        default=configs.get('training_steps', [50, 10, 10, 10, 10]),
        help='How many training loops to run')
    parser.add_argument(
        '--learning_rate',
        type=list,
        default=configs.get(
            'learning_rate', [0.01, 0.007, 0.005, 0.003, 0.001]),
        help='How large a learning rate to use when training.')

    # config wav samples
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
        '--data_aug_columns',
        type=list,
        default=configs.get('data_aug_columns', ['data_aug_1', 'data_aug_2']),
        help='Data augmentation columns to load')

    # config Feature
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
        default=configs.get('feature', 'new_mfcc'),
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
        '--filter_width',
        type=list,
        default=configs.get('filter_width', [2, 3, 5, 7, 9]),
        help='What filter width to use')
    parser.add_argument(
        '--filter_height',
        type=list,
        default=configs.get('filter_height', [2, 3, 5, 7, 9]),
        help='What filter height to use')
    parser.add_argument(
        '--filter_count',
        type=list,
        default=configs.get('filter_count', [5, 5, 5, 5, 5]),
        help='What filter count to use')
    parser.add_argument(
        '--stride',
        type=list,
        default=configs.get('stride', [1, 2, 2, 2, 2]),
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

    # config FC
    parser.add_argument(
        '--hidden_units',
        type=list,
        default=configs.get('hidden_units', [25, 25]),
        help='Number of units in hidden layers.')

    # config summary and checkpoint load
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
        help='If specified, restore pretrained model before any training.')

    return parser.parse_known_args()
