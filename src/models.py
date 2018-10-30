"""Model definitions for speech quality assessment.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf


def prepare_model_settings(sample_rate, 
                           clip_duration_ms,
                           window_size_ms, 
                           window_stride_ms,
                           feature_used,
                           dct_coefficient_count,
                           conv_layers,
                           filter_width,
                           filter_count,
                           stride,
                           pooling,
                           fc_layers,
                           hidden_units):

  """Calculates common settings needed for all models.

  Args:
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.

  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)

  fingerprint_size = dct_coefficient_count * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'fingerprint_size': fingerprint_size,
      'sample_rate': sample_rate,
      'spectrogram_length': spectrogram_length,
      'feature_used': feature_used,
      'dct_coefficient_count': dct_coefficient_count,
      'conv_layers': conv_layers,
      'filter_width': filter_width,
      'filter_count': filter_count,
      'stride': stride,
      'pooling': pooling,
      'fc_layers': fc_layers,
      'hidden_units': hidden_units
  }


def create_model(fingerprint_input, model_settings, model_architecture,
                 is_training, runtime_settings=None):
  """Builds a model of the requested architecture compatible with the settings.

  There are many possible ways of deriving predictions from a spectrogram
  input, so this function provides an abstract interface for creating different
  kinds of models in a black-box way. You need to pass in a TensorFlow node as
  the 'fingerprint' input, and this should output a batch of 1D features that
  describe the audio. Typically this will be derived from a spectrogram that's
  been run through an MFCC, but in theory it can be any feature vector of the
  size specified in model_settings['fingerprint_size'].

  The function will build the graph it needs in the current TensorFlow graph,
  and return the tensorflow output that will contain the 'logits' input to the
  softmax prediction process. If training flag is on, it will also return a
  placeholder node that can be used to control the dropout amount.

  See the implementations below for the possible model architectures that can be
  requested.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    model_architecture: String specifying which kind of model to create.
    is_training: Whether the model is going to be used for training.
    runtime_settings: Dictionary of information about the runtime.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.

  Raises:
    Exception: If the architecture type isn't recognized.
  """

  if model_architecture == 'conv':
    return create_conv_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'conv2':
    return create_conv2_model(fingerprint_input, model_settings)
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be "conv"')


def load_variables_from_checkpoint(sess, start_checkpoint):
  """Utility function to centralize checkpoint restoration.

  Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
  """
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)


def apply_pooling(inputTensor, type, ksize, strides, padding, filter_count):

  if type == 'max':
    pooling = tf.nn.max_pool(inputTensor, ksize, strides, padding)
  elif type == 'avg':
    pooling = tf.nn.avg_pool(inputTensor, ksize, strides, padding)

  conv_shape = pooling.get_shape()
  conv_output_width = conv_shape[2]
  conv_output_height = conv_shape[1]
  conv_element_count = int(conv_output_width * conv_output_height * filter_count)
  flattened = tf.reshape(pooling,[-1, conv_element_count])
  return conv_element_count, flattened


def create_conv_model(fingerprint_input, model_settings, is_training):
  """Builds a standard convolutional model.

  This is roughly, with some different parameters, the network in the:
  'Convolutional Neural Networks for No-Reference Image Quality Assessment' paper:
  http://ieeexplore.ieee.org/document/6909620/

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v    
      [MaxPool]
          v
    [FullConected] (1 or 2 layers)


  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  hidden_units = model_settings['hidden_units']
  stride = model_settings['stride']

  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  
  first_filter_width = model_settings['filter_width']
  first_filter_height = model_settings['filter_width']
  first_filter_count = model_settings['filter_count']

  # conv
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, stride, stride, 1],
                            'SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu

  # if model_settings['conv_layers'] == 2:
    # second conv
  second_filter_width = max(int(first_filter_width/2),5)
  second_filter_height = max(int(first_filter_height/2),5)
  second_filter_count = first_filter_count
  second_weights = tf.Variable(
      tf.truncated_normal(
          [
              second_filter_height, second_filter_width, first_filter_count,
              second_filter_count
          ],
          stddev=0.01))
  second_bias = tf.Variable(tf.zeros([second_filter_count]))
  second_conv = tf.nn.conv2d(first_dropout, second_weights, [1, stride, stride, 1],
                            'SAME') + second_bias
  second_relu = tf.nn.relu(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, dropout_prob)
  else:
    second_dropout = second_relu

  # if model_settings['conv_layers'] == 3:
    # third conv
  third_filter_width = max(int(first_filter_width/3),3)
  third_filter_height = max(int(first_filter_height/3),3)
  third_filter_count = first_filter_count
  third_weights = tf.Variable(
      tf.truncated_normal(
          [
              third_filter_height, third_filter_height, second_filter_count,
              third_filter_count
          ],
          stddev=0.01))
  third_bias = tf.Variable(tf.zeros([second_filter_count]))
  third_conv = tf.nn.conv2d(second_dropout, third_weights, [1, stride, stride, 1],
                            'SAME') + third_bias
  third_relu = tf.nn.relu(second_conv)
  if is_training:
    third_dropout = tf.nn.dropout(third_relu, dropout_prob)
  else:
    third_dropout = third_relu

  # pooling
  if model_settings['pooling'] == 'max':
    pooling = tf.nn.max_pool(third_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  elif model_settings['pooling'] == 'avg':
    pooling = tf.nn.avg_pool(third_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

  pooling_shape = pooling.get_shape()
  pooling_output_width = pooling_shape[2]
  pooling_output_height = pooling_shape[1]
  pooling_element_count = int(
      pooling_output_width * pooling_output_height * third_filter_count)
  flattened_pooling = tf.reshape(pooling,[-1, pooling_element_count])

  # first full-connected layer
  first_fc_weights = tf.Variable(
      tf.truncated_normal(
          [pooling_element_count, hidden_units], stddev=0.01), name='weights')
  first_fc_bias = tf.Variable(tf.zeros([hidden_units]), name='biases')
  final_fc = tf.nn.relu(tf.matmul(flattened_pooling, first_fc_weights) + first_fc_bias)

  # regression 
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [hidden_units, 1], stddev=0.01), name='weights')
  final_fc_bias = tf.Variable(tf.zeros([1]), name='biases')
  estimator = tf.matmul(final_fc, final_fc_weights) + final_fc_bias

  if is_training:
    return estimator, dropout_prob
  else:
    return estimator



def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
      beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                   name='beta', trainable=True)
      gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                    name='gamma', trainable=True)
      batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
      ema = tf.train.ExponentialMovingAverage(decay=0.5)

      def mean_var_with_update():
          ema_apply_op = ema.apply([batch_mean, batch_var])
          with tf.control_dependencies([ema_apply_op]):
              return tf.identity(batch_mean), tf.identity(batch_var)

      mean, var = tf.cond(phase_train,
                          mean_var_with_update,
                          lambda: (ema.average(batch_mean), ema.average(batch_var)))
      normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed




def create_conv2_model(fingerprint_input, model_settings):
  """Builds a standard convolutional model.

  This is roughly, with some different parameters, the network in the:
  'Convolutional Neural Networks for No-Reference Image Quality Assessment' paper:
  http://ieeexplore.ieee.org/document/6909620/

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
  [BatchNormaliztion]
          v
        [Relu]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
  [BatchNormaliztion]
          v
        [Relu]
          v    
      [MaxPool/AvgPool]
          v
    [FullConected] (1 or 2 layers)

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  phase_train = tf.placeholder(tf.bool, name='phase_train')

  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])

  hidden_units = model_settings['hidden_units']
  stride = model_settings['stride']

  # first conv
  first_filter_width = model_settings['filter_width']
  first_filter_height = model_settings['filter_width']
  first_filter_count = model_settings['filter_count']

  first_weights = tf.Variable(
      tf.truncated_normal(
          [
              first_filter_height, first_filter_width, 1, first_filter_count
          ],
          stddev=0.01))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, stride, stride, 1],
                            'SAME') + first_bias
  first_norm_conv = batch_norm(first_conv, first_filter_count, phase_train)
  final_relu = tf.nn.relu(first_norm_conv)

  if model_settings['conv_layers'] == 2:

    # second conv
    second_filter_width = max(int(first_filter_width/2),3)
    second_filter_height = max(int(first_filter_height/2),3)
    second_filter_count = first_filter_count
    second_weights = tf.Variable(
        tf.truncated_normal(
            [
                second_filter_height, second_filter_width, first_filter_count,
                second_filter_count
            ],
            stddev=0.01))
    second_bias = tf.Variable(tf.zeros([second_filter_count]))
    second_conv = tf.nn.conv2d(final_relu, second_weights, [1, stride, stride, 1],
                              'SAME') + second_bias
    second_norm_conv = batch_norm(second_conv, second_filter_count, phase_train)
    final_relu = tf.nn.relu(second_norm_conv)

  if model_settings['conv_layers'] == 3:

    # third conv
    third_filter_width = max(int(first_filter_width/3),2)
    third_filter_height = max(int(first_filter_height/3),2)
    third_filter_count = first_filter_count
    third_weights = tf.Variable(
        tf.truncated_normal(
            [
                third_filter_height, third_filter_width, first_filter_count,
                third_filter_count
            ],
            stddev=0.01))
    third_bias = tf.Variable(tf.zeros([third_filter_count]))
    third_conv = tf.nn.conv2d(final_relu, third_weights, [1, stride, stride, 1],
                            'SAME') + third_bias
    third_norm_conv = batch_norm(third_conv, third_filter_count, phase_train)
    final_relu = tf.nn.relu(third_norm_conv)

  # pooling
  flattened_conv = []
  if model_settings['pooling'] == 'max':
    conv_element_count, flattened_conv = apply_pooling(final_relu, 'max', [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', first_filter_count)
  elif model_settings['pooling'] == 'avg':
    conv_element_count, flattened_conv = apply_pooling(final_relu, 'avg', [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', first_filter_count)
  elif model_settings['pooling'] == 'max,avg' or model_settings['pooling'] == 'avg,max':
    element_count_max, flattened_max = apply_pooling(final_relu, 'max', [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', first_filter_count)
    element_count_avg, flattened_avg = apply_pooling(final_relu, 'avg', [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', first_filter_count)
    conv_element_count = element_count_max + element_count_avg
    flattened_conv = tf.concat([flattened_max, flattened_avg], 1)

  # first full-connected layer
  first_fc_weights = tf.Variable(
      tf.truncated_normal(
          [conv_element_count, hidden_units], stddev=0.01), name='weights')
  first_fc_bias = tf.Variable(tf.zeros([hidden_units]), name='biases')
  final_fc = tf.nn.relu(tf.matmul(flattened_conv, first_fc_weights) + first_fc_bias)

  # second full-connected layer
  if model_settings['fc_layers'] == 2:
    second_fc_weights = tf.Variable(
        tf.truncated_normal(
            [hidden_units, hidden_units], stddev=0.01), name='weights')
    second_fc_bias = tf.Variable(tf.zeros([hidden_units]), name='biases')
    final_fc = tf.nn.relu(tf.matmul(final_fc, second_fc_weights) + second_fc_bias)

  # regression 
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [hidden_units, 1], stddev=0.01), name='weights')
  final_fc_bias = tf.Variable(tf.zeros([1]), name='biases')
  estimator = tf.matmul(final_fc, final_fc_weights) + final_fc_bias

  return estimator, phase_train

