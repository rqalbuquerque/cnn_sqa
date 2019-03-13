"""Model definitions for speech quality assessment.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from six.moves import xrange
from functools import partial

"""Calculates common settings needed for all models.

Args:
  sample_rate: Number of audio samples per second.
  clip_duration_ms: Length of each audio clip to be analyzed.
  window_size_ms: Duration of frequency analysis window.
  window_stride_ms: How far to move in time between frequency windows.
  n_coeffs: Number of frequency bins to use for analysis.

Returns:
  Dictionary containing common settings.
"""
def prepare_model_settings(enable_hist_summary,
                           input_processing_lib,
                           sample_rate, 
                           clip_duration_ms,
                           window_size_ms, 
                           window_stride_ms,
                           data_aug_algorithms,
                           feature,
                           n_coeffs,
                           conv_layers,
                           filter_width,
                           filter_height,
                           filter_count,
                           stride,
                           apply_batch_norm,
                           activation,
                           kernel_regularizer,
                           apply_dropout,
                           fc_layers,
                           hidden_units):

  desired_samples = int(sample_rate * clip_duration_ms / 1000.0)
  window_size_samples = int(sample_rate * window_size_ms / 1000.0)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000.0)
  n_frames = 1 + int((desired_samples-window_size_samples) / window_stride_samples)
  fingerprint_size = n_coeffs * n_frames
  
  return {
      'enable_hist_summary': enable_hist_summary,
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'fingerprint_size': fingerprint_size,
      'input_processing_lib': input_processing_lib,
      'sample_rate': sample_rate,
      'n_frames': n_frames,
      'data_aug_algorithms': data_aug_algorithms,
      'feature': feature,
      'n_coeffs': n_coeffs,
      'conv_layers': conv_layers,
      'filter_width': filter_width,
      'filter_height': filter_height,
      'filter_count': filter_count,
      'stride': stride,
      'apply_dropout': apply_dropout,
      'apply_batch_norm': apply_batch_norm,
      'activation': activation,
      'kernel_regularizer': kernel_regularizer,
      'fc_layers': fc_layers,
      'hidden_units': hidden_units
  }

"""Utility function to centralize checkpoint restoration.

Args:
  sess: TensorFlow session.
  start_checkpoint: Path to saved checkpoint on disk.
"""
def load_variables_from_checkpoint(sess, start_checkpoint):
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)

def apply_activation(input_tensor, mode):
  if mode == "softplus":
    return tf.nn.softplus(input_tensor, name="softplus")
  elif mode == "leaky_relu":
    return tf.nn.leaky_relu(input_tensor, alpha=0.2, name="leaky_relu")
  elif mode == "elu":
    return tf.nn.elu(input_tensor, name="elu")
  else:
    return tf.nn.relu(input_tensor, name="relu")

def get_activation_func(mode):
  if mode == "softplus":
    return tf.nn.softplus 
  return tf.nn.relu

def get_kernel_regularizer(mode, scale=0.0001):
  if mode == "l1":
    return tf.contrib.layers.l1_regularizer(scale)
  elif mode == "l2":
    return tf.contrib.layers.l2_regularizer(scale)
  elif mode == "l1_l2":
    return tf.contrib.layers.l1_l2_regularizer(scale)
  return None

"""Utility function to add pooling to the graph.

Args:
  input_tensor: input tensor graph.
  type:         pooling type.
  ksize:        window size.
  strides:      stride size.
  padding:      padding algorithm.

Returns:
  TensorFlow node outputting pooling results.
"""
def apply_pooling(input_tensor, mode, ksize, strides, padding):
  with tf.name_scope('pooling'):
    return tf.nn.max_pool(input_tensor, ksize, strides, padding, name="max_pool") if mode == "max" else input_tensor
    return tf.nn.avg_pool(input_tensor, ksize, strides, padding, name="avg_pool") if mode == "avg" else input_tensor

"""
Batch normalization on convolutional maps.
Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
Args:
    input_tensor: Tensor, 4D BHWD input maps
    n_out:        integer, depth of input maps
    phase_train:  boolean tf.Varialbe, true indicates training phase
    scope:        string, variable scope
Return:
    Batch-normalized maps
"""
def batch_normalization(input_tensor, n_out, phase_train):
  with tf.name_scope('batch_norm'):
    beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(input_tensor, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    
    return tf.nn.batch_normalization(input_tensor, mean, var, beta, gamma, 1e-3)

def conv_2d(input_tensor, 
               filter_height, 
               filter_width, 
               filters_depth, 
               stride,
               output_maps_count,
               enable_hist_summary):
  with tf.name_scope('conv'):
    weights = tf.Variable(
      tf.truncated_normal(
        [
          filter_height, 
          filter_width, 
          filters_depth, 
          output_maps_count
        ],
        stddev=0.01,
        name='random'),
      name='weights')
    bias = tf.Variable(tf.zeros([output_maps_count]), name='biases')
    conv = tf.math.add(tf.nn.conv2d(input_tensor, weights, [1, stride, stride, 1], 'SAME'), bias, name='sum')

    if enable_hist_summary:
      tf.summary.histogram('weights', weights)
      tf.summary.histogram('bias', bias)

    return conv

def fully_connected(input_tensor, 
                    n_inputs, 
                    units, 
                    enable_hist_summary):
  weights = tf.Variable(
    tf.truncated_normal(
      [
        n_inputs, 
        units
      ], 
      stddev=0.01,
      name='random'), 
    name='weights')
  bias = tf.Variable(tf.zeros([units]), name='biases')
  fc = tf.math.add(tf.matmul(input_tensor, weights, name='mult'), bias, name='sum')

  if enable_hist_summary:
    tf.summary.histogram('weights', weights)
    tf.summary.histogram('bias', bias)

  return fc



"""
Fully-connected layer.
Args:
    input_tensor: Tensor, 4D BHWD input maps
    n_inputs: integer, number of input
    units: integer, number of units
Return:
    output matrix
"""
def fc_layer(input_tensor, 
              n_inputs, 
              units, 
              activation_func,
              enable_hist_summary):
  with tf.name_scope('fc'):
    fc = fully_connected(input_tensor, n_inputs, units, enable_hist_summary)
    actv = apply_activation(fc, activation_func)
    return actv


def regression_layer(input_tensor,
                      n_inputs,
                      enable_hist_summary):
  with tf.name_scope('fc'):
    return fully_connected(input_tensor, n_inputs, 1, enable_hist_summary)

"""Builds a model of the requested architecture compatible with the settings.

There are many possible ways of deriving predictions from a spectrogram
input, so this function provides an abstract interface for creating different
kinds of models in a black-box way. You need to pass in a TensorFlow node as
the 'fingerprint' input, and this should output a batch of 1D features that
describe the audio. Typically this will be derived from a spectrogram that's
been run through an MFCC, but in theory it can be any feature vector of the
size specified in params['fingerprint_size'].

The function will build the graph it needs in the current TensorFlow graph,
and return the tensorflow output that will contain the 'logits' input to the
softmax prediction process. If training flag is on, it will also return a
placeholder node that can be used to control the dropout amount.

Args:
  fingerprint_input: TensorFlow node that will output audio feature vectors.
  params: Dictionary of information about the model.
  model_architecture: String specifying which kind of model to create.
  runtime_settings: Dictionary of information about the runtime.

Returns:
  TensorFlow node outputting logits results, and optionally a dropout
  placeholder.

Raises:
  Exception: If the architecture type isn't recognized.
"""
def create_model(fingerprint_input, 
                 params, 
                 model_architecture,
                 runtime_settings=None):

  if model_architecture == 'conv':
    return create_conv_model(fingerprint_input, params)
  elif model_architecture == 'slim_conv':
    return create_slim_conv_model(fingerprint_input, params)
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be "conv"')

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
    [Pooling]
        v
  [FullConected] (1 or 2 layers)

Args:
  fingerprint_input: TensorFlow node that will output audio feature vectors.
  params: Dictionary of information about the model.

Returns:
  TensorFlow node outputting logits results, and optionally a dropout
  placeholder.
"""
def create_conv_model(fingerprint_input, params):
  fingerprint = tf.reshape(fingerprint_input, 
    [-1, params['n_coeffs'], params['n_frames'], 1])
  phase_train = tf.placeholder(tf.bool, name='phase_train')

  conv_layer = partial(conv_2d, enable_hist_summary=params['enable_hist_summary'])
  batch_norm_layer = partial(batch_normalization, phase_train=phase_train)
  activation_layer = partial(apply_activation, mode=params['activation'])
  dropout_layer = partial(tf.layers.dropout, rate=0.5, training=phase_train, name='dropout')

  conv = conv_layer(
    fingerprint, params['filter_height'][0], params['filter_width'][0], 
    fingerprint.shape[-1].value, params['stride'][0], params['filter_count'][0])
  batch_norm = batch_norm_layer(conv, params['filter_count'][0]) if params['apply_batch_norm'] else conv
  activation = activation_layer(batch_norm)
  dropout = dropout_layer(activation) if params['apply_dropout'] else activation

  conv_1 = conv_layer(
    dropout, params['filter_height'][1], params['filter_width'][1], 
    dropout.shape[-1].value, params['stride'][1], params['filter_count'][1])
  batch_norm_1 = batch_norm_layer(conv_1, params['filter_count'][1]) if params['apply_batch_norm'] else conv_1
  activation_1 = activation_layer(batch_norm_1)
  dropout_1 = dropout_layer(activation_1) if params['apply_dropout'] else activation_1

  conv_2 = conv_layer(
    dropout_1, params['filter_height'][2], params['filter_width'][2], 
    dropout_1.shape[-1].value, params['stride'][2], params['filter_count'][2])
  batch_norm_2 = batch_norm_layer(conv_2, params['filter_count'][2]) if params['apply_batch_norm'] else conv_2
  activation_2 = activation_layer(batch_norm_2)
  dropout_2 = dropout_layer(activation_2) if params['apply_dropout'] else activation_2

  conv_3 = conv_layer(
    dropout_2, params['filter_height'][3], params['filter_width'][3], 
    dropout_2.shape[-1].value, params['stride'][3], params['filter_count'][3])
  batch_norm_3 = batch_norm_layer(conv_3, params['filter_count'][3]) if params['apply_batch_norm'] else conv_3
  activation_3 = activation_layer(batch_norm_3)
  dropout_3 = dropout_layer(activation_3) if params['apply_dropout'] else activation_3

  conv_4 = conv_layer(
    dropout_3, params['filter_height'][4], params['filter_width'][4], 
    dropout_3.shape[-1].value, params['stride'][4], params['filter_count'][4])
  batch_norm_4 = batch_norm_layer(conv_4, params['filter_count'][4]) if params['apply_batch_norm'] else conv_4
  activation_4 = activation_layer(batch_norm_4)
  dropout_4 = dropout_layer(activation_4) if params['apply_dropout'] else activation_4

  # flattened 
  [_, output_height, output_width, output_depth] = output_conv.get_shape()
  element_count = int(output_height * output_width * output_depth)
  flattened = tf.reshape(output_conv, [-1, element_count])

  dense_layer = partial(
    tf.layers.dense, 
    activation=get_activation_func(params['activation']),
    kernel_regularizer=get_kernel_regularizer(params['kernel_regularizer']))

  fc1 = dense_layer(flattened, params['hidden_units'][0], name='dense1')
  fc2 = dense_layer(fc1, params['hidden_units'][1], name='dense2')

  # regression 
  estimator = tf.layers.dense(fc2, 1, name='estimator')
  
  # log
  tf.summary.image('input', fingerprint, 1)

  return estimator, phase_train


def create_slim_conv_model(fingerprint_input, params):
  print(params['n_coeffs'])
  print(params['n_frames'])
  fingerprint = tf.reshape(fingerprint_input, 
    [-1, params['n_coeffs'], params['n_frames'], 1])
  phase_train = tf.placeholder(tf.bool, name='phase_train')

  conv_layer = partial(conv_2d, enable_hist_summary=params['enable_hist_summary'])
  batch_norm_layer = partial(batch_normalization, phase_train=phase_train)
  activation_layer = partial(apply_activation, mode=params['activation'])
  dropout_layer = partial(tf.layers.dropout, rate=0.5, training=phase_train, name='dropout')

  convs = ()

  output_conv = fingerprint
  for i in range(0, params['conv_layers']):
    convs += (conv_layer(
      output_conv, params['filter_height'][i], params['filter_width'][i], 
      output_conv.shape[-1].value, params['stride'][i], params['filter_count'][i]),)
    batch_norm = batch_norm_layer(convs[i], params['filter_count'][i]) if params['apply_batch_norm'] else convs[i]
    activation = activation_layer(batch_norm)
    output_conv = dropout_layer(activation) if params['apply_dropout'] else activation

  for op in convs:
    tf.add_to_collection("conv_ops", op)

  # flattened 
  [_, output_height, output_width, output_depth] = output_conv.get_shape()
  element_count = int(output_height * output_width * output_depth)
  flattened = tf.reshape(output_conv, [-1, element_count])

  dense_layer = partial(
    tf.layers.dense, 
    activation=get_activation_func(params['activation']),
    kernel_regularizer=get_kernel_regularizer(params['kernel_regularizer']))

  fc1 = dense_layer(flattened, params['hidden_units'][0], name='dense1')
  fc2 = dense_layer(fc1, params['hidden_units'][1], name='dense2')

  # regression 
  estimator = tf.layers.dense(fc2, 1, name='estimator')
  
  # log
  tf.summary.image('input', fingerprint, 1)

  return estimator, phase_train
