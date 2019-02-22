"""Model definitions for speech quality assessment.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from six.moves import xrange

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
def prepare_model_settings(enable_hist_summary,
                           input_processing_lib,
                           sample_rate, 
                           clip_duration_ms,
                           window_size_ms, 
                           window_stride_ms,
                           data_aug_algorithms,
                           feature,
                           dct_coefficient_count,
                           conv_layers,
                           filter_width,
                           filter_height,
                           filter_count,
                           stride,
                           apply_batch_norm,
                           activation,
                           pooling,
                           fc_layers,
                           hidden_units):

  desired_samples = int(sample_rate * clip_duration_ms / 1000.0)
  window_size_samples = int(sample_rate * window_size_ms / 1000.0)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000.0)
  spectrogram_length = 1 + int((desired_samples-window_size_samples) / window_stride_samples)
  fingerprint_size = dct_coefficient_count * spectrogram_length
  
  return {
      'enable_hist_summary': enable_hist_summary,
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'fingerprint_size': fingerprint_size,
      'input_processing_lib': input_processing_lib,
      'sample_rate': sample_rate,
      'spectrogram_length': spectrogram_length,
      'data_aug_algorithms': data_aug_algorithms,
      'feature': feature,
      'dct_coefficient_count': dct_coefficient_count,
      'conv_layers': conv_layers,
      'filter_width': filter_width,
      'filter_height': filter_height,
      'filter_count': filter_count,
      'stride': stride,
      'pooling': pooling,
      'apply_batch_norm': apply_batch_norm,
      'activation': activation,
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

def activation(input_tensor, mode):
  if mode == "softplus":
    return tf.nn.softplus(input_tensor, name="softplus")
  if mode == "leaky_relu":
    return tf.nn.leaky_relu(input_tensor, alpha=0.2, name="leaky_relu")
  if mode == "elu":
    return tf.nn.elu(input_tensor, name="elu")
  else:
    return tf.nn.relu(input_tensor, name="relu")

def get_activation_func(mode):
  if mode == "softplus":
    return tf.nn.softplus 
  return tf.nn.relu


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
def x_pooling(input_tensor, mode, ksize, strides, padding):
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

def convolution(input_tensor, 
               filter_height, 
               filter_width, 
               filters_depth, 
               stride,
               output_maps_count,
               enable_hist_summary):
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
Convolutional layer.
Args:
    input_tensor: Tensor, 4D BHWD input maps
    filter_height: integer, height of filter
    filter_width: integer, width of filter
    filters_depth: integer, depth of filter
    stride: integer, stride of filter
    output_maps_count: integer, number of feature maps
Return:
    feature maps volume tensor
"""

def conv_layer(input_tensor, 
               filter_height, 
               filter_width, 
               filters_depth, 
               stride,
               output_maps_count,
               apply_batch_norm,
               phase_train,
               activation_func,
               enable_hist_summary):
  with tf.name_scope('conv'):
    conv = convolution(
      input_tensor, filter_height, filter_width, filters_depth, stride, output_maps_count, enable_hist_summary)
    batch_norm = batch_normalization(conv, output_maps_count, phase_train) if apply_batch_norm else conv
    relu = activation(batch_norm, activation_func)

    return relu

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
    actv = activation(fc, activation_func)
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
size specified in model_settings['fingerprint_size'].

The function will build the graph it needs in the current TensorFlow graph,
and return the tensorflow output that will contain the 'logits' input to the
softmax prediction process. If training flag is on, it will also return a
placeholder node that can be used to control the dropout amount.

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
def create_model(fingerprint_input, 
                 model_settings, 
                 model_architecture,
                 is_training, 
                 runtime_settings=None):

  if model_architecture == 'conv':
    return create_conv_model(fingerprint_input, model_settings)
  elif model_architecture == 'slim_conv':
    return create_slim_conv_model(fingerprint_input, model_settings)
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
  model_settings: Dictionary of information about the model.
  is_training: Whether the model is going to be used for training.

Returns:
  TensorFlow node outputting logits results, and optionally a dropout
  placeholder.
"""
def create_conv_model(fingerprint_input, model_settings):
  dct_coefficient_count = model_settings['dct_coefficient_count']
  spectrogram_length = model_settings['spectrogram_length']
  fingerprint = tf.reshape(fingerprint_input, [-1, dct_coefficient_count, spectrogram_length, 1])
  phase_train = tf.placeholder(tf.bool, name='phase_train')

  # conv 1
  conv1 = conv_layer(fingerprint, 
                  model_settings['filter_height'][0], 
                  model_settings['filter_width'][0], 
                  fingerprint.shape[-1].value, 
                  model_settings['stride'][0],
                  model_settings['filter_count'][0],
                  model_settings['apply_batch_norm'],
                  phase_train,
                  model_settings['activation'],
                  model_settings['enable_hist_summary'])
  
  if model_settings['pooling'][0]:
    conv1 = x_pooling(conv1, model_settings['pooling'][0], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

  # conv 2
  conv2 = conv_layer(conv1, 
                  model_settings['filter_height'][1], 
                  model_settings['filter_width'][1], 
                  conv1.shape[-1].value, 
                  model_settings['stride'][1],
                  model_settings['filter_count'][1],
                  model_settings['apply_batch_norm'],
                  phase_train,
                  model_settings['activation'],
                  model_settings['enable_hist_summary'])
  
  if model_settings['pooling'][1]:
    conv2 = x_pooling(conv2, model_settings['pooling'][1], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

  # conv 2
  conv3 = conv_layer(conv2, 
                  model_settings['filter_height'][2], 
                  model_settings['filter_width'][2], 
                  conv2.shape[-1].value, 
                  model_settings['stride'][2],
                  model_settings['filter_count'][2],
                  model_settings['apply_batch_norm'],
                  phase_train,
                  model_settings['activation'],
                  model_settings['enable_hist_summary'])
  
  if model_settings['pooling'][2]:
    conv3 = x_pooling(conv3, model_settings['pooling'][2], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

  # conv 2
  conv4 = conv_layer(conv3, 
                  model_settings['filter_height'][3], 
                  model_settings['filter_width'][3], 
                  conv3.shape[-1].value, 
                  model_settings['stride'][3],
                  model_settings['filter_count'][3],
                  model_settings['apply_batch_norm'],
                  phase_train,
                  model_settings['activation'],
                  model_settings['enable_hist_summary'])
  
  if model_settings['pooling'][3]:
    conv4 = x_pooling(conv4, model_settings['pooling'][3], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')


  # flattened 
  [_, output_height, output_width, output_depth] = conv4.get_shape()
  element_count = int(output_height * output_width * output_depth)
  flattened = tf.reshape(conv4, [-1, element_count])

  # hidden layers
  fc1 = tf.layers.dense(flattened, model_settings['hidden_units'][0], activation=get_activation_func(model_settings['activation']))
  fc2 = tf.layers.dense(fc1, model_settings['hidden_units'][1], activation=get_activation_func(model_settings['activation']))

  # regression 
  estimator = tf.layers.dense(fc2, 1)
  
  # log
  tf.summary.image('input', fingerprint, 1)

  return estimator, phase_train


def create_slim_conv_model(fingerprint_input, model_settings):
  dct_coefficient_count = model_settings['dct_coefficient_count']
  spectrogram_length = model_settings['spectrogram_length']
  fingerprint = tf.reshape(fingerprint_input, [-1, dct_coefficient_count, spectrogram_length, 1])
  phase_train = tf.placeholder(tf.bool, name='phase_train')

  # conv layers
  output_conv = fingerprint
  for i in range(0, model_settings['conv_layers']):
    output_conv = conv_layer(output_conv, 
                    model_settings['filter_height'][i], 
                    model_settings['filter_width'][i], 
                    output_conv.shape[-1].value, 
                    model_settings['stride'][i],
                    model_settings['filter_count'][i],
                    model_settings['apply_batch_norm'],
                    phase_train,
                    model_settings['activation'],
                    model_settings['enable_hist_summary'])
    output_conv = x_pooling(output_conv, model_settings['pooling'][i], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME') if model_settings['pooling'][i] else output_conv

  # flattened 
  [_, output_height, output_width, output_depth] = output_conv.get_shape()
  element_count = int(output_height * output_width * output_depth)
  flattened = tf.reshape(output_conv, [-1, element_count])

  # hidden layers
  output_fc = flattened
  for i in range(0, model_settings['fc_layers']):
    output_fc = tf.layers.dense(output_fc, model_settings['hidden_units'][i], activation=get_activation_func(model_settings['activation']), name='hidden' + str(i))
    # output_fc = fc_layer(output_fc, output_fc.shape[-1].value, model_settings['hidden_units'][i], model_settings['activation'], model_settings['enable_hist_summary'])

  # regression 
  estimator = tf.layers.dense(output_fc, 1, name='estimator')
  # estimator = regression_layer(output_fc, output_fc.shape[-1].value, model_settings['enable_hist_summary'])
  
  # log
  tf.summary.image('input', fingerprint, 1)

  return estimator, phase_train