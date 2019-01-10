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
def prepare_model_settings(input_processing_lib,
                           enable_hist_summary,
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
                           pooling,
                           fc_layers,
                           hidden_units):

  desired_samples = int(sample_rate * clip_duration_ms / 1000.0)
  window_size_samples = int(sample_rate * window_size_ms / 1000.0)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000.0)
  spectrogram_length = 1 + int(desired_samples / window_stride_samples)
  fingerprint_size = dct_coefficient_count * spectrogram_length
  
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'fingerprint_size': fingerprint_size,
      'input_processing_lib': input_processing_lib,
      'enable_hist_summary': enable_hist_summary,
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
  return tf.nn.relu(input_tensor) if mode == "relu" else input_tensor

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
    beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                 name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                  name='gamma', trainable=True)
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
               phase_train,
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
    norm_conv = batch_normalization(conv, output_maps_count, phase_train)
    relu = activation(norm_conv, "relu")

    # tf.summary.image('cnn_weights', tf.transpose(weights,[2,1,0,3]), 1)

    if enable_hist_summary:
      tf.summary.histogram('weights', weights)
      tf.summary.histogram('bias', bias)
      tf.summary.histogram('relu', relu)

    return relu

"""
Fully-connected layer.
Args:
    input_tensor: Tensor, 4D BHWD input maps
    input_units: integer, number of input units
    hidden_units: integer, number of hidden units
Return:
    output matrix
"""
def fully_connected(input_tensor, 
                    input_units, 
                    hidden_units, 
                    enable_hist_summary):
  with tf.name_scope('fc'):
    weights = tf.Variable(
      tf.truncated_normal(
        [
          input_units, 
          hidden_units
        ], 
        stddev=0.01,
        name='random'), 
      name='weights')
    bias = tf.Variable(tf.zeros([hidden_units]), name='biases')

    if enable_hist_summary:
      tf.summary.histogram('weights', weights)
      tf.summary.histogram('bias', bias)

    return tf.math.add(tf.matmul(input_tensor, weights), bias, name='sum')

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
    return create_conv2_model(fingerprint_input, model_settings)
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
def create_conv2_model(fingerprint_input, model_settings):
  dct_coefficient_count = model_settings['dct_coefficient_count']
  spectrogram_length = model_settings['spectrogram_length']
  fingerprint = tf.reshape(fingerprint_input,
                  [-1, dct_coefficient_count, spectrogram_length, 1])
  
  phase_train = tf.placeholder(tf.bool, name='phase_train')

  # conv layer 1
  conv_1 = conv_layer(fingerprint, 
                       model_settings['filter_height'][0], 
                       model_settings['filter_width'][0], 
                       int(fingerprint.shape[-1]), 
                       model_settings['stride'][0],
                       model_settings['filter_count'][0],
                       phase_train,
                       model_settings['enable_hist_summary'])

  # conv layer 2
  conv_2 = conv_layer(conv_1, 
                       model_settings['filter_height'][1], 
                       model_settings['filter_width'][1], 
                       int(conv_1.shape[-1]), 
                       model_settings['stride'][1],
                       model_settings['filter_count'][1],
                       phase_train,
                       model_settings['enable_hist_summary'])

  # conv layer 3
  conv_3 = conv_layer(conv_2, 
                       model_settings['filter_height'][2], 
                       model_settings['filter_width'][2], 
                       int(conv_2.shape[-1]), 
                       model_settings['stride'][2],
                       model_settings['filter_count'][2],
                       phase_train,
                       model_settings['enable_hist_summary'])

  # pooling
  # pooling = x_pooling(conv_3, 
  #                      model_settings['pooling'], 
  #                      [1, 2, 2, 1], 
  #                      [1, 2, 2, 1], 
  #                      'SAME')

  # flattened pooling
  [_, output_height, output_width, output_depth] = conv_3.get_shape()
  element_count = int(output_height * output_width * output_depth)
  flattened = tf.reshape(conv_3, [-1, element_count])

  # fc layer 1
  fc_1 = fully_connected(flattened, element_count, model_settings['hidden_units'][0], model_settings['enable_hist_summary'])
  final_fc_relu = activation(fc_1, "relu")

  # regression 
  estimator = fully_connected(final_fc_relu, int(final_fc_relu.shape[-1]), 1, model_settings['enable_hist_summary'])
  
  # log
  tf.summary.image('input', fingerprint, 1)

  return estimator, phase_train