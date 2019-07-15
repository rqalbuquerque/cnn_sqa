"""Model definitions for speech quality assessment.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from six.moves import xrange
from functools import partial


def prepare_model_settings(sample_rate,
                           clip_duration_ms,
                           window_size_ms,
                           window_stride_ms,
                           feature,
                           n_coeffs,
                           filter_width,
                           filter_height,
                           filter_count,
                           stride,
                           apply_batch_norm,
                           activation,
                           apply_dropout,
                           hidden_units):
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

    desired_samples = int(sample_rate * clip_duration_ms / 1000.0)
    window_size_samples = int(sample_rate * window_size_ms / 1000.0)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000.0)
    n_frames = 1 + int((desired_samples-window_size_samples) / window_stride_samples)
    fingerprint_size = n_coeffs * n_frames

    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'fingerprint_size': fingerprint_size,
        'sample_rate': sample_rate,
        'n_frames': n_frames,
        'feature': feature,
        'n_coeffs': n_coeffs,
        'filter_width': filter_width,
        'filter_height': filter_height,
        'filter_count': filter_count,
        'stride': stride,
        'apply_dropout': apply_dropout,
        'apply_batch_norm': apply_batch_norm,
        'activation': activation,
        'hidden_units': hidden_units
    }


def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.

    Args:
      sess: TensorFlow session.
      start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)


def activation(input_tensor, mode):
    if mode == "softplus":
        return tf.nn.softplus(input_tensor, name="softplus")
    elif mode == "elu":
        return tf.nn.elu(input_tensor, name="elu")
    elif mode == "relu":
        return tf.nn.relu(input_tensor, name="relu")
    else:
        raise Exception("Invalid activation function!")


def get_activation(mode):
    if mode == "softplus":
        return tf.nn.softplus
    elif mode == "elu":
        return tf.nn.elu
    elif mode == "relu":
        return tf.nn.relu
    else:
        raise Exception("Invalid activation function!")


def batch_normalization(input_tensor, n_out, phase_train):
    """
    Batch normalization on convolutional maps.

    Args:
        input_tensor: Tensor, 4D BHWD input maps
        n_out:        integer, depth of input maps
        phase_train:  boolean tf.Varialbe, true indicates training phase
        scope:        string, variable scope

    Return:
        Batch-normalized maps
    """
    with tf.name_scope('batch_norm'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(input_tensor, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

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
            output_maps_count):
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
        convolution = tf.nn.conv2d(input_tensor, weights, [1, stride, stride, 1], 'SAME', name='convolution')
        conv = tf.math.add(convolution, bias, name='sum')
        return conv


def create_model(fingerprint_input,
                 params,
                 model_architecture):
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

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.

    Raises:
      Exception: If the architecture type isn't recognized.
    """

    if model_architecture == 'conv':
        return create_conv_model(fingerprint_input, params)
    else:
        raise Exception('model_architecture argument "' + model_architecture +
                        '" not recognized, should be "conv"')


def create_conv_model(fingerprint_input, params):
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
        [Dropout]
            v
            .
            . (3 more conv layers)
            .
            v
        [Conv2D]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
    [BatchNormaliztion]
            v
          [Relu]
            v
        [Dropout]
            v
      [FullConected]
            v
          [Relu]
            v
      [FullConected]
            v
          [Relu]
            v
        [estimator]

    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      params: Dictionary of information about the model.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    fingerprint = tf.reshape(
        fingerprint_input, [-1, params['n_coeffs'], params['n_frames'], 1])
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    # log
    tf.summary.image('input', fingerprint, 1)

    # partials
    batch_norm_layer = partial(batch_normalization, phase_train=phase_train)
    activation_layer = partial(activation, mode=params['activation'])
    dropout_layer = partial(tf.layers.dropout, rate=0.5,
                            training=phase_train, name='dropout')
    dense_layer = partial(tf.layers.dense, activation=get_activation("relu"))

    # c0
    with tf.name_scope('conv_0'):
        conv_0 = conv_2d(
            fingerprint,
            params['filter_height'][0],
            params['filter_width'][0],
            fingerprint.shape[-1].value,
            params['stride'][0],
            params['filter_count'][0]
        )
        batch_norm_0 = batch_norm_layer(
            conv_0, params['filter_count'][0]) if params['apply_batch_norm'] else conv_0
        activation_0 = activation_layer(batch_norm_0)
        dropout_0 = dropout_layer(
            activation_0) if params['apply_dropout'] else activation_0

    # c1
    with tf.name_scope('conv_1'):
        conv_1 = conv_2d(
            dropout_0,
            params['filter_height'][1],
            params['filter_width'][1],
            dropout_0.shape[-1].value,
            params['stride'][1],
            params['filter_count'][1]
        )
        batch_norm_1 = batch_norm_layer(
            conv_1, params['filter_count'][1]) if params['apply_batch_norm'] else conv_1
        activation_1 = activation_layer(batch_norm_1)
        dropout_1 = dropout_layer(
            activation_1) if params['apply_dropout'] else activation_1

    # c2
    with tf.name_scope('conv_2'):
        conv_2 = conv_2d(
            dropout_1,
            params['filter_height'][2],
            params['filter_width'][2],
            dropout_1.shape[-1].value,
            params['stride'][2],
            params['filter_count'][2]
        )
        batch_norm_2 = batch_norm_layer(
            conv_2, params['filter_count'][2]) if params['apply_batch_norm'] else conv_2
        activation_2 = activation_layer(batch_norm_2)
        dropout_2 = dropout_layer(
            activation_2) if params['apply_dropout'] else activation_2

    # c3
    with tf.name_scope('conv_3'):
        conv_3 = conv_2d(
            dropout_2,
            params['filter_height'][3],
            params['filter_width'][3],
            dropout_2.shape[-1].value,
            params['stride'][3],
            params['filter_count'][3]
        )
        batch_norm_3 = batch_norm_layer(
            conv_3, params['filter_count'][3]) if params['apply_batch_norm'] else conv_3
        activation_3 = activation_layer(batch_norm_3)
        dropout_3 = dropout_layer(
            activation_3) if params['apply_dropout'] else activation_3

    # c4
    with tf.name_scope('conv_4'):
        conv_4 = conv_2d(
            dropout_3,
            params['filter_height'][4],
            params['filter_width'][4],
            dropout_3.shape[-1].value,
            params['stride'][4],
            params['filter_count'][4]
        )
        batch_norm_4 = batch_norm_layer(
            conv_4, params['filter_count'][4]) if params['apply_batch_norm'] else conv_4
        activation_4 = activation_layer(batch_norm_4)
        dropout_4 = dropout_layer(
            activation_4) if params['apply_dropout'] else activation_4

    # flattened
    [_, output_height, output_width, output_depth] = dropout_4.get_shape()
    element_count = int(output_height * output_width * output_depth)
    flattened = tf.reshape(dropout_4, [-1, element_count], name='flatten')

    # dense
    fc_1 = dense_layer(flattened, params['hidden_units'][0], name='dense_0')
    fc_2 = dense_layer(fc_1, params['hidden_units'][1], name='dense_1')

    # regression
    estimator = tf.layers.dense(fc_2, 1, name='estimator')

    return estimator, phase_train
