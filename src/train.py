"""Training definitions for speech quality assessment.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tabulate import tabulate
from collections import Counter
from datetime import datetime

import time
import re
import csv
import glob
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import input_data
import models
import config

def create_output_path(output_dir, config_name=''):
  if config_name:
    return output_dir + '/run-' + config_name
  else:
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return output_dir + '/run-' + now

def create_dir(new_dir): 
  if not os.path.exists(new_dir):
    os.makedirs(new_dir)

def main(argv):
  # Get flags
  if len(argv) == 1:
    [FLAGS], config_name = argv, ''
  else:
    [FLAGS, config_name] = argv

  # To see all the logging messages
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  sess = tf.InteractiveSession()

  # Begin by making sure we have the training data we need.
  model_settings = models.prepare_model_settings(
      FLAGS.enable_hist_summary,
      FLAGS.input_processing_lib,
      FLAGS.sample_rate, 
      FLAGS.clip_duration_ms, 
      FLAGS.window_size_ms,
      FLAGS.window_stride_ms, 
      FLAGS.data_aug_algorithms,
      FLAGS.feature,
      FLAGS.dct_coefficient_count,
      FLAGS.conv_layers,
      FLAGS.filter_width,
      FLAGS.filter_height,
      FLAGS.filter_count,
      FLAGS.stride,
      FLAGS.apply_batch_norm,
      FLAGS.activation,
      FLAGS.pooling,
      FLAGS.fc_layers,
      FLAGS.hidden_units)

  audio_processor = input_data.AudioProcessor(
      FLAGS.data_dir,  
      FLAGS.validation_percentage,
      FLAGS.testing_percentage, 
      model_settings)

  # print size of training, validation and testing data
  tf.logging.info("***************** DataBase Length *****************")
  tf.logging.info("Training length: " + str(audio_processor.set_size('training')))
  tf.logging.info("Validation length: " + str(audio_processor.set_size('validation')))
  tf.logging.info("Testing length: " + str(audio_processor.set_size('testing')))

  # input
  fingerprint_input = tf.placeholder(
      tf.float32, [None, model_settings['fingerprint_size']], name='fingerprint_input')

  ground_truth_input = tf.placeholder(
    tf.float32, [None, 1], name='groundtruth_input')
  
  # model
  estimator, phase_train = models.create_model(
      fingerprint_input, model_settings, FLAGS.model_architecture, is_training=True)

  # Optionally we can add runtime checks to spot when NaNs or other symptoms of
  # numerical errors start occurring during training.
  control_dependencies = []
  if FLAGS.check_nans:
    checks = tf.add_check_numerics_ops()
    control_dependencies = [checks]

  # Define loss 
  with tf.name_scope('loss'):
    root_mean_squared_error = tf.sqrt(
      tf.reduce_mean(tf.squared_difference(ground_truth_input, estimator)), name='rmse')
  tf.summary.scalar('rmse', root_mean_squared_error)

  # Create the back propagation and training evaluation machinery in the graph.
  with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
    learning_rate_input = tf.placeholder(tf.float32, [], name='learning_rate_input')
    
    if FLAGS.optimizer == 'momentum':
      train_step = tf.train.MomentumOptimizer(
        learning_rate=learning_rate_input, momentum=0.9, use_nesterov=True).minimize(root_mean_squared_error)
    elif FLAGS.optimizer == 'rms_prop':
      train_step = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate_input, momentum=0.9, decay=0.9, epsilon=1e-10).minimize(root_mean_squared_error)
    elif FLAGS.optimizer == 'adam':
      train_step = tf.train.AdamOptimizer(learning_rate=learning_rate_input).minimize(root_mean_squared_error)
    elif FLAGS.optimizer == 'gradient_descent':
      train_step = tf.train.GradientDescentOptimizer(
            learning_rate_input).minimize(root_mean_squared_error)

  global_step = tf.train.get_or_create_global_step()
  increment_global_step = tf.assign(global_step, global_step + 1)
  saver = tf.train.Saver(tf.global_variables())

  # Merge all the summaries and create file writers 
  merged_summaries = tf.summary.merge_all()
  output_dir = create_output_path(FLAGS.output_dir, config_name)

  train_writer = tf.summary.FileWriter(output_dir + '/summary/train', sess.graph)
  #validation_writer = tf.summary.FileWriter(output_dir + '/summary/validation')
  weighted_validation_writer = tf.summary.FileWriter(output_dir + '/summary/weighted_validation')
  
  if FLAGS.enable_profile:
    profile_dir = output_dir + '/profile'
    create_dir(profile_dir)

  if len(FLAGS.training_steps) != len(FLAGS.learning_rate):
    raise Exception(
        '--training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(FLAGS.training_steps),
                                                   len(FLAGS.training_steps)))

  tf.global_variables_initializer().run()
  start_step = 1

  tf.logging.info('"***************** Training *****************')
  tf.logging.info('Training from step: %d ', start_step)

  options = None
  run_metadata = None
  if FLAGS.enable_profile:
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

  training_steps_max = np.sum(FLAGS.training_steps)
  for training_step in range(start_step, training_steps_max + 1):
    # Figure out what the current learning rate is.
    training_steps_sum = 0
    for i in range(len(FLAGS.training_steps)):
      training_steps_sum += FLAGS.training_steps[i]
      if training_step <= training_steps_sum:
        learning_rate_value = FLAGS.learning_rate[i]
        break

    # Pull the audio samples we'll use for training.
    _, train_fingerprints, train_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, 0, 'training', sess)

    # Run the graph with this batch of training data.
    train_summary, train_rmse, _, _ = sess.run(
        [
            merged_summaries, 
            root_mean_squared_error, 
            train_step,
            increment_global_step
        ],
        feed_dict={
            fingerprint_input: train_fingerprints,
            ground_truth_input: train_ground_truth,
            learning_rate_input: learning_rate_value,
            phase_train: True
        },
        options=options,
        run_metadata=run_metadata)

    tf.logging.info('step #%d: rate %f, rmse %f' %
                    (training_step, learning_rate_value, train_rmse))

    if (training_step % FLAGS.summary_step_interval) == 0:
      train_writer.add_summary(train_summary, training_step)
    
    if FLAGS.enable_profile:
      fetched_timeline = timeline.Timeline(run_metadata.step_stats)
      chrome_trace = fetched_timeline.generate_chrome_trace_format()
      with open(profile_dir + '/' + 'timeline_training_step_%d.json' % training_step, 'w') as f:
        f.write(chrome_trace)

    if (training_step % FLAGS.eval_step_interval) == 0 or (training_step == training_steps_max):
      tf.logging.info('***************** Validation *****************')
      
      set_size = audio_processor.set_size('validation')
      total_rmse = 0
      weights = np.array([], dtype=np.float32)
      values = np.array([], dtype=np.float32)

      for i in range(0, set_size, FLAGS.batch_size):
        _, validation_fingerprints, validation_ground_truth = audio_processor.get_data(FLAGS.batch_size, i, 'validation', sess)

        validation_summary, validation_rmse = sess.run(
            [
              merged_summaries, 
              root_mean_squared_error
            ],
            feed_dict={
              fingerprint_input: validation_fingerprints,
              ground_truth_input: validation_ground_truth,
              phase_train: False
            })

        weights = np.append(weights, validation_fingerprints.shape[0] / set_size)
        values = np.append(values, validation_rmse)
        #validation_writer.add_summary(validation_summary, training_step + i/FLAGS.batch_size)
        tf.logging.info('i=%d: rmse = %.2f' % (i, validation_rmse))

      weighted_rmse = np.dot(values, weights)
      weighted_rmse_summary = tf.Summary(value=[tf.Summary.Value(tag='rmse',
                                                     simple_value=weighted_rmse)])
      weighted_validation_writer.add_summary(weighted_rmse_summary, training_step + set_size/FLAGS.batch_size - 1)
      tf.logging.info('weighted rmse = %.2f (N=%d)' % (weighted_rmse, set_size))
      tf.logging.info('***************** ********** *****************')

  if FLAGS.evaluate_testing:
    tf.logging.info('')
    tf.logging.info('****************** Testing ******************')
    
    set_size = audio_processor.set_size('testing')
    total_rmse = 0
    weights = np.array([], dtype=np.float32)
    values = np.array([], dtype=np.float32)
    
    for i in range(0, set_size, FLAGS.batch_size):
      _, test_fingerprints, test_ground_truth = audio_processor.get_data(
          FLAGS.batch_size, i, 'testing', sess)

      testing_summary, test_rmse = sess.run(
          [
            merged_summaries, 
            root_mean_squared_error
          ],
          feed_dict={
              fingerprint_input: test_fingerprints,
              ground_truth_input: test_ground_truth,
              phase_train: False
          })

      weights = np.append(weights, test_fingerprints.shape[0] / set_size)
      values = np.append(values, test_rmse)
      # test_writer.add_summary(testing_summary, training_steps_max + i/FLAGS.batch_size)
      tf.logging.info('i=%d: rmse = %.2f' % (i, test_rmse))

    weighted_rmse = np.dot(values, weights)
    tf.logging.info('weighted rmse = %.2f (N=%d)' % (weighted_rmse, set_size))
    tf.logging.info('***************** ********** *****************')

  # Save the model
  if FLAGS.enable_checkpoint_save:
    create_dir(output_dir + '/checkpoint')
    FLAGS.start_checkpoint = os.path.join(output_dir + '/checkpoint', FLAGS.model_architecture + '.ckpt')
    tf.logging.info('Saving to "%s-%d"', FLAGS.start_checkpoint, training_steps_max)
    saver.save(sess, FLAGS.start_checkpoint, global_step=training_steps_max)
    FLAGS.start_checkpoint += '-' + str(training_steps_max)

  config.save_configs(output_dir, FLAGS.__dict__, )

  train_writer.close()
  #validation_writer.close()
  weighted_validation_writer.close()
  sess.close()

if __name__ == '__main__':
  FLAGS, _unparsed = config.set_flags()
  tf.app.run(main=main, argv=[FLAGS])
