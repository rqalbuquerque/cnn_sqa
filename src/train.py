"""Training definitions for speech quality assessment.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tabulate import tabulate
from collections import Counter
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

def create_log_path(log_dir):
  dirs = os.listdir(log_dir)
  if len(dirs) > 0:
    indexes = [int(x.replace('run','')) for x in dirs]
    num = sorted(indexes)[-1] + 1
    return log_dir + '/run' + str(num)
  else:
    return log_dir + '/run1'

def create_dir(directory): 
  if not os.path.exists(directory):
    os.makedirs(directory)

def main(argv):
  # Get flags
  [FLAGS] = argv

  # To see all the logging messages
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  sess = tf.InteractiveSession()

  # Begin by making sure we have the training data we need.
  model_settings = models.prepare_model_settings(
      FLAGS.input_processing_lib,
      FLAGS.enable_hist_summary,
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

  # model
  estimator, phase_train = models.create_model(
      fingerprint_input,
      model_settings,
      FLAGS.model_architecture,
      is_training=True)

  # Define loss and optimizer
  ground_truth_input = tf.placeholder(tf.float32, [None, 1], name='groundtruth_input')

  # Optionally we can add runtime checks to spot when NaNs or other symptoms of
  # numerical errors start occurring during training.
  control_dependencies = []
  if FLAGS.check_nans:
    checks = tf.add_check_numerics_ops()
    control_dependencies = [checks]

  # Create the back propagation and training evaluation machinery in the graph.
  with tf.name_scope('loss'):
    root_mean_squared_error = tf.sqrt(
      tf.reduce_mean(tf.squared_difference(ground_truth_input, estimator)))
  tf.summary.scalar('rmse', root_mean_squared_error)

  with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
    learning_rate_input = tf.placeholder(
        tf.float32, [], name='learning_rate_input')
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate_input).minimize(root_mean_squared_error)

  global_step = tf.train.get_or_create_global_step()
  increment_global_step = tf.assign(global_step, global_step + 1)

  # saver = tf.train.Saver(tf.global_variables())

  # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
  merged_summaries = tf.summary.merge_all()
  log_dir = create_log_path(FLAGS.summaries_dir)

  train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
  validation_writer = tf.summary.FileWriter(log_dir + '/validation')
  weighted_validation = tf.summary.FileWriter(log_dir + '/weighted_validation')

  if FLAGS.enable_profile:
    profile_dir = FLAGS.summaries_dir + '/profile'
    create_dir(profile_dir)

  tf.global_variables_initializer().run()

  start_step = 1
  if FLAGS.start_checkpoint:
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    start_step = global_step.eval(session=sess)

  config.save_configs(FLAGS.__dict__, log_dir)

  # Save graph.pbtxt.
  #tf.train.write_graph(sess.graph_def, FLAGS.train_dir, FLAGS.model_architecture + '.pbtxt')

  training_steps_list = FLAGS.training_steps
  learning_rates_list = FLAGS.learning_rate
  if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
        '--training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))
  
  tf.logging.info('"***************** Training *****************')
  tf.logging.info('Training from step: %d ', start_step)

  # options = None
  # run_metadata = None
  # if FLAGS.enable_profile:
  #   options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  #   run_metadata = tf.RunMetadata()

  training_steps_max = np.sum(training_steps_list)
  for training_step in range(start_step, training_steps_max + 1):
    # Figure out what the current learning rate is.
    training_steps_sum = 0
    for i in range(len(training_steps_list)):
      training_steps_sum += training_steps_list[i]
      if training_step <= training_steps_sum:
        learning_rate_value = learning_rates_list[i]
        break

    # Pull the audio samples we'll use for training.
    train_fingerprints, train_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, 0, model_settings, 'training', sess)

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
        })
        # options=options,
        # run_metadata=run_metadata)

    train_writer.add_summary(train_summary, training_step)
    tf.logging.info('step #%d: rate %f, rmse %f' %
                    (training_step, learning_rate_value, train_rmse))

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
        validation_fingerprints, validation_ground_truth = (
            audio_processor.get_data(FLAGS.batch_size, i, model_settings,
                                     'validation', sess))

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
        validation_writer.add_summary(validation_summary, training_step)
        tf.logging.info('i=%d: rmse = %.2f' % (i, validation_rmse))

      weighted_rmse = np.dot(values, weights)
      weighted_rmse_summary = tf.Summary(value=[tf.Summary.Value(tag='rmse',
                                                     simple_value=weighted_rmse)])
      weighted_validation.add_summary(weighted_rmse_summary, training_step)
      tf.logging.info('weighted rmse = %.2f (N=%d)' % (weighted_rmse, set_size))
      tf.logging.info('***************** ********** *****************')

  if FLAGS.apply_testing:
    tf.logging.info('')
    tf.logging.info('****************** Testing ******************')
    
    set_size = audio_processor.set_size('testing')
    total_rmse = 0
    weights = []
    values = []
    
    for i in range(0, set_size, FLAGS.batch_size):
      test_fingerprints, test_ground_truth = audio_processor.get_data(
          FLAGS.batch_size, i, model_settings, 'testing', sess)

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

      weights.append(test_fingerprints.shape[0] / set_size)
      values.append(test_rmse)
      # test_writer.add_summary(testing_summary, i)
      tf.logging.info('i=%d: rmse = %.2f' % (i, test_rmse))

    weighted_rmse = np.dot(values, weights)
    tf.logging.info('weighted rmse = %.2f (N=%d)' % (weighted_rmse, set_size))
    tf.logging.info('***************** ********** *****************')

  sess.close()

if __name__ == '__main__':
  FLAGS, _unparsed = config.set_flags()
  tf.app.run(main=main, argv=[FLAGS])