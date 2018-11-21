"""Training definitions for speech quality assessment.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tabulate import tabulate
import time
import os.path

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import models
import config

def save_test_setting(FLAGS, audio_processor, results):
  # Write test settings 
  fileName = 'test_' + time.strftime('%Y%m%d') + '_' + time.strftime('%H%M%S') 
  with open(FLAGS.summaries_dir + '/' + fileName + '.txt', 'w') as testFile:
    #testFile.write('+----------------------------+\n')
    #testFile.write('|Using pre-processed signals!|\n')
    #testFile.write('|Silence removed            !|\n')
    #testFile.write('+----------------------------+\n\n')

    testFile.write('+-------------+\n')
    testFile.write('|Configuration|\n')
    testFile.write('+-------------+\n')
    testFile.write(tabulate([['training_percentage', 100-FLAGS.testing_percentage-FLAGS.validation_percentage],
      ['training_size', audio_processor.set_size('training')], 
      ['validation_percentage', FLAGS.testing_percentage], 
      ['validation_size', audio_processor.set_size('validation')], 
      ['testing_percentage', FLAGS.validation_percentage], 
      ['testing_size', audio_processor.set_size('testing')], 
      ['batch_size', FLAGS.batch_size], 
      ['training_steps', FLAGS.training_steps], 
      ['learning_rate', FLAGS.learning_rate]], 
      headers=['Learning', ''], tablefmt='psql') + '\n')
    testFile.write(tabulate([['window_size_ms', FLAGS.window_size_ms], 
      ['window_stride_ms', FLAGS.window_stride_ms], 
      ['dct_coefficient_count', FLAGS.dct_coefficient_count]], 
      headers=['Spectrogram', ''], tablefmt='psql') + '\n')
    testFile.write(tabulate([['conv_layers', FLAGS.conv_layers], 
      ['activation_function', 'ReLU'], 
      ['pooling', FLAGS.pooling], 
      ['filter_width', FLAGS.filter_width], 
      ['filter_count', FLAGS.filter_count], 
      ['stride', FLAGS.stride]], 
      headers=['CNN', ''], tablefmt='psql') + '\n')   
    testFile.write(tabulate([['fc_layers', FLAGS.fc_layers], 
      ['activation_function', 'ReLU'], 
      ['hidden_units', FLAGS.hidden_units]], 
      headers=['FC', ''], tablefmt='psql') + '\n')   

    testFile.write('\n\n')
    testFile.write('+-------------+\n')
    testFile.write('|Results      |\n')
    testFile.write('+-------------+\n')
    testFile.write(tabulate([['Initial Training', round(results['initial_training_rmse'],2)], 
      ['Final Training', round(results['final_training_rmse'],2)], 
      ['Initial Validation', round(results['initial_validation_rmse'],2)], 
      ['Final Validation', round(results['final_validation_rmse'],2)], 
      ['Test', round(results['final_rmse'],2)]], 
      headers=['Set', 'RMSE'], tablefmt='psql'))

def main(argv):
  # Get flags
  [FLAGS] = argv

  # To see all the logging messages
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  sess = tf.InteractiveSession()

  # Begin by making sure we have the training data we need.
  model_settings = models.prepare_model_settings(
      FLAGS.sample_rate, 
      FLAGS.clip_duration_ms, 
      FLAGS.window_size_ms,
      FLAGS.window_stride_ms, 
      FLAGS.feature_used,
      FLAGS.dct_coefficient_count,
      FLAGS.conv_layers,
      FLAGS.filter_width,
      FLAGS.filter_height,
      FLAGS.filter_count,
      FLAGS.stride,
      FLAGS.pooling,
      FLAGS.fc_layers,
      FLAGS.hidden_units)
  audio_processor = input_data.AudioProcessor(
      FLAGS.data_dir,  
      FLAGS.validation_percentage,
      FLAGS.testing_percentage, 
      model_settings)

  # print size of training, validation and testing data
  print("***************** DataBase Length *****************")
  print("Training length: " + str(audio_processor.set_size('training')))
  print("Validation length: " + str(audio_processor.set_size('validation')))
  print("Testing length: " + str(audio_processor.set_size('testing')))

  # Figure out the learning rates for each training phase. Since it's often
  # effective to have high learning rates at the start of training, followed by
  # lower levels towards the end, the number of steps and learning rates can be
  # specified as comma-separated lists to define the rate at each stage.
  training_steps_list = list(map(int, FLAGS.training_steps.split(';')))
  learning_rates_list = list(map(float, FLAGS.learning_rate.split(';')))
  if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
        '--training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))

  fingerprint_input = tf.placeholder(
      tf.float32, [None, model_settings['fingerprint_size']], name='fingerprint_input')

  if FLAGS.model_architecture == 'conv':
    estimator, dropout_prob = models.create_model(
        fingerprint_input,
        model_settings,
        FLAGS.model_architecture,
        is_training=True)
  else:
    estimator, phase_train = models.create_model(
        fingerprint_input,
        model_settings,
        FLAGS.model_architecture,
        is_training=True)

  # Define loss and optimizer
  ground_truth_input = tf.placeholder(
      tf.float32, [None, 1], name='groundtruth_input')

  # Optionally we can add runtime checks to spot when NaNs or other symptoms of
  # numerical errors start occurring during training.
  control_dependencies = []
  if FLAGS.check_nans:
    checks = tf.add_check_numerics_ops()
    control_dependencies = [checks]

  # Create the back propagation and training evaluation machinery in the graph.
  root_mean_squared_error = tf.sqrt(tf.reduce_mean(tf.squared_difference(ground_truth_input, estimator)))
  tf.summary.scalar('rmse', root_mean_squared_error)

  with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
    learning_rate_input = tf.placeholder(
        tf.float32, [], name='learning_rate_input')
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate_input).minimize(root_mean_squared_error)

  global_step = tf.train.get_or_create_global_step()
  increment_global_step = tf.assign(global_step, global_step + 1)

  saver = tf.train.Saver(tf.global_variables())

  # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
  merged_summaries = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',sess.graph)
  validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

  tf.global_variables_initializer().run()

  start_step = 1

  if FLAGS.start_checkpoint:
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    start_step = global_step.eval(session=sess)

  results = {}
  results['initial_training_rmse'] = -1
  results['final_training_rmse'] = -1
  results['initial_validation_rmse'] = -1
  results['final_validation_rmse'] = -1

  # Save graph.pbtxt.
  tf.train.write_graph(sess.graph_def, FLAGS.train_dir, FLAGS.model_architecture + '.pbtxt')
  
  print("***************** Training *****************")
  tf.logging.info('Training from step: %d ', start_step)

  # ******* train loop *******
  training_steps_max = np.sum(training_steps_list)
  for training_step in xrange(start_step, training_steps_max + 1):

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
    train_summary = None
    train_rmse = None
    if FLAGS.model_architecture == 'conv':
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
              # phase_train: True,
              dropout_prob: 0.5
          })
    else:
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

    train_writer.add_summary(train_summary, training_step)
    tf.logging.info('Step #%d: rate %f, root mean squared error %f' %
                    (training_step, learning_rate_value, train_rmse))

    if training_step == 1:
      results['initial_training_rmse'] = train_rmse

    # ******* validation loop *******
    is_last_step = (training_step == training_steps_max)
    if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:

      if is_last_step:
        results['final_training_rmse'] = train_rmse

      set_size = audio_processor.set_size('validation')
      total_rmse = 0

      print("***************** Validation *****************")

      for i in xrange(0, set_size, FLAGS.batch_size):
        validation_fingerprints, validation_ground_truth = (
            audio_processor.get_data(FLAGS.batch_size, i, model_settings,
                                     'validation', sess))

        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary = None
        validation_rmse = None
        if FLAGS.model_architecture == 'conv':
          validation_summary, validation_rmse = sess.run(
              [
                merged_summaries, 
                root_mean_squared_error
              ],
              feed_dict={
                  fingerprint_input: validation_fingerprints,
                  ground_truth_input: validation_ground_truth,
                  # phase_train: False,
                  dropout_prob: 0.5
              })
        else:
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

        validation_writer.add_summary(validation_summary, training_step)
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_rmse += (validation_rmse * batch_size) / set_size

        tf.logging.info('i=%d: rmse = %.2f' % (i, validation_rmse))

      if training_step <= FLAGS.eval_step_interval:
        results['initial_validation_rmse'] = total_rmse
      if is_last_step:
        results['final_validation_rmse'] = total_rmse

      tf.logging.info('Step %d: RMSE = %.2f (N=%d)' %
                      (training_step, total_rmse, set_size))
      print("***************** ********** *****************")

    # Save the model checkpoint periodically.
    if (training_step % FLAGS.save_step_interval == 0 or
        training_step == training_steps_max):
      checkpoint_path = os.path.join(FLAGS.train_dir,
                                     FLAGS.model_architecture + '.ckpt')
      tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
      saver.save(sess, checkpoint_path, global_step=training_step)

  # ******* testing loop *******
  set_size = audio_processor.set_size('testing')
  tf.logging.info('set_size=%d', set_size)
  total_rmse = 0
  for i in xrange(0, set_size, FLAGS.batch_size):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, i, model_settings, 'testing', sess)

    testing_summary = None
    test_rmse = None

    if FLAGS.model_architecture == 'conv':
      testing_summary, test_rmse = sess.run(
          [merged_summaries, root_mean_squared_error],
          feed_dict={
              fingerprint_input: test_fingerprints,
              ground_truth_input: test_ground_truth,
              # phase_train: False,
              dropout_prob: 1.0
          })
    else: 
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

    batch_size = min(FLAGS.batch_size, set_size - i)
    total_rmse += (test_rmse * batch_size) / set_size
    tf.logging.info('i=%d: rmse = %.2f' % (i, test_rmse))
  tf.logging.info('Final RMSE = %.2f (N=%d)' % (total_rmse, set_size))
  results['final_rmse'] = total_rmse
  # save test settings 
  save_test_setting(FLAGS, audio_processor, results)

if __name__ == '__main__':
  FLAGS, _unparsed = config.set_flags()
  tf.app.run(main=main, argv=[FLAGS])
