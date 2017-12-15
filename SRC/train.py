"""Training definitions for speech quality assessment.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tabulate import tabulate
import argparse
import time
import os.path
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import models
from tensorflow.python.platform import gfile

FLAGS = None


def main(_):

  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  sess = tf.InteractiveSession()

  # Begin by making sure we have the training data we need. If you already have
  # training data of your own, use `--data_url= ` on the command line to avoid
  # downloading.
  model_settings = models.prepare_model_settings(
      FLAGS.sample_rate, 
      FLAGS.clip_duration_ms, 
      FLAGS.window_size_ms,
      FLAGS.window_stride_ms, 
      FLAGS.dct_coefficient_count,
      FLAGS.conv_layers,
      FLAGS.filter_width,
      FLAGS.filter_count,
      FLAGS.stride,
      FLAGS.pooling,
      FLAGS.fc_layers,
      FLAGS.hidden_units)
  audio_processor = input_data.AudioProcessor(
      # FLAGS.data_url, 
      FLAGS.data_dir,  
      FLAGS.validation_percentage,
      FLAGS.testing_percentage, 
      FLAGS.feature_used,
      model_settings)

  # print size of training, validation and testing data
  train_len, val_len, test_len = audio_processor.get_set_sizes()
  print("***************** DataBase Length *****************")
  print("Training length: " + str(train_len))
  print("Validation length: " + str(val_len))
  print("Testing length: " + str(test_len))

  fingerprint_size = model_settings['fingerprint_size']
  time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
  # Figure out the learning rates for each training phase. Since it's often
  # effective to have high learning rates at the start of training, followed by
  # lower levels towards the end, the number of steps and learning rates can be
  # specified as comma-separated lists to define the rate at each stage. For
  # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
  # will run 13,000 training loops in total, with a rate of 0.001 for the first
  # 10,000, and 0.0001 for the final 3,000.
  training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
  learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
  if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
        '--how_many_training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))

  fingerprint_input = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')

  if FLAGS.model_architecture == 'conv':
    estimator, dropout_prob = models.create_model(
        fingerprint_input,
        model_settings,
        FLAGS.model_architecture,
        is_training=True)

  elif FLAGS.model_architecture == 'conv2':
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
  # cross_entropy_mean = tf.reduce_mean(
  #     tf.nn.softmax_cross_entropy_with_logits(
  #         labels=ground_truth_input, logits=estimator))
  root_mean_squared_error = tf.sqrt(tf.reduce_mean(tf.squared_difference(ground_truth_input, estimator)))
  tf.summary.scalar('rmse', root_mean_squared_error)

  with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
    learning_rate_input = tf.placeholder(
        tf.float32, [], name='learning_rate_input')
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate_input).minimize(root_mean_squared_error)

  #global_step = tf.contrib.framework.get_or_create_global_step()
  global_step = tf.train.get_or_create_global_step()
  increment_global_step = tf.assign(global_step, global_step + 1)

  saver = tf.train.Saver(tf.global_variables())

  # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
  merged_summaries = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                       sess.graph)
  validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

  tf.global_variables_initializer().run()

  start_step = 1

  if FLAGS.start_checkpoint:
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    start_step = global_step.eval(session=sess)
  
  print("***************** Training *****************")
  tf.logging.info('Training from step: %d ', start_step)

  # Save graph.pbtxt.
  tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                       FLAGS.model_architecture + '.pbtxt')

  initial_training_rmse = -1
  final_training_rmse = -1
  initial_validation_rmse = -1
  final_validation_rmse = -1

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
        FLAGS.batch_size, 0, model_settings, time_shift_samples, 'training', sess)

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
              phase_train: True,
              dropout_prob: 0.5
          })
    elif FLAGS.model_architecture == 'conv2':

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
      initial_training_rmse = train_rmse

    # ******* validation loop *******
    is_last_step = (training_step == training_steps_max)
    if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:

      if is_last_step:
        final_training_rmse = train_rmse

      set_size = audio_processor.set_size('validation')
      total_rmse = 0

      print("***************** Validation *****************")

      for i in xrange(0, set_size, FLAGS.batch_size):
        validation_fingerprints, validation_ground_truth = (
            audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
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
                  phase_train: True,
                  dropout_prob: 0.5
              })

        elif FLAGS.model_architecture == 'conv2':

          validation_summary, validation_rmse = sess.run(
              [
                merged_summaries, 
                root_mean_squared_error
              ],
              feed_dict={
                  fingerprint_input: validation_fingerprints,
                  ground_truth_input: validation_ground_truth,
                  phase_train: True
              })

        validation_writer.add_summary(validation_summary, training_step)
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_rmse += (validation_rmse * batch_size) / set_size

        tf.logging.info('i=%d: rmse = %.2f' % (i, validation_rmse))

      if training_step <= FLAGS.eval_step_interval:
        initial_validation_rmse = total_rmse
      if is_last_step:
        final_validation_rmse = total_rmse

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
        FLAGS.batch_size, i, model_settings, 0.0, 'testing', sess)

    testing_summary = None
    test_rmse = None
    if FLAGS.model_architecture == 'conv':
      
      testing_summary, test_rmse = sess.run(
          [merged_summaries, root_mean_squared_error],
          feed_dict={
              fingerprint_input: test_fingerprints,
              ground_truth_input: test_ground_truth,
              phase_train: True,
              dropout_prob: 1.0
          })

    elif FLAGS.model_architecture == 'conv2': 

      testing_summary, test_rmse = sess.run(
          [merged_summaries, root_mean_squared_error],
          feed_dict={
              fingerprint_input: test_fingerprints,
              ground_truth_input: test_ground_truth,
                  phase_train: True
          })

    batch_size = min(FLAGS.batch_size, set_size - i)
    total_rmse += (test_rmse * batch_size) / set_size
  tf.logging.info('Final RMSE = %.2f (N=%d)' % (total_rmse, set_size))
  
  # Write test configuration settings 
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
      ['training_size', train_len], 
      ['validation_percentage', FLAGS.testing_percentage], 
      ['validation_size', val_len], 
      ['testing_percentage', FLAGS.validation_percentage], 
      ['testing_size', test_len], 
      ['batch_size', FLAGS.batch_size], 
      ['how_many_training_steps', FLAGS.how_many_training_steps], 
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
    testFile.write(tabulate([['Initial Training', round(initial_training_rmse,2)], 
      ['Final Training', round(final_training_rmse,2)], 
      ['Initial Validation', round(initial_validation_rmse,2)], 
      ['Final Validation', round(final_validation_rmse,2)], 
      ['Test', round(total_rmse,2)]], 
      headers=['Set', 'RMSE'], tablefmt='psql'))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # parser.add_argument(
  #     '--data_url',
  #     type=str,
  #     # pylint: disable=line-too-long
  #     default='http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
  #     # pylint: enable=line-too-long
  #     help='Location of speech training data archive on the web.')
#DIRs
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/home/rqa/git/A-Convolutional-Neural-Network-Approach-for-Speech-Quality-Assessment/DATABASE/speech_dataset',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/home/rqa/git/A-Convolutional-Neural-Network-Approach-for-Speech-Quality-Assessment/Resultados/retrain_logs',
      help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/home/rqa/git/A-Convolutional-Neural-Network-Approach-for-Speech-Quality-Assessment/Resultados/speech_quality_evaluation_train',
      help='Directory to write event logs and checkpoint.')

#Signal
  parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=0.0,
      help="""\
      Range to randomly shift the training audio by in time.
      """)
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs')
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=8000,
      help='Expected duration in milliseconds of the wavs')

#Learning
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=30,
      help='How many items to train with at once')
  parser.add_argument(
      '--how_many_training_steps',
      type=str,
      default='5000,1000',
      help='How many training loops to run')  
  parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.01,0.001',
      help='How large a learning rate to use when training.')

#Spectrogram
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is')
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How long each spectrogram timeslice is')
  parser.add_argument(
      '--feature_used',
      type=str,
      default='mfcc',
      help='How long each spectrogram timeslice is')
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint')

#CNN
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='conv2',
      help='What model architecture to use')
  parser.add_argument(
      '--conv_layers',
      type=int,
      default=3,
      help='What model architecture to use')
  parser.add_argument(
      '--filter_width',
      type=int,
      default=5,
      help='What model architecture to use')
  parser.add_argument(
      '--filter_count',
      type=int,
      default=50,
      help='What model architecture to use')
  parser.add_argument(
      '--stride',
      type=int,
      default=1,
      help='What model architecture to use')
  parser.add_argument(
      '--pooling',
      type=str,
      default='avg',
      help='Number of units in hidden layer 1.')
  
#FC
  parser.add_argument(
      '--fc_layers',
      type=int,
      default=1,
      help='Number of units in hidden layer 1.')
  parser.add_argument(
      '--hidden_units',
      type=int,
      default=400,
      help='Number of units in hidden layer 1.')

  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=100,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--save_step_interval',
      type=int,
      default=200,
      help='Save model checkpoint every save_steps.')
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')


  parser.add_argument(
      '--check_nans',
      type=bool,
      default=False,
      help='Whether to check for invalid numbers during processing')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)