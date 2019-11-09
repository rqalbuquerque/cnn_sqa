"""Training definitions for speech quality assessment.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime

import numpy as np
import tensorflow as tf

import sys

import utils
import config
import input_data
import models


def create_output_path(output_dir, config_name=''):
    if config_name:
        return output_dir + '/run-' + config_name
    else:
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return output_dir + '/run-' + now


def apply_optimizer(learning_rate):
  with tf.name_scope('optimizer'):
    return tf.train.GradientDescentOptimizer(learning_rate)


def apply_loss(gt_input, estimated):
  with tf.name_scope('loss'):
    return tf.sqrt(tf.reduce_mean(tf.squared_difference(gt_input, estimated)))

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

    # Audio processor 
    model_settings = models.prepare_model_settings(
        FLAGS.sample_rate,
        FLAGS.clip_duration_ms,
        FLAGS.window_size_ms,
        FLAGS.window_stride_ms,
        FLAGS.feature,
        FLAGS.n_coeffs,
        FLAGS.filter_width,
        FLAGS.filter_height,
        FLAGS.filter_count,
        FLAGS.stride,
        FLAGS.apply_batch_norm,
        FLAGS.activation,
        FLAGS.apply_dropout,
        FLAGS.hidden_units)

    audio_processor = input_data.AudioProcessor(model_settings)
    audio_processor.index_from_csv(
        FLAGS.data_dir,
        FLAGS.data_file,
        FLAGS.validation_percentage,
        FLAGS.testing_percentage,
        FLAGS.data_aug_columns
    )

    # print size of training, validation and testing
    tf.logging.info("")
    tf.logging.info("************** DataBase Length **************")
    tf.logging.info("Training length: " +
                    str(audio_processor.get_size_by_index('training')))
    tf.logging.info("Validation length: " +
                    str(audio_processor.get_size_by_index('validation')))
    tf.logging.info("Testing length: " +
                    str(audio_processor.get_size_by_index('testing')))

    # Input
    fingerprint_input = tf.placeholder(
        tf.float32, [None, model_settings['fingerprint_size']], name='fingerprint_input')
    gt_input = tf.placeholder(tf.float32, [None, 1], name='groundtruth_input')

    # Model
    estimator, phase_train = models.create_model(
        fingerprint_input, model_settings, FLAGS.model_architecture)

    # Loss
    loss = apply_loss(gt_input, estimator)
    tf.summary.scalar('rmse', loss)

    # Create the back-propagation and training evaluation machinery in the graph.
    with tf.name_scope('train'), tf.control_dependencies([]):
        learning_rate_input = tf.placeholder(
            tf.float32, [], name='learning_rate_input')
        optimizer = apply_optimizer(learning_rate_input)
        train_step = optimizer.minimize(loss)

    global_step = tf.train.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)
    saver = tf.train.Saver(tf.global_variables())

    # Merge all the summaries and create file writers
    merged_summaries = tf.summary.merge_all()
    output_dir = create_output_path(FLAGS.output_dir, config_name)

    train_writer = tf.summary.FileWriter(
        output_dir + '/summary/train', sess.graph)
    val_writer = tf.summary.FileWriter(
        output_dir + '/summary/validation')

    if len(FLAGS.training_steps) != len(FLAGS.learning_rate):
        raise Exception(
            '--training_steps and --learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(FLAGS.training_steps),
                                                       len(FLAGS.training_steps)))

    tf.global_variables_initializer().run()
    start_step = 1

    tf.logging.info('"***************** Training *****************')
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
        train_names, train_fingerprints, train_gt_scores = audio_processor.get_data_by_index(
            FLAGS.batch_size, 0, 'training', sess)

        # Run the graph with this batch of training data.
        train_summary, train_rmse, _, _ = sess.run(
            [
                merged_summaries,
                loss,
                train_step,
                increment_global_step
            ],
            feed_dict={
                fingerprint_input: train_fingerprints,
                gt_input: train_gt_scores,
                learning_rate_input: learning_rate_value,
                phase_train: True
            })

        tf.logging.info('step #%d: rate %f, rmse %f' %
                        (training_step, learning_rate_value, train_rmse))

        if (training_step % FLAGS.summary_step_interval) == 0:
            train_writer.add_summary(train_summary, training_step)

        if (training_step % FLAGS.eval_step_interval) == 0 or (training_step == training_steps_max):
            tf.logging.info('***************** Validation ****************')

            set_size = audio_processor.get_size_by_index('validation')
            total_rmse = 0
            weights = np.array([], dtype=np.float32)
            values = np.array([], dtype=np.float32)

            for i in range(0, set_size, FLAGS.batch_size):
                _, val_fingerprints, val_ground_truth = audio_processor.get_data_by_index(
                    FLAGS.batch_size, i, 'validation', sess)

                validation_summary, validation_rmse = sess.run(
                    [
                        merged_summaries,
                        loss
                    ],
                    feed_dict={
                        fingerprint_input: val_fingerprints,
                        gt_input: val_ground_truth,
                        phase_train: False
                    })

                weights = np.append(
                    weights, val_fingerprints.shape[0] / set_size)
                values = np.append(values, validation_rmse)
                tf.logging.info('i=%d: rmse = %.2f' % (i, validation_rmse))

            weighted_rmse = np.dot(values, weights)
            weighted_rmse_summary = tf.Summary(
                value=[tf.Summary.Value(tag='rmse', simple_value=weighted_rmse)])
            val_writer.add_summary(
                weighted_rmse_summary, training_step)
            tf.logging.info('weighted rmse = %.2f (N=%d)' %
                            (weighted_rmse, set_size))
            
            if training_step < training_steps_max:
              tf.logging.info('"***************** Training *****************')
      
    tf.logging.info('****************** Testing ******************')
    set_size = audio_processor.get_size_by_index('testing')
    weights = np.array([], dtype=np.float32)
    values = np.array([], dtype=np.float32)

    for i in range(0, set_size, FLAGS.batch_size):
        _, test_fingerprints, test_ground_truth = audio_processor.get_data_by_index(
            FLAGS.batch_size, i, 'testing', sess)

        testing_summary, test_rmse = sess.run(
            [
                merged_summaries,
                loss
            ],
            feed_dict={
                fingerprint_input: test_fingerprints,
                gt_input: test_ground_truth,
                phase_train: False
            })

        weights = np.append(weights, test_fingerprints.shape[0] / set_size)
        values = np.append(values, test_rmse)
        tf.logging.info('i=%d: rmse = %.2f' % (i, test_rmse))

    weighted_rmse = np.dot(values, weights)
    tf.logging.info('weighted rmse = %.2f (N=%d)' %
                    (weighted_rmse, set_size))
    tf.logging.info('***************** ********* *****************')

    # Save cross validation partitions 
    utils.create_dir(output_dir + '/cross_val_sets')
    rows = audio_processor.get_index('training')
    if(rows):
      utils.save_dict_as_csv(
        output_dir + '/cross_val_sets/training.csv', ',', rows[0].keys(), rows)
    rows = audio_processor.get_index('validation')
    if(rows):
      utils.save_dict_as_csv(
          output_dir + '/cross_val_sets/validation.csv', ',', rows[0].keys(), rows)
    rows = audio_processor.get_index('testing')
    if(rows):
      utils.save_dict_as_csv(
          output_dir + '/cross_val_sets/testing.csv', ',', rows[0].keys(), rows)

    # Save configuration
    config.save(output_dir, FLAGS.__dict__)

    # Save the model
    if FLAGS.enable_checkpoint_save:
        utils.create_dir(output_dir + '/checkpoint')
        FLAGS.start_checkpoint = os.path.join(
            output_dir + '/checkpoint', FLAGS.model_architecture + '.ckpt')
        tf.logging.info('Saving to "%s-%d"',
                        FLAGS.start_checkpoint, training_steps_max)
        saver.save(sess, FLAGS.start_checkpoint,
                   global_step=training_steps_max)
        FLAGS.start_checkpoint += '-' + str(training_steps_max)

    train_writer.close()
    val_writer.close()
    sess.close()

if __name__ == '__main__':
    FLAGS, _unparsed = config.set()
    tf.app.run(main=main, argv=[FLAGS])
