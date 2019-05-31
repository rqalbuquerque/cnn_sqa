"""Testing the model on new samples.

"""
import re
import csv
import sys
import os

import numpy as np
import tensorflow as tf

import utils
import config
import input_data
import models


def main(argv):
    # Get flags
    [FLAGS, config_path, input_path, output_path] = argv

    # To see all the logging messages
    tf.logging.set_verbosity(tf.logging.INFO)

    # Start a new TensorFlow session.
    sess = tf.InteractiveSession()

    # Create model settings.
    model_settings = models.prepare_model_settings(
        FLAGS.enable_hist_summary,
        FLAGS.input_processing_lib,
        FLAGS.sample_rate,
        FLAGS.clip_duration_ms,
        FLAGS.window_size_ms,
        FLAGS.window_stride_ms,
        FLAGS.data_aug_algorithms,
        FLAGS.feature,
        FLAGS.n_coeffs,
        FLAGS.conv_layers,
        FLAGS.filter_width,
        FLAGS.filter_height,
        FLAGS.filter_count,
        FLAGS.stride,
        FLAGS.apply_batch_norm,
        FLAGS.activation,
        FLAGS.kernel_regularizer,
        FLAGS.apply_dropout,
        FLAGS.fc_layers,
        FLAGS.hidden_units)

    audio_processor = input_data.TFAudioProcessor(model_settings, sess)
    audio_processor.index_from_dir(input_path, '.wav')

    tf.logging.info('***************** Testing *****************')
    tf.logging.info('Testing on config: ' + config_path)
    tf.logging.info('Checkpoint path: ' + FLAGS.start_checkpoint)
    tf.logging.info('Database path: ' + input_path)
    tf.logging.info("Database length: " + str(audio_processor.get_size()))
    tf.logging.info('***************** ******** *****************')

    # input
    fingerprint_input = tf.placeholder(
        tf.float32, [None, model_settings['fingerprint_size']], name='fingerprint_input')

    # model
    estimator, phase_train = models.create_model(
        fingerprint_input, model_settings, FLAGS.model_architecture)

    # Merge all the summaries and create file writers
    merged_summaries = tf.summary.merge_all()

    tf.global_variables_initializer().run()
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    
    utils.create_dir(output_path)

    with open(output_path + '/scores.csv', 'wb') as csvfile:
        names, scores = [], []
        csv_writer = csv.DictWriter(csvfile, fieldnames=['Name', 'Score'])
        set_size = audio_processor.get_size()

        for i in range(0, set_size, FLAGS.batch_size):
            testing_names, testing_fingerprints = (
                audio_processor.get_data(FLAGS.batch_size, i))

            testing_summary, testing_scores = sess.run(
                [
                    merged_summaries,
                    estimator
                ],
                feed_dict={
                    fingerprint_input: testing_fingerprints,
                    phase_train: False
                })

            names += testing_names
            scores += testing_scores.flatten().tolist()
            tf.logging.info('Running on batch: ' + str(i))

        csv_writer.writeheader()
        for i in range(0, len(names)):
            csv_writer.writerow({'Name': names[i], 'Score': str(scores[i])})

    tf.logging.info('***************** ******** *****************')
    sess.close()


if __name__ == '__main__':
    if len(sys.argv) == 4:
        config_path = sys.argv[1]
        input_path = sys.argv[2]
        output_path = sys.argv[3]
        FLAGS, _ = config.set(config.read(config_path))
        tf.app.run(main=main, argv=[FLAGS, config_path, input_path, output_path])
    else:
        raise ValueError('Invalid number of args!')
