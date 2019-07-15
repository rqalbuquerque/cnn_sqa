import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.python.platform import gfile
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

import datetime
import os
import csv

from src import input_data
from src import models
from src import train
from src import utils


class DictStruct(object):
  def __init__(self, **entries):
    self.__dict__.update(entries)

class TrainTest(test.TestCase):
  def _getWavData(self):
    with self.cached_session() as sess:
      sample_data = tf.zeros([128000, 2])
      wav_encoder = contrib_audio.encode_wav(sample_data, 16000)
      wav_data = self.evaluate(wav_encoder)
    return wav_data

  def _saveTestWavFile(self, filename, wav_data):
    with open(filename, "wb") as f:
      f.write(wav_data)

  def _saveWavFolders(self, root_dir, labels, how_many):
    wav_data = self._getWavData()
    for label in labels:
      dir_name = os.path.join(root_dir, label)
      os.mkdir(dir_name)
      for i in range(how_many):
        file_path = os.path.join(dir_name, "some_audio_%d.wav" % i)
        self._saveTestWavFile(file_path, wav_data)

  def _saveCsvFile(self, root_dir, labels, how_many):
    with open(root_dir + '/scores.csv', mode='w') as csv_file:
      writer = csv.DictWriter(csv_file, fieldnames=['file', 'score'])
      writer.writeheader()
      for label in labels:
        dir_name = os.path.join(root_dir, label)
        for i in range(how_many):
          file_path = os.path.join(dir_name, "some_audio_%d.wav" % i)
          writer.writerow({'file': file_path, 'score': float(i % 5 + 1)})

  def _getDefaultFlags(self, input_dir, output_dir):
    flags = {
        'data_dir': input_dir,
        'output_dir': output_dir,
        'enable_checkpoint_save': True,
        'testing_percentage': 10,
        'validation_percentage': 10,
        'batch_size': 3,
        'training_steps': [10, 5],
        'learning_rate': [0.01, 0.005],
        'sample_rate': 16000,
        'clip_duration_ms': 8000,
        'data_aug_algorithms': ['flip', 'random_circular_shift'],
        'window_size_ms': 32.0,
        'window_stride_ms': 8.0,
        'feature': 'mfcc',
        'n_coeffs': 40,
        'model_architecture': 'conv',
        'filter_width': [2, 3, 5, 7, 9],
        'filter_height': [2, 3, 5, 7, 9],
        'filter_count': [2, 2, 2, 2, 2],
        'stride': [1, 2, 2, 2, 2],
        'apply_batch_norm': True,
        'apply_dropout': True,
        'activation': 'relu',
        'hidden_units': [5, 5],
        'summary_step_interval': 5,
        'eval_step_interval': 10,
        'start_checkpoint': ''
        }
    return DictStruct(**flags)

  def testApplyLoss(self):
    gt = tf.constant([1.0,2.0,3.0,4.0])
    estimated = tf.constant([4.0,3.0,2.0,1.0])
    expected = 2.23606797749979
    loss = train.apply_loss(gt, estimated)
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      rmse = sess.run(loss)
      self.assertNear(expected, rmse, 0.0001)

  def testGenAugmentedData(self):
    names = ['f1.wav', 'f2.wav']
    data = [[1, 2, 3, 4, 5], [4, 3, 2, 1, 0]]
    scores = [1.0, 2.0]
    modes = ['flip', 'random_circular_shift']

    aug_names, aug_data, aug_scores = train.gen_augmented_data(
        names, data, scores, modes)

    self.assertEqual(['f1.wav', 
                      'f2.wav',
                      'f1_augmented_by_flip.wav', 
                      'f2_augmented_by_flip.wav',
                      'f1_augmented_by_random_circular_shift.wav', 
                      'f2_augmented_by_random_circular_shift.wav'], aug_names)
    self.assertEqual([sorted(data) for data in aug_data],
                     [[1, 2, 3, 4, 5], 
                      [0, 1, 2, 3, 4],
                      [1, 2, 3, 4, 5], 
                      [0, 1, 2, 3, 4], 
                      [1, 2, 3, 4, 5], 
                      [0, 1, 2, 3, 4]])
    self.assertEqual([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], aug_scores)

  def testTrain(self):
    tmp_dir = self.get_temp_dir()
    database = os.path.join(tmp_dir, "database")
    output_dir = os.path.join(tmp_dir, "output")
    os.mkdir(database)
    os.mkdir(output_dir)
    self._saveWavFolders(database, ["a", "b", "c"], 10)
    self._saveCsvFile(database, ["a", "b", "c"], 10)
    csv_path = os.path.join(database, "scores.csv")
    flags = self._getDefaultFlags(database, output_dir)
    train.main([flags, 'test'])

    self.assertTrue(gfile.Exists(
        os.path.join(output_dir, 'run-test/config.json')))
    self.assertTrue(gfile.Exists(
        os.path.join(output_dir, 'run-test/checkpoint/checkpoint')))
    self.assertTrue(gfile.Exists(
        os.path.join(output_dir, 'run-test/checkpoint/' + flags.model_architecture + '.ckpt-15.meta')))
    self.assertTrue(gfile.Exists(
        os.path.join(output_dir, 'run-test/checkpoint/' + flags.model_architecture + '.ckpt-15.data-00000-of-00001')))
    self.assertTrue(gfile.Exists(
        os.path.join(output_dir, 'run-test/checkpoint/' + flags.model_architecture + '.ckpt-15.index')))
    self.assertTrue(os.path.exists(os.path.join(output_dir, 'run-test/summary/train')))
    self.assertTrue(os.path.exists(os.path.join(output_dir, 'run-test/summary/validation')))

if __name__ == '__main__':
  tf.test.main()
