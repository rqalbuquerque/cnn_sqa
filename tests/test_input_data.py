import os
import csv
import random
import tensorflow as tf

from tensorflow.python.platform import test
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

import input_data
import models

class InputDataTest(test.TestCase):
  def _getWavData(self):
    with self.cached_session() as sess:
      sample_data = tf.zeros([128000, 2])
      wav_encoder = contrib_audio.encode_wav(sample_data, 16000)
      wav_data = self.evaluate(wav_encoder)
    return wav_data

  def _saveTestWavFile(self, filename, wav_data):
    with open(filename, "wb") as f:
      f.write(wav_data)

  def _saveWavFolders(self, root_dir, labels, how_many, prefix=''):
    wav_data = self._getWavData()
    for label in labels:
      dir_name = os.path.join(root_dir, label)
      os.mkdir(dir_name)
      for i in range(how_many):
        file_path = os.path.join(dir_name, prefix + "some_audio_%d.wav" % i)
        self._saveTestWavFile(file_path, wav_data)

  def _saveCsv(self, root_dir, labels, how_many):
    with open(root_dir + 'scores.csv', mode='w') as csv_file:
      writer = csv.DictWriter(csv_file, fieldnames=['file', 'score'])
      writer.writeheader()
      for label in labels:
        for i in range(how_many):
          file_path = os.path.join(label, "some_audio_%d.wav" % i)
          sample = {'file': file_path, 'score': float(i % 5 + 1)}
          writer.writerow(sample)
  
  def _addAugDataToCsv(self, root_dir, labels, how_many):
    aug_samples = {label: [] for label in labels}
    for label in labels:
      for i in range(how_many):
        aug_samples[label].append(os.path.join(label, 'aug_some_audio_%d.wav' % i))
    samples = input_data.load_samples_from_csv(os.path.join(root_dir, 'scores.csv'))
    for sample in samples:
      for label in labels: 
        sample.update({label: aug_samples[label].pop(0)})
    with open(root_dir + 'scores.csv', mode='w') as csv_file:
      writer = csv.DictWriter(csv_file, fieldnames=['file', 'score'] + labels)
      writer.writeheader()
      writer.writerows(samples)

  def _basicConfig(self, labels, qty_samples):
    tmp_dir = self.get_temp_dir()
    database = os.path.join(tmp_dir, "database/")
    os.mkdir(database)
    self._saveWavFolders(database, labels, qty_samples)
    self._saveCsv(database, labels, qty_samples)
    csv_path = os.path.join(database, "scores.csv")
    return database, csv_path

  def _model_settings(self):
    return {
      'fingerprint_size': 2360, 
      'n_coeffs': 40, 
      'sample_rate': 16000, 
      'desired_samples': 128000, 
      'window_stride_samples': 2048, 
      'window_size_samples': 8192 
    }

  def testIndexFromDir(self):
    database, _ = self._basicConfig(["a", "b", "c"], 1)
    expected_index = [
      {'file': database + "/a/some_audio_0.wav"},
      {'file': database + "/b/some_audio_0.wav"},
      {'file': database + "/c/some_audio_0.wav"}
    ]
    audio_processor = input_data.AudioProcessor(self._model_settings())
    audio_processor.index_from_dir(database)

    self.assertEqual(3, audio_processor.get_size())
    self.assertIn(expected_index[0], audio_processor.get_samples())
    self.assertIn(expected_index[1], audio_processor.get_samples())
    self.assertIn(expected_index[2], audio_processor.get_samples())


  def testIndexFromCSV(self):
    database, csv_path = self._basicConfig(["a", "b", "c"], 2)

    full_index = [
      {'file': database + "a/some_audio_0.wav", 'score': 1.0},
      {'file': database + "a/some_audio_1.wav", 'score': 2.0},
      {'file': database + "b/some_audio_0.wav", 'score': 1.0},
      {'file': database + "b/some_audio_1.wav", 'score': 2.0},
      {'file': database + "c/some_audio_0.wav", 'score': 1.0},
      {'file': database + "c/some_audio_1.wav", 'score': 2.0},
    ]
    audio_processor = input_data.AudioProcessor(self._model_settings())
    audio_processor.index_from_csv(database, csv_path, 0.33, 0.0)
    training_index = audio_processor.get_index('training')
    validation_index = audio_processor.get_index('validation')

    self.assertTrue("training" in audio_processor.data)
    self.assertTrue("validation" in audio_processor.data)
    self.assertTrue("testing" in audio_processor.data)
    self.assertEqual(4, audio_processor.get_size_by_index('training'))
    self.assertEqual(2, audio_processor.get_size_by_index('validation'))
    self.assertEqual(0, audio_processor.get_size_by_index('testing'))
    self.assertIn(full_index[0], training_index + validation_index)
    self.assertIn(full_index[1], training_index + validation_index)
    self.assertIn(full_index[2], training_index + validation_index)
    self.assertIn(full_index[3], training_index + validation_index)
    self.assertIn(full_index[4], training_index + validation_index)
    self.assertIn(full_index[5], training_index + validation_index)


  def testIndexWithDataAugmentation(self):
    database, csv_path = self._basicConfig(["a", "b"], 2)
    self._saveWavFolders(database, ["aug_1", "aug_2"], 4, "aug_")
    self._addAugDataToCsv(database, ["aug_1", "aug_2"], 4)

    full_index = [
        {'file': database + "a/some_audio_0.wav", 'score': 1.0},
        {'file': database + "a/some_audio_1.wav", 'score': 2.0},
        {'file': database + "b/some_audio_0.wav", 'score': 1.0},
        {'file': database + "b/some_audio_1.wav", 'score': 2.0},
        {'file': database + "aug_1/aug_some_audio_0.wav", 'score': 1.0},
        {'file': database + "aug_1/aug_some_audio_1.wav", 'score': 2.0},
        {'file': database + "aug_1/aug_some_audio_2.wav", 'score': 1.0},
        {'file': database + "aug_1/aug_some_audio_3.wav", 'score': 2.0},
        {'file': database + "aug_2/aug_some_audio_0.wav", 'score': 1.0},
        {'file': database + "aug_2/aug_some_audio_1.wav", 'score': 2.0},
        {'file': database + "aug_2/aug_some_audio_2.wav", 'score': 1.0},
        {'file': database + "aug_2/aug_some_audio_3.wav", 'score': 2.0}
    ]
    audio_processor = input_data.AudioProcessor(self._model_settings())
    audio_processor.index_from_csv(
        database, csv_path, 0.0, 0.0, ['aug_1', 'aug_2'])
    training_index = audio_processor.get_index('training')
    validation_index = audio_processor.get_index('validation')

    self.assertTrue("training" in audio_processor.data)
    self.assertEqual(12, audio_processor.get_size_by_index('training'))
    self.assertIn(full_index[0], training_index)
    self.assertIn(full_index[1], training_index)
    self.assertIn(full_index[2], training_index)
    self.assertIn(full_index[3], training_index)
    self.assertIn(full_index[4], training_index)
    self.assertIn(full_index[5], training_index)
    self.assertIn(full_index[7], training_index)
    self.assertIn(full_index[8], training_index)
    self.assertIn(full_index[9], training_index)
    self.assertIn(full_index[10], training_index)
    self.assertIn(full_index[11], training_index)


  def testGetData(self):
    database, _ = self._basicConfig(["a", "b", "c"], 10)

    with self.cached_session() as sess:
      audio_processor = input_data.AudioProcessor(self._model_settings())
      audio_processor.index_from_dir(database)

      result_names, result_data = audio_processor.get_data(5, 5, sess)
      self.assertEqual(5, len(result_data))
      self.assertEqual(5, len(result_names))

      result_names, result_data = audio_processor.get_data(-1, 0, sess)
      self.assertEqual(30, len(result_data))
      self.assertEqual(30, len(result_names))


  def testGetDataByIndex(self):
    database, csv_path = self._basicConfig(["a", "b", "c"], 10)

    with self.cached_session() as sess:
      audio_processor = input_data.AudioProcessor(self._model_settings())

      audio_processor.index_from_csv(database, csv_path, 0.5, 0)
      result_names, result_data, result_scores = audio_processor.get_data_by_index(5, 3, "validation", sess)
      self.assertEqual(5, len(result_data))
      self.assertEqual(5, len(result_names))
      self.assertEqual(5, len(result_scores))

      audio_processor.index_from_csv(database, csv_path, 0.99, 0)
      result_names, result_data, result_scores = audio_processor.get_data_by_index(-1, 1, "validation", sess)
      self.assertEqual(30, len(result_data))
      self.assertEqual(30, len(result_names))
      self.assertEqual(30, len(result_scores))

  
if __name__ == '__main__':
  tf.test.main()
