import os
import csv
import tensorflow as tf

from tensorflow.python.platform import test
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

from src import input_data
from src import models

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

  def _basicConfig(self, qty_samples):
    tmp_dir = self.get_temp_dir()
    database = os.path.join(tmp_dir, "database")
    os.mkdir(database)
    self._saveWavFolders(database, ["a", "b", "c"], qty_samples)
    self._saveCsvFile(database, ["a", "b", "c"], qty_samples)
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

  def testPrepareDataIndex(self):
    database, _ = self._basicConfig(1)
    expected_index = [
      {'file': database + "/a/some_audio_0.wav"},
      {'file': database + "/b/some_audio_0.wav"},
      {'file': database + "/c/some_audio_0.wav"}
    ]
    audio_processor = input_data.AudioProcessor(self._model_settings())
    audio_processor.prepare_data_index(database, ".wav")

    self.assertEqual(3, audio_processor.get_size())
    self.assertIn(expected_index[0], audio_processor.get_index())
    self.assertIn(expected_index[1], audio_processor.get_index())
    self.assertIn(expected_index[2], audio_processor.get_index())


  def testPrepareDataIndexPartitioned(self):
    database, csv_path = self._basicConfig(2)
    full_index = [
      {'file': database + "/a/some_audio_0.wav", 'score': 1.0},
      {'file': database + "/a/some_audio_1.wav", 'score': 2.0},
      {'file': database + "/b/some_audio_0.wav", 'score': 1.0},
      {'file': database + "/b/some_audio_1.wav", 'score': 2.0},
      {'file': database + "/c/some_audio_0.wav", 'score': 1.0},
      {'file': database + "/c/some_audio_1.wav", 'score': 2.0}
    ]
    audio_processor = input_data.AudioProcessor(self._model_settings())
    audio_processor.prepare_data_index_from_csv_file(csv_path, 0.33, 0)
    training_index = audio_processor.get_index_by_partition('training')
    validation_index = audio_processor.get_index_by_partition('validation')

    self.assertEqual(4, audio_processor.get_size_by_partition('training'))
    self.assertEqual(2, audio_processor.get_size_by_partition('validation'))
    self.assertTrue("training" in audio_processor.data_index)
    self.assertTrue("validation" in audio_processor.data_index)
    self.assertIn(full_index[0], training_index + validation_index)
    self.assertIn(full_index[1], training_index + validation_index)
    self.assertIn(full_index[2], training_index + validation_index)
    self.assertIn(full_index[3], training_index + validation_index)
    self.assertIn(full_index[4], training_index + validation_index)
    self.assertIn(full_index[5], training_index + validation_index)
        

  def testGetData(self):
    database, _ = self._basicConfig(10)

    with self.cached_session() as sess:
      audio_processor = input_data.AudioProcessor(self._model_settings())
      audio_processor.prepare_data_index(database, ".wav")

      result_names, result_data = audio_processor.get_data(5, 5, sess)
      self.assertEqual(5, len(result_data))
      self.assertEqual(5, len(result_names))

      result_names, result_data = audio_processor.get_data(-1, 0, sess)
      self.assertEqual(30, len(result_data))
      self.assertEqual(30, len(result_names))


  def testGetDataByPartition(self):
    database, csv_path = self._basicConfig(10)

    with self.cached_session() as sess:
      audio_processor = input_data.AudioProcessor(self._model_settings())

      audio_processor.prepare_data_index_from_csv_file(csv_path, 0.5, 0)
      result_names, result_data, result_scores = audio_processor.get_data_by_partition(5, 3, "validation", sess)
      self.assertEqual(5, len(result_data))
      self.assertEqual(5, len(result_names))
      self.assertEqual(5, len(result_scores))

      audio_processor.prepare_data_index_from_csv_file(csv_path, 0.99, 0)
      result_names, result_data, result_scores = audio_processor.get_data_by_partition(-1, 1, "validation", sess)
      self.assertEqual(30, len(result_data))
      self.assertEqual(30, len(result_names))
      self.assertEqual(30, len(result_scores))

  
if __name__ == '__main__':
  tf.test.main()
