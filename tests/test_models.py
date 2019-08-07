import tensorflow as tf
from tensorflow.python.platform import test
import models

class ModelsTest(test.TestCase):
  def _modelSettings(self):
    return models.prepare_model_settings(
            sample_rate=16000,
            clip_duration_ms=1000,
            window_size_ms=20,
            window_stride_ms=10,
            feature='mfcc',
            n_coeffs=40,
            filter_width=[1,2,3,4,5],
            filter_height=[1,2,3,4,5],
            filter_count=[2,2,2,2,2],
            stride=[1,2,2,2,2],
            apply_batch_norm=True,
            activation='relu',
            apply_dropout=True,
            hidden_units=[5,5]
          )

  def testPrepareModelSettings(self):
    model_settings = self._modelSettings()
    expected_settings = {
        'desired_samples': 16000,
        'window_size_samples': 320,
        'window_stride_samples': 160,
        'fingerprint_size': 3960,
        'sample_rate': 16000,
        'n_frames': 99,
        'feature': 'mfcc',
        'n_coeffs': 40,
        'filter_width': [1,2,3,4,5],
        'filter_height': [1,2,3,4,5],
        'filter_count': [2,2,2,2,2],
        'stride': [1,2,2,2,2],
        'apply_batch_norm': True,
        'activation': 'relu',
        'apply_dropout': True,
        'hidden_units': [5,5]
      }
    self.assertEqual(expected_settings, model_settings)

  # def testBatchNormalization(self):
  #   input_tensor = tf.constant([[[[1.], [2.], [3.], [4.], [5.]],
  #                                [[1.], [2.], [3.], [4.], [5.]],
  #                                [[1.], [2.], [3.], [4.], [5.]]]])
  #   phase_train = tf.placeholder(tf.bool, name='phase_train')
  #   batch_norm = models.batch_normalization(input_tensor, 1, phase_train)
  #   with self.cached_session() as sess:
  #     sess.run(tf.global_variables_initializer())

  #     print(list(sess.run(batch_norm,feed_dict={phase_train: True})))

  #     print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'batch_norm/beta:0')[0].eval(sess))
  #     print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'batch_norm/gamma:0')[0].eval(sess))
      
  #     batch_mean, batch_var = tf.nn.moments(input_tensor, [0, 1, 2], name='moments')
  #     ema = tf.train.ExponentialMovingAverage(decay=0.99)
      
  #     mean, var = ema.average(batch_mean), ema.average(batch_var)

  #     print(sess.run([mean, var]))
  #     print("mean: " + str(batch_mean))
  #     print("var: " + str(batch_var))

  def testActivation(self):
    const = tf.constant([-2.5])
    activation = models.activation(const, "relu")
    self.assertEqual(activation.name, "relu:0")
    self.assertEqual(activation.shape, (1,))
    self.assertEqual(activation.dtype, tf.float32)

  def testConv2d(self):
    input_tensor = tf.constant([[[[1.],[2.],[3.],[4.],[5.]],
                          [[1.],[2.],[3.],[4.],[5.]],
                          [[1.],[2.],[3.],[4.],[5.]]]])
    expected = [[[[6., 6.], [14., 14.], [10., 10.]],
                 [[3., 3.], [7., 7.], [5., 5.]]]]
    with test.mock.patch('tensorflow.truncated_normal') as mock_truncated_normal:
      mock_truncated_normal.return_value = tf.ones((2, 2, 1, 2))
      conv2d = models.conv_2d(input_tensor, 2, 2, 1, 2, 2)
      with self.cached_session() as sess:
        sess.run(tf.global_variables_initializer())
        self.assertEqual(sess.graph.get_tensor_by_name('Const:0').shape, [1, 3, 5, 1])
        self.assertEqual(sess.graph.get_tensor_by_name('conv/weights:0').shape, [2, 2, 1, 2])
        self.assertEqual(sess.graph.get_tensor_by_name('conv/biases:0').shape, [2,])
        self.assertEqual(sess.graph.get_tensor_by_name('conv/convolution:0').shape, [1, 2, 3, 2])
        self.assertEqual(sess.graph.get_tensor_by_name('conv/sum:0').shape, [1, 2, 3, 2])
        self.assertTrue((expected == sess.run(conv2d)).all())
      
  def testCreateModelConv(self):
    model_settings = self._modelSettings()
    fingerprint_input = tf.zeros(
        [1, model_settings["fingerprint_size"]], name='fingerprint_input')
    estimator, phase_train = models.create_model(fingerprint_input, model_settings, "conv")
    with self.cached_session() as sess:
      self.assertIsNotNone(sess.graph.get_tensor_by_name(estimator.name))
      self.assertIsNotNone(sess.graph.get_tensor_by_name(phase_train.name))
      self.assertIsNotNone(sess.graph.get_tensor_by_name('input:0'))

      self.assertEqual(sess.graph.get_tensor_by_name('fingerprint_input:0').shape, [1, 3960])
      self.assertEqual(sess.graph.get_tensor_by_name('Reshape:0').shape, [1, 40, 99, 1])

      self.assertEqual(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv_0/conv/weights:0')[0].shape, [1,1,1,2])
      self.assertEqual(len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv_0/batch_norm')), 2)
      self.assertEqual(sess.graph.get_tensor_by_name('conv_0/relu:0').shape, [1,40,99,2])
      self.assertIsNotNone(sess.graph.get_operation_by_name('conv_0/dropout/cond/dropout/Shape'))
      
      self.assertEqual(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv_1/conv/weights:0')[0].shape, [2,2,2,2])
      self.assertEqual(len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv_1/batch_norm')), 2)
      self.assertEqual(sess.graph.get_tensor_by_name('conv_1/relu:0').shape, [1,20,50,2])
      self.assertIsNotNone(sess.graph.get_operation_by_name('conv_1/dropout/cond/dropout/Shape'))

      self.assertEqual(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv_2/conv/weights:0')[0].shape, [3,3,2,2])
      self.assertEqual(len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv_2/batch_norm')), 2)
      self.assertEqual(sess.graph.get_tensor_by_name('conv_2/relu:0').shape, [1,10,25,2])
      self.assertIsNotNone(sess.graph.get_operation_by_name('conv_2/dropout/cond/dropout/Shape'))

      self.assertEqual(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv_3/conv/weights:0')[0].shape, [4,4,2,2])
      self.assertEqual(len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv_3/batch_norm')), 2)
      self.assertEqual(sess.graph.get_tensor_by_name('conv_3/relu:0').shape, [1,5,13,2])
      self.assertIsNotNone(sess.graph.get_operation_by_name('conv_3/dropout/cond/dropout/Shape'))

      self.assertEqual(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv_4/conv/weights:0')[0].shape, [5,5,2,2])
      self.assertEqual(len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv_4/batch_norm')), 2)
      self.assertEqual(sess.graph.get_tensor_by_name('conv_4/relu:0').shape, [1,3,7,2])
      self.assertIsNotNone(sess.graph.get_operation_by_name('conv_4/dropout/cond/dropout/Shape'))

      self.assertEqual(sess.graph.get_tensor_by_name('flatten:0').shape, [1, 42])

      self.assertEqual(len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dense_0')), 2)
      self.assertEqual(len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dense_1')), 2)
      self.assertEqual(len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'estimator')), 2)

