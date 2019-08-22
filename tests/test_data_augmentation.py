import mock
import unittest
import numpy

import data_augmentation

class TestFlip(unittest.TestCase):
  def test_flip_empty_list(self):
    """
    Test with empty list.
    """
    result = data_augmentation.flip([])
    self.assertEqual(list(result), [])

  def test_flip(self):
    """
    Test with lists.
    """  
    test_input = [4.0,5,6]
    expected = [6,5,4.0]
    result = data_augmentation.flip(test_input)
    self.assertEqual(result,expected)


class TestRandomCircularShift(unittest.TestCase):
  def test_random_circular_shift_empty_list(self):
    """
    Test with an empty list.
    """
    expected = []
    with self.assertRaises(ValueError):
      result = data_augmentation.rcs([])

  def test_random_circular_shift_bad_type(self):
    """
    Test with invalid data type.
    """
    self.assertEqual("test", data_augmentation.rcs("test"))

  def test_random_circular_shift_list(self):
    """
    Test with valid list.
    """
    data = [1,2,3,4,5,6,7]
    expected = [4,5,6,7,1,2,3]
    numpy.random.randint = mock.Mock(return_value=[3])
    result = data_augmentation.rcs(data)
    self.assertEqual(result,expected)

if __name__ == '__main__':
  unittest.main()
