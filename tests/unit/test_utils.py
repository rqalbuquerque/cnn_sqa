import unittest

from src import utils

class TestAddSuffixInFilename(unittest.TestCase):
  def test_insert_with_extension(self):
    """
    Test with file name with extension.
    """
    expected = './test/file_suffix.txt'
    result = utils.add_suffix_in_filename('./test/file.txt', '_suffix')
    self.assertEqual(expected, result)

  def test_insert_without_extension(self):
    """
    Test with file name without extension.
    """
    expected = './test/file_suffix'
    result = utils.add_suffix_in_filename('./test/file', '_suffix')
    self.assertEqual(expected, result)

class TestReadJsonAsDict(unittest.TestCase):
  def test_nonexistent_file(self):
    """
    Test with nonexistent file.
    """
    with self.assertRaises(IOError):
      result = utils.read_json_as_dict("tests/fixtures/nonexistent.file")

  def test_invalid_file(self):
    """
    Test with invalid file type.
    """
    with self.assertRaises(ValueError):
      result = utils.read_json_as_dict("tests/fixtures/test_basic.py")

  def test_valid_json(self):
    """
    Test with valid json file.
    """
    expected = {"str": "test", "int": 123, "float": 1.24, "list": [1,2,3], "dict": {"key": "value"}}
    result = utils.read_json_as_dict("tests/fixtures/test_basic.json")
    self.assertEqual(expected, result)

class TestReadCsvAsDict(unittest.TestCase):
  def test_nonexistent_file(self):
    """
    Test with nonexistent file.
    """
    with self.assertRaises(IOError):
      result = utils.read_csv_as_dict("tests/fixtures/nonexistent.file", ',')

  def test_valid_file(self):
    """
    Test with valid csv file.
    """
    expected = [
      {'value3': None, 'value2': None, 'value1': 'test', 'label': 'str'}, 
      {'value3': None, 'value2': None, 'value1': '123', 'label': 'int'}, 
      {'value3': None, 'value2': None, 'value1': ' 1.24', 'label': 'float'}, 
      {'value3': '3', 'value2': '2', 'value1': '1', 'label': 'list'}, 
      {'value3': None, 'value2': 'value', 'value1': 'key', 'label': 'dict'}
    ]
    result = utils.read_csv_as_dict("tests/fixtures/test_basic.csv", ',')
    self.assertEqual(expected, result)


class TestFindByExtension(unittest.TestCase):
  def test_not_found_extension(self):
    """
    Test without files by extension.
    """
    result = utils.find_by_extension("tests/fixtures/test_basic", "not_found_extension")
    self.assertEqual([], result)

  def test_valid_dir(self):
    """
    Test with valid directory.
    """
    expected = ['test_basic.json']
    result = utils.find_by_extension("tests/fixtures", "json")
    self.assertEqual(expected, result)

if __name__ == '__main__':
  unittest.main()
