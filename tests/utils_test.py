import unittest
import os
import tempfile
import json
import csv

from src.utils import *

def _saveFile(name, content):
  tmp_file = tempfile.gettempdir() + '/' + name
  with open(tmp_file, 'w') as file:
    file.write(content)
    file.close()
  return tmp_file

def _saveCsv(name, fieldnames, dict_data):
  tmp_file = tempfile.gettempdir() + '/' + name
  with open(tmp_file, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for data in dict_data:
        writer.writerow(data)
  return tmp_file

def _loadCsv(csv_path, delimiter, fieldnames):
  data = []
  with open(csv_path) as csv_file:
    reader = csv.DictReader(
        csv_file, delimiter=delimiter, fieldnames=fieldnames)
    data = [row for row in reader]
  return data

def _getTempPath(file_name):
  return tempfile.gettempdir() + '/' + file_name

def _removeFile(name):
  os.remove(name)

class TestAddSuffixInFilename(unittest.TestCase):
  def test_insert_with_extension(self):
    """
    Test with file name with extension.
    """
    expected = './test/file_suffix.txt'
    result = add_suffix_in_filename('./test/file.txt', '_suffix')
    self.assertEqual(expected, result)

  def test_insert_without_extension(self):
    """
    Test with file name without extension.
    """
    expected = './test/file_suffix'
    result = add_suffix_in_filename('./test/file', '_suffix')
    self.assertEqual(expected, result)

class TestReadJsonAsDict(unittest.TestCase):
  def test_nonexistent_file(self):
    """
    Test with nonexistent file.
    """
    with self.assertRaises(IOError):
      result = read_json_as_dict("tests/fixtures/nonexistent.file")

  def test_invalid_file(self):
    """
    Test with invalid file type.
    """
    tmp_file = _saveFile(
      'test_basic.py',
      'a={"str":"test","int":123,"float":1.24,"list":[1,2,3],"dict":{"key":"value"}}')
    with self.assertRaises(ValueError):
      result = read_json_as_dict(tmp_file)
    _removeFile(tmp_file)

  def test_valid_json(self):
    """
    Test with valid json file.
    """
    expected = {"str": "test", "int": 123, "float": 1.24, "list": [1,2,3], "dict": {"key": "value"}}
    tmp_file = _saveFile('test_basic.json', json.dumps(expected))
    result = read_json_as_dict(tmp_file)
    self.assertEqual(expected, result)
    _removeFile(tmp_file)

class TestReadCsvAsDict(unittest.TestCase):
  def test_nonexistent_file(self):
    """
    Test with nonexistent file.
    """
    with self.assertRaises(IOError):
      result = read_csv_as_dict("nonexistent.file", ',')

  def test_valid_file(self):
    """
    Test with valid csv file.
    """
    expected = [
      {'col2': 'test2', 'col1': 'test1', 'label': 'str'},
      {'col2': '2.5', 'col1': '1.24', 'label': 'float'},
      {'col2': '2', 'col1': '1', 'label': 'list'} 
    ]
    tmp_file = _saveCsv('test_basic.csv', expected[0].keys(), expected)
    result = read_csv_as_dict(tmp_file, ',')
    self.assertEqual(expected, result)
    _removeFile(tmp_file)

class TestSaveDictAsCsv(unittest.TestCase):
  def test_invalid_path(self):
    """
    Test with invalid csv path.
    """
    with self.assertRaises(IOError):
      save_dict_as_csv("invalid_path/test.csv",
                             ';', [{'t1': 'test1'}], ['t1'])
      
  def test_unmatched_fields(self):
    """
    Test unmatched header fields. 
    """
    temp_path = _getTempPath('test.csv')
    fieldnames = ['t1', 't2']
    rows = [{'t1': 1, 't2': 'hey'}, {'t1': 2, 't3': 1.0}]
    with self.assertRaises(ValueError):
      save_dict_as_csv(temp_path, ';', fieldnames, rows)

  def test_valid_fields(self):
    """
    Test valid fields. 
    """
    temp_path = _getTempPath('test.csv')
    fieldnames = ['t1', 't2']
    rows = [{'t1': 1, 't2': 'hey'},
            {'t1': 2, 't2': 2.3}, 
            {'t2': True}]
    save_dict_as_csv(temp_path, ';', fieldnames, rows)
    
    expected_rows = [{'t1': '1', 't2': 'hey'},
                     {'t1': '2', 't2': '2.3'},
                     {'t1': '', 't2': 'True'}]
    rows = _loadCsv(temp_path, ';', None)
    self.assertEquals(expected_rows, rows)

class TestFindByExtension(unittest.TestCase):
  def test_not_found_extension(self):
    """
    Test without files by extension.
    """
    result = find_by_extension("./tests", "not_found_extension")
    self.assertEqual([], result)

  def test_valid_dir(self):
    """
    Test with valid directory.
    """
    tmp_file = _saveFile('test_basic.json', "")
    tmp_dir = os.path.dirname(tmp_file)
    expected = ['test_basic.json']
    result = find_by_extension(tmp_dir, "json")
    self.assertEqual(expected, result)

if __name__ == '__main__':
  unittest.main()
