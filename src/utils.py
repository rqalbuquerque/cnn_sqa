"""Util definitions to manipulate directories and files.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import csv


def create_dir(name):
    """Create a new directory if not exist yet.
      
      Args: 
        name: String path with directory name.
    """
    if not os.path.exists(name):
        os.makedirs(name)


def add_suffix_in_filename(path_file, suffix):
    """Add a suffix into file name before extension.
      
      Args: 
        path_file: Path file name.
        suffix: String suffix to insert.
    """
    name, extension = os.path.splitext(path_file)
    return name + suffix + extension


def find_by_extension(data_dir, ext):
    """Find all files in directory by extension.

    Args:
      data_dir: Data directory.
      ext: Extension of files to looking for.

    Returns:
      List of string with filenames.
    """
    files = []
    for root, _, filenames in os.walk(data_dir):
        for filename in [f for f in filenames if f.endswith(ext)]:
            rel_dir = os.path.relpath(root, data_dir)
            norm_path = os.path.normpath(os.path.join(rel_dir, filename))
            files.append(norm_path)
    return files


def read_csv_as_dict(csv_path, delimiter, fieldnames=None):
    """It reads csv as dictionary.

    Args:
      csv_path: Path to csv file.
      delimiter: Delimiter of csv.

    Returns:
      A list of samples represented as dictionaries.
    """
    data = []
    with open(csv_path) as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimiter)
        data = [row for row in reader]
    return data


def read_json_as_dict(path):
    """It reads json as dictionary.

    Args:
      path: Path to csv file.

    Returns:
      A dictionary with data.
    """
    with open(path) as f:
        return json.load(f)
