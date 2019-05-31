"""Utils definitions to manipulate files.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

def find_by_extension(data_dir, ext):
    """Find all files in directory with a defined extension.

    Args:
      data_dir: Data directory.
      ext: Extension of files to looking for.

    Returns:
      List of string with filenames.
    """
    files = []

    for dirpath, _, filenames in os.walk(data_dir):
        for filename in [f for f in filenames if f.endswith(ext)]:
            files.append(dirpath + '/' + filename)

    return files


def read_csv_as_dict(csv_path, delimiter):
    """It reads csv as dictionary.

    Args:
      csv_path: Path to csv file.
      delimiter: Delimiter of csv.

    Returns:
      A list of samples represented as dictionaries.
    """
    data_index = []
    with open(csv_path) as csv_file:
        data_index = csv.DictReader(fh, delimiter=delimiter)
    return data_index