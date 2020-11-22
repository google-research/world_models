# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import functools
import io
import os
import random
import uuid

import numpy as np
import tensorflow.compat.v1 as tf

from world_models.utils import nested

tfdd = tf.data.Dataset

# pylint:disable=missing-docstring


def save_dictionaries(dictionaries, directory):
  for dictionary in dictionaries:
    save_dictionary(dictionary, directory)


def save_dictionary(dictionary, directory):
  """Save a dictionary as npz."""
  timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
  identifier = str(uuid.uuid4()).replace('-', '')
  filename = '{}-{}.npz'.format(timestamp, identifier)
  filename = os.path.join(directory, filename)
  if not tf.gfile.Exists(directory):
    tf.gfile.MakeDirs(directory)
  with io.BytesIO() as file_:
    np.savez_compressed(file_, **dictionary)
    file_.seek(0)
    with tf.gfile.Open(filename, 'w') as ff:
      ff.write(file_.read())
  return filename


def load_dataset_from_directory(directory,
                                length,
                                batch,
                                cache_update_every=1000,
                                buffer_size=10):
  loader = functools.partial(numpy_loader, directory, cache_update_every)
  dtypes, shapes = read_spec(loader)
  dtypes = {key: tf.as_dtype(value) for key, value in dtypes.items()}
  shapes = {key: (None,) + shape[1:] for key, shape in shapes.items()}
  chunking = functools.partial(chunk_sequence, length=length)
  dataset = tfdd.from_generator(loader, dtypes, shapes)
  dataset = dataset.flat_map(chunking)
  dataset = dataset.batch(batch, drop_remainder=True)
  dataset = dataset.prefetch(buffer_size)
  return dataset


def numpy_loader(directory, cache_update_every=1000):
  """A generator for loading npzs from a directory."""
  cache = {}
  while True:
    data = _sample(list(cache.values()), cache_update_every)
    for dictionary in _permuted(data, cache_update_every):
      yield dictionary
    directory = os.path.expanduser(directory)
    filenames = tf.gfile.Glob(os.path.join(directory, '*.npz'))
    filenames = [filename for filename in filenames if filename not in cache]
    for filename in filenames:
      with tf.gfile.Open(filename, 'rb') as file_:
        cache[filename] = dict(np.load(file_))


def _sample(sequence, amount):
  amount = min(amount, len(sequence))
  return random.sample(sequence, amount)


def _permuted(sequence, amount):
  """a generator for `amount` elements from permuted elements in `sequence`."""
  if not sequence:
    return
  index = 0
  while True:
    for element in np.random.permutation(sequence):
      if index >= amount:
        return
      yield element
      index += 1


def read_spec(loader):
  dictionaries = loader()
  dictionary = next(dictionaries)
  dictionaries.close()
  dtypes = {key: value.dtype for key, value in dictionary.items()}
  shapes = {key: value.shape for key, value in dictionary.items()}
  return dtypes, shapes


def chunk_sequence(sequence, length):
  """Randomly chunks a sequence into smaller ones.

     This is useful for sampling short videos from long ones.

  Args:
    sequence: the original dataset with long sequences.
    length: length of the desired chunks.

  Returns:
    chuncked dataset.
  """
  with tf.device('/cpu:0'):
    seq_length = tf.shape(nested.flatten(sequence)[0])[0]
    max_offset = seq_length - length
    op = tf.Assert(tf.greater_equal(max_offset, 0), data=[length, seq_length])
    with tf.control_dependencies([op]):
      offset = tf.random_uniform((), 0, max_offset + 1, dtype=tf.int32)
      clipped = nested.map(lambda x: x[offset:offset + length], sequence)
      chunks = tfdd.from_tensor_slices(
          nested.map(
              lambda x: tf.reshape(x, [-1, length] + x.shape[1:].as_list()),
              clipped))
    return chunks
