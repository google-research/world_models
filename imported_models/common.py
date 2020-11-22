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

"""Basic utils for video."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i, dim in enumerate(static):
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret


def to_int32(tensor):
  return tf.cast(tensor, tf.int32)


def to_uint8(tensor):
  return tf.cast(tensor, tf.uint8)


def to_float(tensor):
  return tf.cast(tensor, tf.float32)


def tinyify(array, tiny_mode, small_mode):
  if tiny_mode:
    return [1 for _ in array]
  if small_mode:
    return [max(x // 4, 1) for x in array]
  return array
