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

"""Implementation of objectives."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import numpy as np
import tensorflow.compat.v1 as tf
from typing import Dict, Text


class Objective(object):
  """Base class for objectives."""

  def __call__(self, predictions: Dict[Text, np.ndarray]):
    """Calculates the reward from predictions.

    Args:
      predictions: a dictionary with possibly the following entries:
        * "image": [batch, steps, height, width, channels] np array.
        * "reward": [batch, steps, 1] np array.

    Returns:
      a [batch, 1] ndarray for the rewards.
    """
    raise NotImplementedError


@gin.configurable
class RandomObjective(Objective):
  """A test objective that returns random rewards sampled from a normal dist."""

  def __call__(self, predictions):
    batch = predictions["image"].shape[0]
    return np.random.normal(size=[batch, 1])


@gin.configurable
class DiscountedReward(Objective):
  """To be used with world model already predicting rewards."""

  def __call__(self, predictions):
    return np.sum(predictions["reward"], axis=1)


@gin.configurable
class TensorFlowDiscountedReward(Objective):
  """TensorFlow version of discounted reward."""

  @tf.function
  def __call__(self, predictions):
    return tf.reduce_sum(predictions["reward"], axis=1)
