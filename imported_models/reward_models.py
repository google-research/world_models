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

"""Reward Models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.layers as tfl

from world_models.imported_models import common
from world_models.imported_models import layers
from tensorflow.contrib import layers as tfcl



def reward_prediction_basic(prediction):
  """The most simple reward predictor.

     This works by averaging the pixels and running a dense layer on top.

  Args:
    prediction: The predicted image.

  Returns:
    the predicted reward.
  """
  x = prediction
  x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
  x = tf.squeeze(x, axis=[1, 2])
  x = tfl.dense(x, 128, activation=tf.nn.relu, name="reward_pred")
  return x


def reward_prediction_mid(input_images):
  """A reward predictor network from intermediate layers.

     The inputs can be any image size (usually the intermediate conv outputs).
     The model runs 3 conv layers on top of each with a dense layer at the end.
     All of these are combined with 2 additional dense layer.

  Args:
    input_images: the input images. size is arbitrary.

  Returns:
    the predicted reward.
  """
  encoded = []
  for i, x in enumerate(input_images):
    enc = x
    enc = tfl.conv2d(enc, 16, [3, 3], strides=(1, 1), activation=tf.nn.relu)
    enc = tfl.conv2d(enc, 8, [3, 3], strides=(2, 2), activation=tf.nn.relu)
    enc = tfl.conv2d(enc, 4, [3, 3], strides=(2, 2), activation=tf.nn.relu)
    enc = tfl.flatten(enc)
    enc = tfl.dense(enc, 8, activation=tf.nn.relu, name="rew_enc_%d" % i)
    encoded.append(enc)
  x = encoded
  x = tf.stack(x, axis=1)
  x = tfl.flatten(x)
  x = tfl.dense(x, 32, activation=tf.nn.relu, name="rew_dense1")
  x = tfl.dense(x, 16, activation=tf.nn.relu, name="rew_dense2")
  return x


def reward_prediction_big(input_images, input_reward, action, latent,
                          action_injection, small_mode):
  """A big reward predictor network that incorporates lots of additional info.

  Args:
    input_images: context frames.
    input_reward: context rewards.
    action: next action.
    latent: predicted latent vector for this frame.
    action_injection: action injection method.
    small_mode: smaller convs for faster runtiume.

  Returns:
    the predicted reward.
  """
  conv_size = common.tinyify([32, 32, 16, 8], False, small_mode)

  x = tf.concat(input_images, axis=3)
  x = tfcl.layer_norm(x)

  if not small_mode:
    x = tfl.conv2d(
        x,
        conv_size[1], [3, 3],
        strides=(2, 2),
        activation=tf.nn.relu,
        name="reward_conv1")
    x = tfcl.layer_norm(x)

  # Inject additional inputs
  if action is not None:
    x = layers.inject_additional_input(x, action, "action_enc",
                                       action_injection)
  if input_reward is not None:
    x = layers.inject_additional_input(x, input_reward, "reward_enc")
  if latent is not None:
    latent = tfl.flatten(latent)
    latent = tf.expand_dims(latent, axis=1)
    latent = tf.expand_dims(latent, axis=1)
    x = layers.inject_additional_input(x, latent, "latent_enc")

  x = tfl.conv2d(
      x,
      conv_size[2], [3, 3],
      strides=(2, 2),
      activation=tf.nn.relu,
      name="reward_conv2")
  x = tfcl.layer_norm(x)
  x = tfl.conv2d(
      x,
      conv_size[3], [3, 3],
      strides=(2, 2),
      activation=tf.nn.relu,
      name="reward_conv3")
  return x


def reward_prediction_video_conv(frames, rewards, prediction_len):
  """A reward predictor network from observed/predicted images.

     The inputs is a list of frames.

  Args:
    frames: the list of input images.
    rewards: previously observed rewards.
    prediction_len: the length of the reward vector.

  Returns:
    the predicted rewards.
  """
  x = tf.concat(frames, axis=-1)
  x = tfl.conv2d(x, 32, [3, 3], strides=(2, 2), activation=tf.nn.relu)
  x = tfl.conv2d(x, 32, [3, 3], strides=(2, 2), activation=tf.nn.relu)
  x = tfl.conv2d(x, 16, [3, 3], strides=(2, 2), activation=tf.nn.relu)
  x = tfl.conv2d(x, 8, [3, 3], strides=(2, 2), activation=tf.nn.relu)
  x = tfl.flatten(x)

  y = tf.concat(rewards, axis=-1)
  y = tfl.dense(y, 32, activation=tf.nn.relu)
  y = tfl.dense(y, 16, activation=tf.nn.relu)
  y = tfl.dense(y, 8, activation=tf.nn.relu)

  z = tf.concat([x, y], axis=-1)
  z = tfl.dense(z, 32, activation=tf.nn.relu)
  z = tfl.dense(z, 16, activation=tf.nn.relu)
  z = tfl.dense(z, prediction_len, activation=None)
  z = tf.expand_dims(z, axis=-1)
  return z
