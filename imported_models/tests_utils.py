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

"""Utilties for testing video models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()


def fill_hparams(hparams, in_frames, out_frames):
  hparams.video_num_input_frames = in_frames
  hparams.video_num_target_frames = out_frames
  hparams.tiny_mode = True
  hparams.reward_prediction = False
  return hparams


def create_basic_features(in_frames, out_frames, is_training):
  video_len = in_frames + out_frames if is_training else in_frames
  x = np.random.randint(0, 256, size=(8, video_len, 64, 64, 3))
  features = {
      "frames": tf.constant(x, dtype=tf.int32),
  }
  return features


def create_action_features(in_frames, out_frames, is_training):
  features = create_basic_features(in_frames, out_frames, is_training)
  video_len = in_frames + out_frames  # future actions should be present
  x = np.random.randint(0, 5, size=(8, video_len, 1))
  features["actions"] = tf.constant(x, dtype=tf.float32)
  return features


def create_full_features(in_frames, out_frames, is_training):
  features = create_action_features(in_frames, out_frames, is_training)
  video_len = in_frames + out_frames if is_training else in_frames
  x = np.random.randint(0, 5, size=(8, video_len, 1))
  features["rewards"] = tf.constant(x, dtype=tf.float32)
  return features


def get_shape_list(tensor):
  return [d.value for d in tensor.shape]


def get_expected_shape(video, expected_len):
  shape = get_shape_list(video)
  shape[1] = expected_len
  return shape


class BaseModelTest(tf.test.TestCase):
  """Base helper class for next frame tests."""

  def TrainModel(self, model_cls, hparams, features):
    model = model_cls(hparams)
    tf.train.get_or_create_global_step()
    with tf.GradientTape() as tape:
      loss = model.train(features)
    variables = tape.watched_variables()
    grads = tape.gradient(loss, variables)
    # Make sure the backward pass works as well.
    optimizer = tf.train.AdamOptimizer(1e-3)
    optimizer.apply_gradients(zip(grads, variables))
    return loss

  def InferModel(self, model_cls, hparams, features):
    model = model_cls(hparams)
    tf.train.get_or_create_global_step()
    predictions = model.infer(features)
    return predictions

  def TestVideoModel(self, in_frames, out_frames, hparams, model):
    hparams = fill_hparams(hparams, in_frames, out_frames)

    features = create_basic_features(in_frames, out_frames, True)
    loss = self.TrainModel(model, hparams, features)

    self.assertEqual(get_shape_list(loss), [8])
    self.assertEqual(loss.dtype, tf.float32)

  def TestVideoModelInfer(self, in_frames, out_frames, hparams, model):
    hparams = fill_hparams(hparams, in_frames, out_frames)

    features = create_basic_features(in_frames, out_frames, False)
    output, _ = self.InferModel(model, hparams, features)

    self.assertIsInstance(output, dict)
    self.assertIn("frames", output)
    expected_shape = get_expected_shape(features["frames"], out_frames)
    output_shape = get_shape_list(output["frames"])
    self.assertEqual(output_shape, expected_shape)

  def TestVideoModelWithActions(self, in_frames, out_frames, hparams, model):
    hparams = fill_hparams(hparams, in_frames, out_frames)
    hparams.reward_prediction = False

    features = create_action_features(in_frames, out_frames, True)
    loss = self.TrainModel(model, hparams, features)

    self.assertEqual(get_shape_list(loss), [8])
    self.assertEqual(loss.dtype, tf.float32)

  def TestVideoModelWithActionsInfer(self, in_frames, out_frames, hparams,
                                     model):
    hparams = fill_hparams(hparams, in_frames, out_frames)
    hparams.reward_prediction = False

    features = create_action_features(in_frames, out_frames, False)
    output = self.InferModel(model, hparams, features)

    self.assertIsInstance(output, dict)
    self.assertIn("frames", output)
    expected_shape = get_expected_shape(features["frames"], out_frames)
    output_shape = get_shape_list(output["frames"])
    self.assertEqual(output_shape, expected_shape)

  def TestVideoModelWithActionAndRewards(self, in_frames, out_frames, hparams,
                                         model):
    hparams = fill_hparams(hparams, in_frames, out_frames)
    hparams.reward_prediction = True

    features = create_full_features(in_frames, out_frames, True)
    loss, _ = self.TrainModel(model, hparams, features)

    self.assertEqual(get_shape_list(loss), [8])
    self.assertEqual(loss.dtype, tf.float32)

  def TestVideoModelWithActionAndRewardsInfer(self, in_frames, out_frames,
                                              hparams, model):
    hparams = fill_hparams(hparams, in_frames, out_frames)
    hparams.reward_prediction = True

    features = create_full_features(in_frames, out_frames, False)

    output = self.InferModel(model, hparams, features)

    self.assertIsInstance(output, dict)
    self.assertIn("frames", output)
    self.assertIn("rewards", output)
    expected_shape = get_expected_shape(features["frames"], out_frames)
    output_shape = get_shape_list(output["frames"])
    self.assertEqual(output_shape, expected_shape)
    expected_shape = get_expected_shape(features["rewards"], out_frames)
    output_shape = get_shape_list(output["rewards"])
    self.assertEqual(output_shape, expected_shape)

  def TestOnVariousInputOutputSizes(self, hparams, model):
    test_funcs = [self.TestVideoModel]
    test_funcs += [self.TestVideoModelInfer]
    for test_func in test_funcs:
      test_func(1, 1, hparams, model)
      test_func(1, 6, hparams, model)
      test_func(4, 1, hparams, model)
      test_func(7, 5, hparams, model)

  def TestWithActions(self, hparams, model):
    test_funcs = [self.TestVideoModelWithActions]
    test_funcs += [self.TestVideoModelWithActionsInfer]
    for test_func in test_funcs:
      test_func(1, 1, hparams, model)
      test_func(1, 6, hparams, model)
      test_func(4, 1, hparams, model)
      test_func(7, 5, hparams, model)

  def TestWithActionAndRewards(self, hparams, model):
    test_funcs = [self.TestVideoModelWithActionAndRewards]
    test_funcs += [self.TestVideoModelWithActionAndRewardsInfer]
    for test_func in test_funcs:
      test_func(1, 1, hparams, model)
      test_func(1, 6, hparams, model)
      test_func(4, 1, hparams, model)
      test_func(7, 5, hparams, model)
