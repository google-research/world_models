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

# python3
"""A simple forward reward predictor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Callable, Text

import gin
import gym
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2.keras as tfk


from world_models.tasks import tasks
from world_models.utils import npz

tfkl = tfk.layers

_model = None


def get_model():
  global _model
  if _model is None:
    _model = PureReward()
    _model.compile(
        optimizer=tfk.optimizers.Adam(),
        loss=tfk.losses.MeanSquaredError(),
        metrics=[tfk.metrics.MeanSquaredError()])
  return _model


@gin.configurable
class PureReward(tfk.Model):
  """Pure reward model."""

  def __init__(self,
               recurrent: bool = gin.REQUIRED,
               task: tasks.Task = gin.REQUIRED,
               output_length: int = gin.REQUIRED,
               model_dir: Text = gin.REQUIRED):
    super(PureReward, self).__init__()
    self._action_space = task.create_env().action_space
    self.output_length = output_length
    self.model_dir = model_dir
    self.ckpt_file = os.path.join(self.model_dir, "ckpt.hd5")
    self.epoch = 0
    self.callbacks = [
        tfk.callbacks.TensorBoard(
            log_dir=model_dir, write_graph=False, profile_batch=0),
    ]
    self.recurrent = recurrent
    if self.recurrent:
      self._init_recurrent_model()
    else:
      self._init_model()

  def _init_model(self):
    x = 16
    self.frame_enc = tfk.Sequential([
        tfkl.Conv2D(2 * x, 3, 2, input_shape=(64, 64, 3)),
        tfkl.LeakyReLU(),
        tfkl.Conv2D(4 * x, 3, 2),
        tfkl.LeakyReLU(),
        tfkl.Conv2D(1 * x, 3, 2),
        tfkl.LeakyReLU(),
        tfkl.Flatten(),
        tfkl.Dense(1 * x),
    ])

    if self.is_discrete_action:
      action_space_size = self._action_space.n
    else:
      action_space_size = self._action_space.shape[0]
    self.action_enc = tfk.Sequential([
        tfkl.Flatten(
            input_shape=(self.output_length, action_space_size)),
        tfkl.Dense(4 * x),
        tfkl.LeakyReLU(),
        tfkl.Dense(2 * x),
        tfkl.LeakyReLU(),
        tfkl.Dense(1 * x),
        tfkl.LeakyReLU(),
    ])

    self.reward_pred = tfk.Sequential([
        tfkl.Flatten(input_shape=(2 * x,)),
        tfkl.Dense(8 * x),
        tfkl.LeakyReLU(),
        tfkl.Dropout(0.2),
        tfkl.Dense(2 * x),
        tfkl.LeakyReLU(),
        tfkl.Dropout(0.2),
        tfkl.Dense(self.output_length)
    ])

  def _init_recurrent_model(self):
    x = 16
    self.frame_enc = tfk.Sequential([
        tfkl.Conv2D(2 * x, 3, 2, input_shape=(64, 64, 3)),
        tfkl.LeakyReLU(),
        tfkl.Conv2D(4 * x, 3, 2),
        tfkl.LeakyReLU(),
        tfkl.Conv2D(1 * x, 3, 2),
        tfkl.LeakyReLU(),
        tfkl.Flatten(),
        tfkl.Dense(1 * x),
    ])

    if self.is_discrete_action:
      action_space_size = self._action_space.n
    else:
      action_space_size = self._action_space.shape[0]
    self.action_enc = tfk.Sequential([
        tfkl.Dense(4 * x, input_shape=(self.output_length, action_space_size)),
        tfkl.LeakyReLU(),
        tfkl.Dense(2 * x),
        tfkl.LeakyReLU(),
        tfkl.Dense(1 * x),
        tfkl.LeakyReLU(),
    ])

    self.reward_pred = tfk.Sequential([
        tfkl.LSTM(256, return_sequences=True,
                  input_shape=(self.output_length, 2 * x)),
        tfkl.LayerNormalization(),
        tfkl.LeakyReLU(),
        tfkl.LSTM(128, return_sequences=True),
        tfkl.LayerNormalization(),
        tfkl.LeakyReLU(),
        tfkl.LSTM(64, return_sequences=True),
        tfkl.LayerNormalization(),
        tfkl.LeakyReLU(),
        tfkl.Dense(8 * x),
        tfkl.LeakyReLU(),
        tfkl.Dropout(0.2),
        tfkl.Dense(2 * x),
        tfkl.LeakyReLU(),
        tfkl.Dropout(0.2),
        tfkl.Dense(1)
    ])

  @property
  def is_discrete_action(self):
    return isinstance(self._action_space, gym.spaces.Discrete)

  @tf.function
  def preprocess(self, inputs):
    frames, actions = inputs
    frames = tf.image.convert_image_dtype(frames, tf.float32)
    if self.is_discrete_action:
      actions = tf.one_hot(
          actions[:, :, 0], self._action_space.n, dtype=tf.float32)
    else:
      actions = tf.to_float(actions)
    return frames, actions

  @tf.function
  def call(self, inputs):
    frames, actions = self.preprocess(inputs)
    enc_frame = self.frame_enc(frames)
    enc_actions = self.action_enc(actions)
    if self.recurrent:
      # Add fake time dimension
      enc_frame = tf.expand_dims(enc_frame, axis=1)
      enc_frame = tf.tile(enc_frame, [1, self.output_length, 1])
    stacked = tf.concat([enc_frame, enc_actions], axis=-1)
    output = self.reward_pred(stacked)
    if not self.recurrent:
      output = tf.expand_dims(output, axis=-1)
    return output


@gin.configurable
def observe_fn(last_image, last_action, last_reward, state):
  """the observe_fn for the model."""
  del last_action, last_reward, state
  state = last_image[:, 0]
  return state


@gin.configurable
def create_predict_fn(batch: int = gin.REQUIRED, proposals: int = gin.REQUIRED):
  """Create predict fn."""

  del batch, proposals
  model = get_model()
  if tf.io.gfile.exists(model.ckpt_file + ".index"):
    model.load_weights(model.ckpt_file)

  @tf.function
  def predict_fn(future_action, state):
    """A predict_fn for the model.

    Args:
      future_action: a [batch, time, action_dims] np array.
      state: a dictionary generated by `observe_fn`.

    Returns:
      predictions: a dictionary with possibly the following entries:
        * "image": [batch, time, height, width, channels] np array.
        * "reward": [batch, time] np array.
    """
    model = get_model()
    rewards = model((state, future_action))
    return {"reward": rewards}

  return predict_fn


@gin.configurable
def create_train_fn(
    train_steps: int = gin.REQUIRED,
    batch: int = gin.REQUIRED,
) -> Callable[[Text], None]:
  """creates a train_fn to train SV2P model referenced in state.

  Args:
    train_steps: number of training steps.
    batch: the batch size.

  Returns:
    A train_fn with the following positional arguments:
        * data_path: the path to the directory containing all episodes.
      This function returns nothing.
  """

  iterator = None

  def generator(iterator):
    while True:
      yield next(iterator)

  def train_fn(data_path: Text, save_rewards: bool = True):
    """Training function."""
    nonlocal iterator
    model = get_model()
    if iterator is None:
      duration = 1 + model.output_length
      dataset = npz.load_dataset_from_directory(data_path, duration, batch)
      dataset = dataset.map(lambda x: (  # pylint: disable=g-long-lambda
          (x["image"][:, 0], x["action"][:, 1:]), x["reward"][:, 1:]))
      iterator = iter(dataset)

    if tf.io.gfile.exists(model.ckpt_file + ".index"):
      model.load_weights(model.ckpt_file)
    model.fit_generator(
        generator(iterator),
        callbacks=model.callbacks,
        initial_epoch=model.epoch,
        epochs=model.epoch + 1,
        steps_per_epoch=train_steps)

    if save_rewards:
      reward_dir = os.path.join(model.model_dir, "train_rewards")
      x, y = next(iterator)
      model.fit(x, y)
      p = model(x)
      rewards_to_save = {"true": y, "pred": p}
      npz.save_dictionary(rewards_to_save, reward_dir)
      model.epoch += 1
      model.save_weights(model.ckpt_file)

  return train_fn


@gin.configurable(blacklist=["state", "batch_size"])
def reset_fn(**kwargs):
  """A reset_fn for SV2P.

  Args:
    **kwargs: a dictionary of inputs, including previous state.

  Returns:
    a new dictionary with posteriors removed from the state.
  """
  return kwargs["state"]
