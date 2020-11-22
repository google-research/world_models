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
"""The wrapper for SV2P model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Callable, Text, Tuple

import gin
import gym
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow
from world_models.imported_models import sv2p
from world_models.imported_models import sv2p_hparams
from world_models.tasks import tasks
from world_models.utils import npz

from tensorflow.python.distribute import values

tfs = tensorflow.compat.v2.summary

gin.external_configurable(tf.distribute.MirroredStrategy,
                          "tf.distribute.MirroredStrategy")


@gin.configurable
class SV2P(object):
  """Wrapper for SV2P."""

  def __init__(self,
               task: tasks.Task = gin.REQUIRED,
               input_length: int = gin.REQUIRED,
               output_length: int = gin.REQUIRED,
               frame_size: Tuple[int] = (64, 64, 3),
               include_frames_in_prediction=False,
               model_dir: Text = gin.REQUIRED):
    self.action_space = task.create_env().action_space
    self.frame_size = frame_size
    self.input_length = input_length
    self.output_length = output_length
    self.model_dir = model_dir
    self._include_frames_in_prediction = include_frames_in_prediction
    self._hparams = sv2p_hparams.sv2p_hparams()
    self._hparams.video_num_input_frames = input_length
    self._hparams.video_num_target_frames = output_length
    self._model = sv2p.SV2P(self._hparams)
    self._train = self._model.train
    self._infer = self._model.infer
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-3)
    self.summary_writer = tfs.create_file_writer(self.model_dir)

  def train(self, features):
    return self._train(features)

  def infer(self, features, prediction_len):
    """Enable predicting more frames than training width."""
    # Adjust hparams.
    hp = self._hparams
    hp.latent_num_frames = self.input_length + self.output_length
    hp.video_num_target_frames = prediction_len
    # Call model.
    output = self._infer(features)
    # Roll-back hparams changes after infer
    hp.latent_num_frames = 0
    hp.video_num_target_frames = self.output_length
    result = {"reward": output["rewards"]}
    if self._include_frames_in_prediction:
      result["image"] = output["frames"]
    return result

  @property
  def is_discrete_action(self):
    return isinstance(self.action_space, gym.spaces.Discrete)

  def format_actions(self, actions):
    if self.is_discrete_action:
      return tf.one_hot(actions[:, :, 0], self.action_space.n, dtype=tf.float32)
    else:
      return tf.to_float(actions)

  def create_features(self, images, rewards, actions):
    return {
        "frames": images,
        "rewards": rewards,
        "actions": actions,
    }

  def _get_trackables(self, global_step, optimizer):
    trackables = self._model.trackables
    if global_step is not None:
      trackables["global_step"] = global_step
    if optimizer is not None:
      trackables["optimizer"] = optimizer
    return trackables

  def _get_checkpoint_manager(self, global_step, optimizer):
    trackables = self._get_trackables(global_step, optimizer)
    checkpoint = tf.train.Checkpoint(**trackables)
    manager = tf.train.CheckpointManager(
        checkpoint, self.model_dir, max_to_keep=1)
    return checkpoint, manager

  def restore_latest_checkpoint(self, global_step=None, optimizer=None):
    checkpoint, manager = self._get_checkpoint_manager(global_step, optimizer)
    checkpoint.restore(manager.latest_checkpoint)
    return global_step, optimizer

  def save_checkpoint(self, global_step, optimizer=None):
    _, manager = self._get_checkpoint_manager(global_step, optimizer)
    manager.save(global_step)


@gin.configurable
def create_observe_fn(model=gin.REQUIRED):
  """Creates an observe function for SV2P."""

  @tf.function
  def observe_fn(last_image, last_action, last_reward, state):
    """the observe_fn for sv2p."""
    last_action = model.format_actions(last_action)
    last_reward = tf.to_float(last_reward)
    new_state = {
        "images": tf.concat([state["images"], last_image], axis=1)[:, 1:],
        "actions": tf.concat([state["actions"], last_action], axis=1)[:, 1:],
        "rewards": tf.concat([state["rewards"], last_reward], axis=1)[:, 1:]
    }
    return new_state

  return observe_fn


@gin.configurable
def create_predict_fn(model=gin.REQUIRED,
                      prediction_horizon=gin.REQUIRED,
                      strategy=gin.REQUIRED):
  """Creates a predict function for SV2P."""
  model.restore_latest_checkpoint(global_step=None, optimizer=None)

  @tf.function
  def predict_fn(future_action, state):
    """A predict_fn for SV2P model referenced in state.

    Args:
      future_action: a [batch, time, action_dims] tensor.
      state: a dictionary generated by `observe_fn`.

    Returns:
      predictions: a dictionary with possibly the following entries:
        * "reward": [batch, time, 1] tensor.
    """
    future_action = model.format_actions(future_action)
    actions = tf.concat((state["actions"], future_action), axis=1)
    infer_data = model.create_features(state["images"], state["rewards"],
                                       actions)

    # break down the inputs along the batch dimension to form equal sized
    # tensors in each replica.
    num_replicas = strategy.num_replicas_in_sync
    inputs = {
        key: tf.split(value, num_replicas) for key, value in infer_data.items()
    }
    dist_inputs = []
    for i in range(num_replicas):
      dist_inputs.append({key: value[i] for key, value in inputs.items()})
    devices = values.ReplicaDeviceMap(strategy.extended.worker_devices)
    dist_inputs = values.PerReplica(devices, tuple(dist_inputs))
    dist_predictions = strategy.experimental_run_v2(
        model.infer, args=(dist_inputs, prediction_horizon))
    dist_predictions = {
        key: strategy.experimental_local_results(value)
        for key, value in dist_predictions.items()
    }
    predictions = {
        key: tf.concat(value, axis=0)
        for key, value in dist_predictions.items()
    }
    return predictions

  return predict_fn


@gin.configurable
def create_train_fn(train_steps: int = gin.REQUIRED,
                    batch: int = gin.REQUIRED,
                    model: SV2P = gin.REQUIRED,
                    strategy: tf.distribute.Strategy = gin.REQUIRED,
                    save_rewards: bool = True) -> Callable[[Text], None]:
  """creates a train_fn to train SV2P model referenced in state.

  Args:
    train_steps: number of training steps.
    batch: the batch size.
    model: an SV2P model reference.
    strategy: a tf.distribute.Strategy object.
    save_rewards: whether or not to save the predicted rewards.

  Returns:
    A train_fn with the following positional arguments:
        * data_path: the path to the directory containing all episodes.
      This function returns nothing.
  """
  iterator = None

  @tf.function
  def train_step(obs):
    """Single training step."""

    def train_iter(obs):
      with tf.GradientTape() as tape:
        actions = model.format_actions(obs["action"])
        features = model.create_features(obs["image"], obs["reward"], actions)
        loss, pred_rewards = model.train(features)
        loss = tf.reduce_mean(loss)
      variables = tape.watched_variables()
      grads = tape.gradient(loss, variables)
      grads, _ = tf.clip_by_global_norm(grads, 1000)
      model.optimizer.apply_gradients(zip(grads, variables))
      return loss, pred_rewards, obs["reward"]

    return strategy.experimental_run_v2(train_iter, args=(obs,))

  def train_fn(data_path: Text):
    """Training function for SV2P."""
    nonlocal iterator
    if iterator is None:
      duration = model.input_length + model.output_length
      dataset = npz.load_dataset_from_directory(data_path, duration, batch)
      dataset = strategy.experimental_distribute_dataset(dataset)
      iterator = dataset

    with strategy.scope():
      global_step = tf.train.get_or_create_global_step()
      tfs.experimental.set_step(global_step)
      global_step, model.optimizer = model.restore_latest_checkpoint(
          global_step, model.optimizer)
      with model.summary_writer.as_default(), tfs.record_if(
          tf.math.equal(tf.math.mod(global_step, 100), 0)):
        true_rewards, pred_rewards = None, None
        for step, data in enumerate(iterator):
          if step > train_steps:
            if save_rewards:
              # We are only saving the last training batch.
              reward_dir = os.path.join(model.model_dir, "train_rewards")
              true_rewards = strategy.experimental_local_results(true_rewards)
              pred_rewards = strategy.experimental_local_results(pred_rewards)
              true_rewards = np.concatenate([x.numpy() for x in true_rewards])
              pred_rewards = np.concatenate([x.numpy() for x in pred_rewards])
              rewards_to_save = {"true": true_rewards, "pred": pred_rewards}
              npz.save_dictionary(rewards_to_save, reward_dir)
            break
          loss, pred_rewards, true_rewards = train_step(data)
          if step % 100 == 0:
            loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss)
            tf.logging.info("Training loss at %d step: %f", step, loss)
          global_step.assign_add(1)

        model.save_checkpoint(global_step, model.optimizer)

  return train_fn


@gin.configurable
def create_reset_fn(model=gin.REQUIRED):
  """Creates a reset_fn function."""

  @tf.function
  def reset_fn(**kwargs):
    """A reset_fn for SV2P.

    Args:
      **kwargs: a dictionary of inputs, including previous state.

    Returns:
      a new dictionary with posteriors removed from the state.
    """
    batch_size = kwargs["proposals"]
    input_len = model.input_length
    image_shape = model.frame_size
    if model.is_discrete_action:
      action_shape = (model.action_space.n,)
    else:
      action_shape = model.action_space.shape
    action_dtype = tf.float32
    return {
        "images":
            tf.zeros((batch_size, input_len) + image_shape, dtype=tf.int32),
        "actions":
            tf.zeros(
                (batch_size, input_len) + action_shape,
                dtype=tf.as_dtype(action_dtype)),
        "rewards":
            tf.zeros((batch_size, input_len, 1)),
    }

  return reset_fn
