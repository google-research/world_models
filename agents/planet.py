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
"""A fork of PlaNet model.

Archive paper: https://arxiv.org/abs/1811.04551
OSS code repo: https://github.com/google-research/planet
"""
# pylint:disable=missing-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import gin
import gym
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2.summary as tfs
from tensorflow_probability import distributions as tfd
from world_models.imported_models import reward_models
from world_models.utils import npz
from world_models.utils import visualization

from tensorflow.python.distribute import values

gin.external_configurable(tf.distribute.MirroredStrategy,
                          'tf.distribute.MirroredStrategy')


def static_scan(fn, inputs, start, reverse=False):
  # pylint: disable=expression-not-assigned
  # pylint: disable=cell-var-from-loop
  """drop-in replacement for tf.scan.

  tf.scan has some issues with multiple devices.
  """
  last = start
  outputs = [[] for _ in tf.nest.flatten(start)]
  indices = range(tf.nest.flatten(inputs)[0].shape[0])
  if reverse:
    indices = reversed(indices)
  for index in indices:
    inp = tf.nest.map_structure(lambda x: x[index], inputs)
    last = fn(last, inp)
    [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
  if reverse:
    outputs = [list(reversed(x)) for x in outputs]
  outputs = [tf.stack(x, 0) for x in outputs]
  return tf.nest.pack_sequence_as(start, outputs)


@gin.configurable
class RecurrentStateSpaceModel(object):

  def __init__(self,
      stoch_size=30,
      deter_size=200,
      min_stddev=0.1,
      layers=1,
      reward_layers=3,
      units=300,
      free_nats=3.0,
      reward_loss_multiplier=10,
      frame_size=(64, 64, 3),
      task=gin.REQUIRED,
      reward_from_frames=False,
      reward_stop_gradient=False,
      include_frames_in_prediction=False,
      activation=tf.nn.relu):
    self._action_space = task.create_env().action_space
    self._stoch_size = stoch_size
    self._deter_size = deter_size
    self._min_stddev = min_stddev
    self._num_layers = layers
    self._num_reward_layers = reward_layers
    self._num_units = units
    self._free_nats = free_nats
    self._include_frames_in_prediction = include_frames_in_prediction
    self._activation = activation
    self._cell = tf.keras.layers.GRUCell(self._deter_size)
    self._prior_tpl = tf.make_template('prior', self._prior)
    self._posterior_tpl = tf.make_template('posterior', self._posterior)
    self._encoder_tpl = tf.make_template('encoder', self._encoder)
    self._reward_loss_mul = reward_loss_multiplier
    self._frame_size = list(frame_size)
    self._reward_from_frames = reward_from_frames
    self._reward_stop_gradient = reward_stop_gradient
    self._predict_frames_tpl = tf.make_template(
        'predict_frames', self._predict_frames, out_shape=self._frame_size)
    self._predict_reward_tpl = tf.make_template(
        'predict_reward', self._predict_reward, out_shape=[1])

  @property
  def is_discrete_action(self):
    return isinstance(self._action_space, gym.spaces.Discrete)

  def get_trackables(self):
    return {
        'prior_tpl': self._prior_tpl,
        'posterior_tpl': self._posterior_tpl,
        'encoder_tpl': self._encoder_tpl,
        'predict_frames_tpl': self._predict_frames_tpl,
        'predict_reward_tpl': self._predict_reward_tpl,
        'cell': self._cell,
    }

  def initialize(self, batch_size):
    return {
        'mean':
          tf.zeros([batch_size, self._stoch_size]),
        'std':
          tf.zeros([batch_size, self._stoch_size]),
        'stoch':
          tf.zeros([batch_size, self._stoch_size]),
        'deter':
          self._cell.get_initial_state(
              batch_size=batch_size, dtype=tf.float32)
    }

  def compute_losses(self, obs):
    image = obs['image']
    action = obs['action']
    reward = obs['reward']
    state = self.initialize(tf.shape(image)[0])
    state['rewards'] = reward
    priors, posteriors = self.observe(action, image, state)
    features = self._get_features(posteriors)
    frames = self._predict_frames_tpl(features)
    if self._reward_from_frames:
      rewards = self._predict_reward_tpl(frames.mode(), reward[:, -1])
    else:
      rewards = self._predict_reward_tpl(features, reward[:, -1])
    obs_likelihood = frames.log_prob(image)
    reward_likelihood = rewards.log_prob(tf.to_float(reward))

    divergence = tfd.kl_divergence(
        self._get_distribution(posteriors), self._get_distribution(priors))
    divergence = tf.maximum(self._free_nats, divergence)
    loss = tf.reduce_mean(divergence - obs_likelihood -
                          reward_likelihood * self._reward_loss_mul)
    frames_mode = tf.clip_by_value((frames.mode() + 0.5) * 255, 0, 255)
    return (loss, tf.reduce_mean(reward_likelihood), tf.reduce_mean(divergence),
            tf.cast(frames_mode, dtype=tf.uint8), rewards.mode(), obs['reward'],
            tf.reduce_mean(tf.math.squared_difference(frames_mode, image)))

  def observe(self, actions, images, state):
    embedded_obs = self._encoder_tpl(images)
    if self.is_discrete_action:
      actions = tf.one_hot(
          actions[:, :, 0], self._action_space.n, dtype=tf.float32)
    else:
      actions = tf.to_float(actions)
    actions = tf.transpose(actions, [1, 0, 2])
    embedded_obs = tf.transpose(embedded_obs, [1, 0, 2])
    state.pop('rewards')
    priors, posteriors = static_scan(
        lambda prev, inp: self._posterior_tpl(prev[1], *inp),
        (actions, embedded_obs), (state, state))
    priors = {
        key: tf.transpose(value, [1, 0, 2]) for key, value in priors.items()
    }
    posteriors = {
        key: tf.transpose(value, [1, 0, 2])
        for key, value in posteriors.items()
    }
    return priors, posteriors

  def predict(self, actions, state):
    if isinstance(self._action_space, gym.spaces.Discrete):
      actions = tf.one_hot(
          actions[:, :, 0], self._action_space.n, dtype=tf.float32)
    else:
      actions = tf.to_float(actions)
    actions = tf.transpose(actions, [1, 0, 2])
    rewards = tf.to_float(state.pop('rewards'))
    priors = static_scan(self._prior_tpl, actions, state)
    priors = {
        key: tf.transpose(value, [1, 0, 2]) for key, value in priors.items()
    }
    features = self._get_features(priors)
    results = {}
    if self._reward_from_frames:
      frames = self._predict_frames_tpl(features).mode()
      results['reward'] = self._predict_reward_tpl(frames, rewards).mode()
    else:
      results['reward'] = self._predict_reward_tpl(features, rewards).mode()
    if self._include_frames_in_prediction:
      results['image'] = tf.cast(
          tf.clip_by_value(
              (self._predict_frames_tpl(features).mode() + 0.5) * 255, 0, 255),
          dtype=tf.uint8)
    return results

  def _get_distribution(self, states):
    return tfd.MultivariateNormalDiag(states['mean'], states['std'])

  def _get_features(self, states):
    return tf.concat([states['stoch'], states['deter']], -1)

  def _prior(self, prev_state, prev_action):
    hidden = tf.concat([prev_state['stoch'], prev_action], -1)
    for _ in range(self._num_layers):
      hidden = tf.layers.dense(hidden, self._num_units,
                               self._activation)
    hidden, deter = self._cell(hidden, [prev_state['deter']])
    deter = deter[0]
    for _ in range(self._num_layers):
      hidden = tf.layers.dense(hidden, self._num_units,
                               self._activation)
    mean, std = tf.split(
        tf.layers.dense(hidden, 2 * self._stoch_size), 2, -1)
    std = tf.nn.softplus(std) + self._min_stddev
    stoch = tfd.MultivariateNormalDiag(mean, std).sample()
    return {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}

  def _posterior(self, prev_state, prev_action, embedded_obs):
    prior = self._prior_tpl(prev_state, prev_action)
    hidden = tf.concat([prior['deter'], embedded_obs], -1)
    for _ in range(self._num_layers):
      hidden = tf.layers.dense(hidden, self._num_units,
                               self._activation)
    mean, std = tf.split(
        tf.layers.dense(hidden, 2 * self._stoch_size), 2, -1)
    std = tf.nn.softplus(std) + self._min_stddev
    stoch = tfd.MultivariateNormalDiag(mean, std).sample()
    post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
    return prior, post

  def _encoder(self, images):
    kwargs = dict(strides=2, activation=tf.nn.relu)
    images = tf.to_float(images)
    hidden = tf.reshape(images, [-1] + images.shape[2:].as_list())
    hidden = tf.layers.conv2d(hidden, 32, 4, **kwargs)
    hidden = tf.layers.conv2d(hidden, 64, 4, **kwargs)
    hidden = tf.layers.conv2d(hidden, 128, 4, **kwargs)
    hidden = tf.layers.conv2d(hidden, 256, 4, **kwargs)
    hidden = tf.layers.flatten(hidden)
    assert hidden.shape[1:].as_list() == [1024], hidden.shape.as_list()
    embedded_obs = tf.reshape(hidden, [
        tf.shape(images)[0],
        tf.shape(images)[1],
        np.prod(hidden.shape[1:].as_list())
    ])
    return embedded_obs

  def _predict_frames(self, features, out_shape):
    kwargs = dict(strides=2, activation=tf.nn.relu)
    hidden = tf.layers.dense(features, 1024, None)
    hidden = tf.reshape(hidden, [-1, 1, 1, hidden.shape[-1]])
    hidden = tf.layers.conv2d_transpose(hidden, 128, 5, **kwargs)
    hidden = tf.layers.conv2d_transpose(hidden, 64, 5, **kwargs)
    hidden = tf.layers.conv2d_transpose(hidden, 32, 6, **kwargs)
    mean = tf.layers.conv2d_transpose(hidden, 3, 6, strides=2)
    assert mean.shape[1:].as_list() == [64, 64, 3], mean.shape
    mean = tf.reshape(mean, tf.concat([tf.shape(features)[:-1], out_shape], 0))
    mean = tf.cast(mean, tf.float32)
    return tfd.Independent(tfd.Normal(mean, 1), len(out_shape))

  def _predict_reward(self, features, rewards, out_shape):
    if self._reward_stop_gradient:
      hidden = tf.stop_gradient(features)
    else:
      hidden = features
    if self._reward_from_frames:
      split_frames = [
          tf.squeeze(f, axis=1)
          for f in tf.split(features, features.shape[1], axis=1)
      ]
      mean = reward_models.reward_prediction_video_conv(split_frames, rewards,
                                                        len(split_frames))
      return tfd.Independent(tfd.Normal(mean, 1), len(out_shape))
    else:
      for _ in range(self._num_reward_layers):
        hidden = tf.layers.dense(hidden, self._num_units,
                                 self._activation)
      mean = tf.layers.dense(hidden, int(np.prod(out_shape)))
      mean = tf.reshape(mean, tf.concat([tf.shape(features)[:-1], out_shape],
                                        0))
      return tfd.Independent(tfd.Normal(mean, 1), len(out_shape))


@gin.configurable
def create_planet_reset_fn(model):
  @tf.function
  def reset(**kwargs):
    batch_size = kwargs['proposals']
    state = model.initialize(batch_size)
    state['rewards'] = tf.zeros([batch_size, 1])
    return state

  return reset


@gin.configurable
def create_planet_observe_fn(model, model_dir, strategy):
  with strategy.scope():
    checkpoint = tf.train.Checkpoint(**model.get_trackables())
    manager = tf.train.CheckpointManager(checkpoint, model_dir, max_to_keep=1)
    checkpoint.restore(manager.latest_checkpoint).expect_partial()

  @tf.function
  def observe(images, actions, rewards, state):
    images = tf.to_float(images) / 255.0 - 0.5
    # break down the inputs along the batch dimension to form equal sized
    # tensors in each replica.
    num_replicas = strategy.num_replicas_in_sync
    images = tf.split(images, num_replicas)
    actions = tf.split(actions, num_replicas)
    state = {key: tf.split(value, num_replicas) for key, value in state.items()}
    devices = values.ReplicaDeviceMap(strategy.extended.worker_devices)
    dist_images = values.PerReplica(devices, tuple(images))
    dist_actions = values.PerReplica(devices, tuple(actions))
    dist_state = []
    for i in range(num_replicas):
      dist_state.append({key: value[i] for key, value in state.items()})
    dist_state = values.PerReplica(devices, tuple(dist_state))
    _, dist_posteriors = strategy.experimental_run_v2(
        model.observe, args=(dist_actions, dist_images, dist_state))
    dist_posteriors = {
        key: strategy.experimental_local_results(value)
        for key, value in dist_posteriors.items()
    }
    posteriors = {
        key: tf.concat(value, axis=0) for key, value in dist_posteriors.items()
    }
    posteriors = {key: value[:, -1] for key, value in posteriors.items()}
    posteriors['rewards'] = rewards[:, -1]
    return posteriors

  return observe


@gin.configurable
def create_planet_predict_fn(model, strategy):
  @tf.function
  def predict(actions, state):
    state = state.copy()
    # break down the inputs along the batch dimension to form equal sized
    # tensors in each replica.
    num_replicas = strategy.num_replicas_in_sync
    actions = tf.split(actions, num_replicas)
    state = {key: tf.split(value, num_replicas) for key, value in state.items()}
    devices = values.ReplicaDeviceMap(strategy.extended.worker_devices)
    dist_actions = values.PerReplica(devices, tuple(actions))
    dist_state = []
    for i in range(num_replicas):
      dist_state.append({key: value[i] for key, value in state.items()})
    dist_state = values.PerReplica(devices, tuple(dist_state))

    dist_predictions = strategy.experimental_run_v2(
        model.predict, args=(dist_actions, dist_state))
    dist_predictions = {
        key: strategy.experimental_local_results(value)
        for key, value in dist_predictions.items()
    }
    predictions = {
        key: tf.concat(value, axis=0)
        for key, value in dist_predictions.items()
    }
    return predictions

  return predict


@gin.configurable
def create_planet_train_fn(model: RecurrentStateSpaceModel = gin.REQUIRED,
    train_steps: int = gin.REQUIRED,
    batch: int = gin.REQUIRED,
    duration: int = gin.REQUIRED,
    learning_rate: float = gin.REQUIRED,
    model_dir=gin.REQUIRED,
    strategy: tf.distribute.Strategy = gin.REQUIRED,
    save_rewards: bool = True):
  """creates a train_fn to train the `tf.Estimator` referenced in state.

  Args:
    model: a reference to the model.
    train_steps: number of training steps.
    batch: the batch size.
    duration: how many timesteps to include in a single video sequence.
    learning_rate: learning rate.
    model_dir: the path to model directory.
    strategy: a tf.distribute.Strategy object.
    save_rewards: whether or not to save the predicted rewards.

  Returns:
    A train_fn with the following positional arguments:
        * data_path: the path to all episodes.
      This function returns nothing.
  """
  iterator = None
  optimizer = tf.keras.optimizers.Adam(learning_rate, epsilon=1e-3)

  @tf.function
  def train_step(obs):

    def train_iter(obs):
      obs = obs.copy()
      obs['image'] = tf.to_float(obs['image']) / 255.0 - 0.5
      with tf.GradientTape() as tape:
        output = model.compute_losses(obs)
        loss, reward_loss, divergence, frames = output[:4]
        pred_rewards, true_rewards, frame_loss = output[4:]
      variables = tape.watched_variables()
      grads = tape.gradient(loss, variables)
      grads, _ = tf.clip_by_global_norm(grads, 1000)
      optimizer.apply_gradients(zip(grads, variables))
      return loss, reward_loss, divergence, frames, pred_rewards, true_rewards, frame_loss

    return strategy.experimental_run_v2(train_iter, args=(obs,))

  def train_fn(data_path):
    """A train_fn to train the planet model."""
    nonlocal iterator
    nonlocal optimizer

    with strategy.scope():
      global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
      checkpoint = tf.train.Checkpoint(
          global_step=global_step,
          optimizer=optimizer,
          **model.get_trackables())
      manager = tf.train.CheckpointManager(checkpoint, model_dir, max_to_keep=1)
      checkpoint.restore(manager.latest_checkpoint)
    if iterator is None:
      dataset = npz.load_dataset_from_directory(data_path, duration, batch)
      dataset = strategy.experimental_distribute_dataset(dataset)
      iterator = dataset

    writer = tfs.create_file_writer(model_dir)
    tfs.experimental.set_step(global_step)
    true_rewards, pred_rewards = None, None
    with writer.as_default():
      for step, obs in enumerate(iterator):
        if step > train_steps:
          if save_rewards:
            # We are only saving the last training batch.
            reward_dir = os.path.join(model_dir, 'train_rewards')
            true_rewards = strategy.experimental_local_results(true_rewards)
            pred_reward = strategy.experimental_local_results(pred_rewards)
            true_rewards = np.concatenate([x.numpy() for x in true_rewards])
            pred_reward = np.concatenate([x.numpy() for x in pred_reward])
            rewards_to_save = {'true': true_rewards, 'pred': pred_reward}
            npz.save_dictionary(rewards_to_save, reward_dir)
          break
        (loss, reward_loss, divergence, frames, pred_rewards, true_rewards,
         frame_loss) = train_step(obs)
        if step % 100 == 0:
          loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss)
          reward_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                        reward_loss)
          divergence = strategy.reduce(tf.distribute.ReduceOp.MEAN, divergence)
          frame_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, frame_loss)
          frames = strategy.experimental_local_results(frames)
          frames = tf.concat(frames, axis=0)
          pred_reward = strategy.experimental_local_results(pred_rewards)
          pred_reward = tf.concat(pred_reward, axis=0)
          tf.logging.info('loss at step %d: %f', step, loss)
          tfs.scalar('loss/total', loss)
          tfs.scalar('loss/reward', reward_loss)
          tfs.scalar('loss/divergence', divergence)
          tfs.scalar('loss/frames', frame_loss)
          tfs.experimental.write_raw_pb(
              visualization.py_gif_summary(
                  tag='predictions/frames',
                  images=frames.numpy(),
                  max_outputs=6,
                  fps=20))
          ground_truth_rewards = (
              tf.concat(
                  strategy.experimental_local_results(obs['reward']),
                  axis=0)[:, :, 0])
          rewards = pred_reward[:, :, 0]
          signals = tf.stack([ground_truth_rewards, rewards], axis=1)
          visualization.py_plot_1d_signal(
              name='predictions/reward',
              signals=signals.numpy(),
              labels=['ground_truth', 'prediction'],
              max_outputs=6)
        global_step.assign_add(1)

      manager.save(global_step)

  return train_fn
