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

"""Implementation of planners."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable, Any, Dict, Text, List

import gin
from gym.spaces import discrete
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from world_models.tasks import tasks


class Planner(object):
  """Base class for planners."""

  def __call__(self, prev_image: np.ndarray, prev_action: np.ndarray,
               prev_reward: float):
    raise NotImplementedError

  def reset(self, **kwargs):
    raise NotImplementedError

  def set_episode_num(self, episode_num):
    pass


@gin.configurable
class CEM(Planner):
  """Cross entropy method."""

  def __init__(
      self,
      predict_fn: Callable[[np.ndarray, Any], Dict[Text, np.ndarray]],
      observe_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, Any], Any],
      reset_fn: Callable[[Any, int], Any],
      task: tasks.Task,
      objective_fn: Callable[[Dict[Text, np.ndarray]], np.ndarray],
      horizon: int,
      iterations: int,
      proposals: int,
      fraction: float,
      weighted: bool = False,
  ):
    """Initialize a CEM planner that queries a world model.

    Args:
      predict_fn: a callable with the following positional arguments:
          * planned_actions: a [batch, steps, action_dims] ndarray
          * state: the state object returned from observe_fn.
        This method is expected to return:
          * predictions: a dictionary with possibly the following entries:
            * "image": [batch, steps, height, width, channels] ndarray of the
              model predictions of the state of the world conditioned actions.
            * "reward": [batch, steps, 1] ndarray of the reward predictions.
      observe_fn: a callable with the following positional arguments:
          * last_images: a [batch, steps, height, width, channels] ndarray
          * last_actions: a [batch, steps, action_dims] ndarray
          * last_reward: a [batch, steps, 1] ndarray
          * state: the previously returned `state`.
        This method is expected to return:
          * state: Anything the model needs to continue predicting current
            episode.
      reset_fn: a callable with the following keyword arguments:
          * state: the state object
          * proposals: the number of proposals.
        This method is expected to return:
          * state: the new state with cleared history.
      task: The task of type `tasks.Task`.
      objective_fn: a callable with the following positional arguments:
          * predictions: a dictionary possibly containing "image" and "reward"
            that will come from the `predict_fn`
        This method is expected to return:
          * rewards: a [batch, steps, 1] ndarray of containing scalar rewards.
      horizon: the planning horizon to specify how many steps into the future to
        consider for choosing the right plan.
      iterations: How many iterations to estimate the final distribution.
      proposals: How many proposals to consider in each iteration.
      fraction: The percentage of proposals to select with the highest score to
        fit the distribution.
      weighted: fit distribution by weighting proposals with their predicted
        rewards.
    """
    super(CEM, self).__init__()
    self._action_space = task.create_env().action_space
    self._objective_fn = objective_fn
    self._predict_fn = predict_fn
    self._observe_fn = observe_fn
    self._reset_fn = reset_fn
    self._horizon = horizon
    self._iterations = iterations
    self._proposals = proposals
    self._fraction = fraction
    self._weighted = weighted
    self._state = {}

  @property
  def is_discrete(self):
    return isinstance(self._action_space, discrete.Discrete)

  def reset(self, **kwargs):
    kwargs["state"] = self._state
    kwargs["proposals"] = self._proposals
    self._state = self._reset_fn(**kwargs)

  def initialize_distribution(self):
    """Returns initial distribution for action space."""
    # start with a uniform distribution to sample from.
    if self.is_discrete:
      n = self._action_space.n
      return [[1. / n] * n] * self._horizon
    else:
      means = [(self._action_space.high + self._action_space.low) / 2.0
              ] * self._horizon
      covs = [
          np.diag((self._action_space.high - self._action_space.low) / 2.0)
      ] * self._horizon
      return means, covs

  def sample_actions(self, dist):
    if self.is_discrete:
      return self._sample_discrete_actions(dist)
    else:
      means, covs = dist
      return self._sample_continuous_actions(means, covs)

  def _sample_continuous_actions(self, means, covs):
    """Samples actions from a multivariate Gaussian."""
    all_actions = []
    for (mean, cov) in zip(means, covs):
      actions = np.random.multivariate_normal(mean, cov, (self._proposals,))
      actions = np.clip(actions, self._action_space.low,
                        self._action_space.high)
      actions = actions.astype(np.float32)
      all_actions.extend([actions])

    all_actions = np.array(all_actions)
    traj_proposals = np.transpose(all_actions, (1, 0, 2))
    return traj_proposals

  def _sample_discrete_actions(self, pvals):
    """Samples actions from multinomial."""
    all_actions = []
    for pval in pvals:
      # [pval_size, proposals]
      actions = np.random.multinomial(n=1, pvals=pval, size=(self._proposals,))
      # [proposals, 1]
      actions = np.expand_dims(np.argmax(actions, axis=1), axis=-1)
      actions = actions.astype(np.int32)
      all_actions.append(actions)

    # [horizon, proposals, 1]
    all_actions = np.array(all_actions)
    traj_proposals = np.transpose(all_actions, (1, 0, 2))
    return traj_proposals

  def generate_rewards(self, traj_proposals):
    """Given a set of actions, outputs the corresponding rewards."""
    predictions = self._predict_fn(traj_proposals, self._state)
    traj_rewards = self._objective_fn(predictions)
    return traj_rewards, predictions

  def _fit_gaussian(self, rewards, traj_proposals):
    """Re-fits a Gaussian to the best actions."""
    top_k = int(self._fraction * self._proposals)
    indices = np.squeeze(
        np.argpartition(rewards, -top_k, axis=0), axis=-1)[-top_k:]
    best_trajectories = traj_proposals[indices]
    if self._weighted:
      weights = rewards.numpy()[indices, 0]
    else:
      weights = np.ones_like(rewards[indices, 0])
    actions_to_fit = np.transpose(best_trajectories[0:top_k, 0:self._horizon],
                                  (1, 0, 2))
    means = []
    covs = []

    for i in range(self._horizon):
      means.append(np.average(actions_to_fit[i], weights=weights, axis=0))
      covs.append(np.cov(actions_to_fit[i].T, aweights=weights))
    return means, covs

  def _fit_multinomial(self, rewards, traj_proposals):
    """Re-fits multinomials to the best actions."""
    top_k = int(self._fraction * self._proposals)
    indices = np.squeeze(
        np.argpartition(rewards, -top_k, axis=0), axis=-1)[-top_k:]
    best_trajectories = traj_proposals[indices]
    if self._weighted:
      weights = rewards.numpy()[indices, 0]
    else:
      weights = np.ones_like(rewards[indices, 0])
    actions_to_fit = np.transpose(best_trajectories[0:top_k, 0:self._horizon],
                                  (1, 0, 2))
    pvals = []
    for i in range(self._horizon):
      action_onehot = np.bincount(
          actions_to_fit[i, :, 0], minlength=self._action_space.n)
      pval = action_onehot * weights / np.sum(weights)
      pvals.append(pval.tolist())
    return pvals

  def fit_dist(self, rewards, traj_proposals):
    if self.is_discrete:
      return self._fit_multinomial(rewards, traj_proposals)
    else:
      return self._fit_gaussian(rewards, traj_proposals)

  def __call__(self, prev_image, prev_action, prev_reward):
    prev_reward = np.asarray([prev_reward])
    prev_action = np.asarray(prev_action)
    if prev_action.ndim == 0:
      prev_action = np.expand_dims(prev_action, axis=-1)
    # Add batch and steps dimensions
    prev_image = np.reshape(prev_image, [1, 1] + list(prev_image.shape))
    prev_action = np.reshape(prev_action, [1, 1] + list(prev_action.shape))
    prev_reward = np.reshape(prev_reward, [1, 1] + list(prev_reward.shape))
    # calculate all proposals in a single batch inference
    prev_image = np.tile(prev_image, [self._proposals, 1, 1, 1, 1])
    prev_action = np.tile(prev_action,
                          [self._proposals, 1] + [1] * (prev_action.ndim - 2))
    prev_reward = np.tile(prev_reward,
                          [self._proposals, 1] + [1] * (prev_action.ndim - 2))
    self._state = self._observe_fn(prev_image, prev_action, prev_reward,
                                   self._state)
    dist = self.initialize_distribution()
    for _ in range(self._iterations):
      traj_proposals = self.sample_actions(dist)
      rewards, predictions = self.generate_rewards(traj_proposals)
      dist = self.fit_dist(rewards, traj_proposals)

    idx = np.argmax(rewards)
    best_trajectory = traj_proposals[idx]
    predictions = {key: value[idx, 0] for key, value in predictions.items()}
    return best_trajectory[0], predictions


@gin.configurable
class MPPI(CEM):
  """Implements MPPI algorithm from https://arxiv.org/abs/1909.11652."""

  def __init__(
      self,
      predict_fn: Callable[[np.ndarray, Any], Dict[Text, np.ndarray]],
      observe_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, Any], Any],
      reset_fn: Callable[[Any, int], Any],
      task: tasks.Task,
      objective_fn: Callable[[Dict[Text, np.ndarray]], np.ndarray],
      horizon: int,
      iterations: int,
      proposals: int,
      fraction: float,
      beta: List[float],
      gamma: float,
  ):
    """Initialize a MPPI planner that queries a world model.

    Args:
      predict_fn: a callable with the following positional arguments:
          * planned_actions: a [batch, steps, action_dims] ndarray
          * state: the state object returned from observe_fn.
        This method is expected to return:
          * predictions: a dictionary with possibly the following entries:
            * "image": [batch, steps, height, width, channels] ndarray of the
              model predictions of the state of the world conditioned actions.
            * "reward": [batch, steps, 1] ndarray of the reward predictions.
      observe_fn: a callable with the following positional arguments:
          * last_images: a [batch, steps, height, width, channels] ndarray
          * last_actions: a [batch, steps, action_dims] ndarray
          * last_reward: a [batch, steps, 1] ndarray
          * state: the previously returned `state` or {} for the start of
            episode
        This method is expected to return:
          * state: Anything the model needs to continue predicting current
            episode.
      reset_fn: a callable with the following positional arguments:
          * state: the state object
          * batch_size: the batch size.
        This method is expected to return:
          * state: the new state with cleared history.
      task: The task of type `tasks.Task`.
      objective_fn: a callable with the following positional arguments:
          * predictions: a dictionary possibly containing "image" and "reward"
            that will come from the `predict_fn`
        This method is expected to return:
          * rewards: a [batch, steps, 1] ndarray of containing scalar rewards.
      horizon: the planning horizon to specify how many steps into the future to
        consider for choosing the right plan.
      iterations: How many iterations to estimate the final distribution.
      proposals: How many proposals to consider in each iteration.
      fraction: The percentage of proposals to select with the highest score to
        fit the distribution.
      beta: Coefficients for correlated noise during sampling.
      gamma: Weight for top_k rewards when fitting the gaussian.
    """
    super(MPPI,
          self).__init__(predict_fn, observe_fn, reset_fn, task, objective_fn,
                         horizon, iterations, proposals, fraction, False)
    self._beta = beta
    self._gamma = gamma

  def _sample_continuous_actions(self, means, covs):
    """Samples actions with correlated noise using MPPI."""

    all_actions = []
    # The variables u and n refer to the variables in Equations 3 and 4 in the
    # MPPI paper (https://arxiv.org/pdf/1909.11652.pdf).
    u = []
    n = []
    for (mean, cov) in zip(means, covs):
      u.append(
          np.random.multivariate_normal(
              np.zeros_like(mean), cov, (self._proposals,)))
    n.append(self._beta[0] * u[0])
    n.append((self._beta[0] * u[1]) + (self._beta[1] * n[0]))
    for i in range(2, len(u)):
      n.append((self._beta[0] * u[i]) + (self._beta[1] * n[-1]) +
               (self._beta[2] * n[-2]))

    for i in range(len(means)):
      actions = n[i] + means[i]
      actions = np.clip(actions, self._action_space.low,
                        self._action_space.high)
      all_actions.extend([actions])
    all_actions = np.array(all_actions)
    traj_proposals = np.transpose(all_actions, (1, 0, 2))
    traj_proposals = traj_proposals.astype(np.float32)
    return traj_proposals

  def _fit_gaussian(self, rewards, traj_proposals):
    """Re-fits a Gaussian to the best actions."""
    top_k = int(self._fraction * self._proposals)
    indices = np.squeeze(
        np.argpartition(rewards, -top_k, axis=0), axis=-1)[-top_k:]
    best_trajectories = traj_proposals[indices]
    best_rewards = np.array(rewards)[indices]
    actions_to_fit = np.transpose(best_trajectories[0:top_k, 0:self._horizon],
                                  (1, 0, 2))
    means = []
    covs = []

    weights = np.exp(self._gamma * best_rewards)
    weights = weights / np.sum(weights)

    for i in range(self._horizon):
      weighted_actions = (weights.T * actions_to_fit[i].T).T
      means.append(np.mean(weighted_actions, axis=0))
      covs.append(np.cov(actions_to_fit[i].T))
    return means, covs

  def _sample_discrete_actions(self, pvals):
    raise NotImplementedError("discrete actions is not implemented for MPPI")

  def _fit_multinomial(self, rewards, traj_proposals):
    raise NotImplementedError("discrete actions is not implemented for MPPI")


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
class TensorFlowCEM(CEM):
  """Cross entropy method implemented in TensorFlow."""

  @tf.function
  def initialize_distribution(self):
    """Returns initial mean and covariance."""
    # start with a uniform distribution to sample from.
    means = tf.stack(
        [(self._action_space.high + self._action_space.low) / 2.0] *
        self._horizon,
        axis=0)
    stdevs = tf.stack(
        [(self._action_space.high - self._action_space.low) / 2.0] *
        self._horizon,
        axis=0)
    return means, stdevs

  @tf.function
  def sample_continuous_actions(self, means, stdevs):
    """Samples actions from a multivariate Gaussian."""
    actions = tfp.distributions.MultivariateNormalDiag(
        loc=means, scale_diag=stdevs).sample([self._proposals])
    actions = tf.clip_by_value(actions, self._action_space.low,
                               self._action_space.high)
    return actions

  @tf.function
  def generate_rewards(self, traj_proposals, state):
    """Given a set of actions, outputs the corresponding rewards."""
    predictions = self._predict_fn(traj_proposals, state)
    traj_rewards = self._objective_fn(predictions)
    return traj_rewards, predictions

  @tf.function
  def fit_gaussian(self, rewards, traj_proposals):
    """Re-fits a Gaussian to the best actions."""
    top_k = int(self._fraction * self._proposals)
    rewards = tf.squeeze(rewards, axis=-1)
    _, indices = tf.nn.top_k(rewards, top_k, sorted=False)
    best_actions = tf.gather(traj_proposals, indices)
    if self._weighted:
      weights = tf.gather(rewards, indices)
      means, variance = tf.nn.weighted_moments(
          best_actions, axes=0, frequency_weights=weights[..., None, None])
    else:
      means, variance = tf.nn.moments(best_actions, axes=0)
    stdevs = tf.sqrt(variance + 1e-6)
    return means, stdevs

  def __call__(self, prev_image, prev_action, prev_reward):
    action, predictions, state = self.__tf_call__(
        tf.convert_to_tensor(prev_image), tf.convert_to_tensor(prev_action),
        tf.convert_to_tensor(prev_reward), self._state)
    self._state = state
    return action, predictions

  @tf.function
  def __tf_call__(self, prev_image, prev_action, prev_reward, state):
    prev_reward = tf.expand_dims(prev_reward, axis=-1)
    if prev_action.get_shape().ndims == 0:
      prev_action = tf.expand_dims(prev_action, axis=-1)
    # Add batch and steps dimensions
    prev_image = tf.reshape(prev_image,
                            [1, 1] + prev_image.get_shape().as_list())
    prev_action = tf.reshape(prev_action,
                             [1, 1] + prev_action.get_shape().as_list())
    prev_reward = tf.reshape(prev_reward,
                             [1, 1] + prev_reward.get_shape().as_list())
    # calculate all proposals in a single batch inference
    prev_image = tf.tile(prev_image, [self._proposals, 1, 1, 1, 1])
    prev_action = tf.tile(prev_action, [self._proposals, 1] + [1] *
                          (prev_action.get_shape().ndims - 2))
    prev_reward = tf.tile(prev_reward, [self._proposals, 1] + [1] *
                          (prev_action.get_shape().ndims - 2))
    state = self._observe_fn(prev_image, prev_action, prev_reward, state)
    means, stdevs = self.initialize_distribution()

    @tf.function
    def iteration(means_stdevs, _):
      means, stdevs = means_stdevs
      traj_proposals = self.sample_continuous_actions(means, stdevs)
      rewards, _ = self.generate_rewards(traj_proposals, state)
      means, stdevs = self.fit_gaussian(rewards, traj_proposals)
      return means, stdevs

    means, stdevs = static_scan(iteration, tf.range(self._iterations - 1),
                                (means, stdevs))
    means = means[-1]
    stdevs = stdevs[-1]
    traj_proposals = self.sample_continuous_actions(means, stdevs)
    rewards, predictions = self.generate_rewards(traj_proposals, state)
    rewards = tf.squeeze(rewards, axis=-1)
    index = tf.arg_max(rewards, dimension=0)
    best_action = traj_proposals[index, 0]
    predictions = {key: value[index, 0] for key, value in predictions.items()}
    return best_action, predictions, state


@gin.configurable
class RandomColdStart(Planner):
  """Random planner for the first few episodes."""

  def __init__(self,
               task=gin.REQUIRED,
               random_episodes=gin.REQUIRED,
               base_planner=gin.REQUIRED):
    super(RandomColdStart, self).__init__()
    self._planner = base_planner
    self._random_budget = random_episodes + 1
    self._action_space = task.create_env().action_space

  def set_episode_num(self, episode_num):
    self._random_budget = max(0, self._random_budget - episode_num)
    self._planner.set_episode_num(episode_num)

  def reset(self, **kwargs):
    self._planner.reset(**kwargs)
    self._random_budget = max(0, self._random_budget - 1)

  def __call__(self, prev_image, prev_action, prev_reward):
    if self._random_budget > 0:
      action = np.array(self._action_space.sample())
      if action.ndim == 0:
        action = np.expand_dims(action, axis=-1)
      return action, {}
    else:
      return self._planner(
          prev_image=prev_image,
          prev_action=prev_action,
          prev_reward=prev_reward)


@gin.configurable
class EpsilonGreedy(Planner):
  """Epsilon greedy policy wrapper for a base planner."""

  def __init__(self,
               task=gin.REQUIRED,
               base_planner=gin.REQUIRED,
               epsilon=gin.REQUIRED):
    super(EpsilonGreedy, self).__init__()
    self._action_space = task.create_env().action_space
    self._planner = base_planner
    self._epsilon = epsilon

  def __call__(self, prev_image, prev_action, prev_reward):
    rand = np.random.random()
    if rand > self._epsilon:
      return self._planner(
          prev_image=prev_image,
          prev_action=prev_action,
          prev_reward=prev_reward)
    else:
      return self._action_space.sample(), {}

  def set_episode_num(self, episode_num):
    self._planner.set_episode_num(episode_num)

  def reset(self, **kwargs):
    self._planner.reset(**kwargs)


@gin.configurable
class GaussianRandomNoise(Planner):
  """Adds a random Gaussian noise to the output of a base planner."""

  def __init__(self,
               task=gin.REQUIRED,
               base_planner=gin.REQUIRED,
               stdev=gin.REQUIRED):
    super(GaussianRandomNoise, self).__init__()
    self._action_space = task.create_env().action_space
    self._planner = base_planner
    self._stdev = stdev

  def __call__(self, prev_image, prev_action, prev_reward):
    noise = np.random.normal(scale=self._stdev, size=prev_action.shape)
    action, preds = self._planner(
        prev_image=prev_image, prev_action=prev_action, prev_reward=prev_reward)
    action += noise
    action = np.clip(action, self._action_space.low, self._action_space.high)
    return action, preds

  def set_episode_num(self, episode_num):
    self._planner.set_episode_num(episode_num)

  def reset(self, **kwargs):
    self._planner.reset(**kwargs)


@gin.configurable
class Randomizer(Planner):
  """Randomize actions chosen by the base planner with a given probability."""

  def __init__(self,
               task=gin.REQUIRED,
               base_planner=gin.REQUIRED,
               rand_prob=gin.REQUIRED):
    super(Randomizer, self).__init__()
    self._action_space = task.create_env().action_space
    self._planner = base_planner
    self.rand_prob = rand_prob

  def __call__(self, prev_image, prev_action, prev_reward):
    should_randomize = np.random.rand() < self.rand_prob
    if should_randomize:
      return np.array(self._action_space.sample()), {}
    else:
      action, preds = self._planner(
          prev_image=prev_image,
          prev_action=prev_action,
          prev_reward=prev_reward)
      return action, preds

  def set_episode_num(self, episode_num):
    self._planner.set_episode_num(episode_num)

  def reset(self, **kwargs):
    self._planner.reset(**kwargs)
