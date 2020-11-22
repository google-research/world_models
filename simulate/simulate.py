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

"""Implements the main logic for running a simulation with a world model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from typing import List, Dict, Text, Tuple

from absl import logging
import gin
import numpy as np
from world_models.planners import planners
from world_models.tasks import tasks


def get_timestep_data(obs, reward, action):
  """Form a dict to hold the info of a single timestep in simulation."""
  timestep = {}
  if isinstance(obs, dict):
    for key, value in obs.items():
      timestep[key] = value
  else:
    timestep['image'] = obs
  if action.ndim == 0:
    action = np.expand_dims(action, axis=-1)
  timestep['action'] = action
  timestep['reward'] = np.asarray([reward], dtype=np.float32)
  return timestep


@gin.configurable(blacklist=['episode'])
def approximate_value(episode, gamma=0.99):
  last_value = 0.0
  for d in reversed(episode):
    d['value'] = d['reward'] + gamma*last_value
    last_value = d['value']
  return episode


def get_prediction_data(prediction, prediction_keys):
  """Form a dict to hold predictions of the model for a single timestep."""
  timestep = {}
  if 'image' in prediction_keys:
    timestep['image'] = prediction.get('image', np.zeros(1))
  if 'reward' in prediction_keys:
    timestep['reward'] = prediction.get('reward', np.zeros(1))
  return timestep


def preprocess(image, reward):
  reward = float(reward)
  # TPUs do not support uint8, convert images to int32.
  image = image.astype(np.int32)
  return image, reward


def single_episode(planner, env):
  """Simulate a single episode.

  Args:
    planner: a `Planner` object that uses a world model for planning.
    env: the environment.

  Returns:
    episode: a dictionary with `image`, `action` and `reward` keys
      and np.ndarray values. may include other keys if the env has additional
      information.
  """
  data = []

  planner.reset()
  obs, reward, done = env.reset(), 0.0, False
  obs['image'], reward = preprocess(obs['image'], reward)
  action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
  data.append(get_timestep_data(obs, reward, action))

  prediction_keys = set()
  predictions = []
  step = 0
  start_time = time.time()
  while not done:
    action, prediction = planner(obs['image'], action, reward)
    obs, reward, done, _ = env.step(action)
    obs['image'], reward = preprocess(obs['image'], reward)
    data.append(get_timestep_data(obs, reward, action))
    prediction_keys.update(prediction.keys())
    predictions.append(prediction)
    step += 1
    if step % 10 == 0:
      step_per_sec = step / np.float(time.time() - start_time)
      logging.info('Environment step %d, step per sec %.4f', step, step_per_sec)

  data = approximate_value(data)
  # stack timesteps in the episode to form numpy arrays
  episode = {
      key: np.stack([d[key] for d in data], axis=0) for key in data[0].keys()
  }
  predictions = [get_prediction_data(p, prediction_keys) for p in predictions]
  predictions = {
      key:
      np.stack(np.broadcast_arrays(*(p[key] for p in predictions)), axis=0)
      for key in prediction_keys
  }

  return episode, predictions


def simulate(
    task: tasks.Task, planner: planners.Planner, num_episodes: int
) -> Tuple[List[Dict[Text, np.ndarray]], List[Dict[Text, np.ndarray]], float]:
  """Simulate the world.

  Args:
    task: a `Task` object.
    planner: a `Planner` object that uses a world model for planning.
    num_episodes: how many episodes to simulate. Each episode continues until it
      is done.

  Returns:
    episodes: a list of episodes. each episode is a dictionary with `image`,
      `action` and `reward` keys and np.ndarray values. may include other
      keys if the env has additional information.
    predictions: a list of episode predictions. each episode prediction is a
      dictionary containing the model predictions at every step of the
      environment in that episode.
    score: the average score of a complete episode.
  """
  env = task.create_env()
  episodes = []
  predictions = []
  for i in range(num_episodes):
    logging.info('Starting episode %d', i)
    episode, prediction = single_episode(planner, env)
    episodes.append(episode)
    predictions.append(prediction)

  score = np.mean(list(np.sum(episode['reward']) for episode in episodes))
  return episodes, predictions, score
