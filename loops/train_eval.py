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

"""Train, simulate and evaluate loop with a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time
from typing import Text, Callable

from absl import flags
from absl import logging
import gin
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2.summary as tfs

from world_models.planners import planners
from world_models.simulate import simulate
from world_models.tasks import tasks
from world_models.utils import npz
from world_models.utils import visualization


FLAGS = flags.FLAGS
flags.DEFINE_string("tpu", None, "gRPC address of the TPU worker.")


@gin.configurable
def get_tpu_strategy():
  """Creates a TPUStrategy for distribution on TPUs."""
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(FLAGS.tpu)
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  strategy = tf.distribute.experimental.TPUStrategy(resolver)
  return strategy


def visualize(summary_dir, global_step, episodes, predictions, scalars):
  """Visualizes the episodes in TensorBoard."""
  if tf.executing_eagerly():
    writer = tfs.create_file_writer(summary_dir)
    with writer.as_default():
      videos = np.stack([e["image"] for e in episodes])
      video_summary = visualization.py_gif_summary(
          tag="episodes/video", images=videos, max_outputs=20, fps=20)
      tfs.experimental.write_raw_pb(video_summary, step=global_step)
      for k in scalars:
        tfs.scalar(name="episodes/%s" % k, data=scalars[k], step=global_step)
      if "image" in predictions[0]:
        videos = np.stack([e["image"] for e in predictions])
        video_summary = visualization.py_gif_summary(
            tag="episodes/video_prediction",
            images=videos,
            max_outputs=6,
            fps=20)
        tfs.experimental.write_raw_pb(video_summary, step=global_step)
      if "reward" in predictions[0]:
        rewards = np.stack([e["reward"][1:] for e in episodes])
        predicted_rewards = np.stack([p["reward"] for p in predictions])
        signals = np.stack([rewards, predicted_rewards], axis=1)
        signals = signals[:, :, :, 0]
        visualization.py_plot_1d_signal(
            name="episodes/reward",
            signals=signals,
            labels=["reward", "prediction"],
            max_outputs=6,
            step=global_step)
        reward_dir = os.path.join(summary_dir, "rewards")
        rewards_to_save = {"true": rewards, "pred": predicted_rewards}
        npz.save_dictionary(rewards_to_save, reward_dir)
  else:
    summary_writer = tf.summary.FileWriter(summary_dir)
    for k in scalars:
      s = tf.Summary()
      s.value.add(tag="episodes/" + k, simple_value=scalars[k])
      summary_writer.add_summary(s, global_step)
    videos = np.stack([e["image"] for e in episodes])
    video_summary = visualization.py_gif_summary(
        tag="episodes/video", images=videos, max_outputs=20, fps=30)
    summary_writer.add_summary(video_summary, global_step)
    summary_writer.flush()


def simulate_and_persist(task, planner, num_episodes, data_dir):
  """Runs the simulation, persists the results and returns the output."""
  start_time = time.time()
  episodes, predictions, score = simulate.simulate(
      task=task, planner=planner, num_episodes=num_episodes)
  simulate_time = time.time() - start_time
  npz.save_dictionaries(episodes, data_dir)
  return episodes, predictions, score, simulate_time


@gin.configurable(blacklist=["offline_train", "offline_train_data_dir"])
def train_eval_loop(task: tasks.Task = gin.REQUIRED,
                    train_planner: planners.Planner = gin.REQUIRED,
                    eval_planner: planners.Planner = gin.REQUIRED,
                    train_fn: Callable[[Text], None] = gin.REQUIRED,
                    num_train_episodes_per_iteration: int = 1,
                    eval_every_n_iterations: int = 1,
                    num_iterations: int = 1,
                    episodes_dir: Text = None,
                    model_dir: Text = None,
                    offline_train: bool = False,
                    offline_train_data_dir: Text = None):
  """train and eval loop."""
  assert episodes_dir, "episodes_dir is required"
  assert model_dir, "model_dir is required"

  # Load iteration info if exists
  iterations_info = []
  iterations_datafile = os.path.join(episodes_dir, "info.json")
  if tf.io.gfile.exists(iterations_datafile):
    with tf.io.gfile.GFile(iterations_datafile, "r") as fp:
      iterations_info = json.load(fp)
  current_iteration = len(iterations_info)
  train_planner.set_episode_num(current_iteration *
                                num_train_episodes_per_iteration)

  logging.info("Starting the simulation of %s", task.name)
  for i in range(current_iteration, num_iterations):
    logging.info("=" * 30)
    logging.info("Starting Iteration %08d", i)
    logging.info("=" * 30)
    iteration_info = {"iteration_num": i}
    if num_train_episodes_per_iteration:
      iteration_start_time = time.time()
      if offline_train:
        assert offline_train_data_dir, ("offline_train_data_dir is required in "
                                        "offline training mode")
        train_dir = offline_train_data_dir
      else:
        train_dir = os.path.join(episodes_dir, "train")
        episodes, predictions, score, simulate_time = simulate_and_persist(
            task, train_planner, num_train_episodes_per_iteration, train_dir)
        logging.info("Average score during training at iteration %d was: %f", i,
                     score)

      logging.info("Training model at iteration %d", i)
      training_start_time = time.time()
      train_fn(train_dir)
      train_time = time.time() - training_start_time
      iteration_time = time.time() - iteration_start_time

      if not offline_train:
        scalars = {
            "score": score,
            "train_time": train_time,
            "simulate_time": simulate_time,
            "iteration_time": iteration_time,
            "iterations_per_hour": 3600.0 / iteration_time,
        }
        visualize(
            os.path.join(model_dir, "train"), i, episodes, predictions, scalars)

    if eval_every_n_iterations and i % eval_every_n_iterations == 0:
      eval_dir = os.path.join(episodes_dir, "eval")
      episodes, predictions, score, simulate_time = simulate_and_persist(
          task, eval_planner, 1, eval_dir)
      logging.info("Average score during evaluation at iteration %d was: %f", i,
                   score)

      scalars = {
          "score": score,
          "simulate_time": simulate_time,
      }
      visualize(
          os.path.join(model_dir, "eval"), i, episodes, predictions, scalars)
    iterations_info.append(iteration_info)
    with tf.io.gfile.GFile(iterations_datafile, "w") as fp:
      json.dump(iterations_info, fp)


def get_gin_override_params(output_dir):
  """Get gin config params to override."""
  model_dir = os.path.join(output_dir, "model")
  episodes_dir = os.path.join(output_dir, "episodes")
  args = [
      "model_dir='%s'" % model_dir,
      "episodes_dir='%s'" % episodes_dir,
  ]
  return args
