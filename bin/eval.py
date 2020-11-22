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

"""Evaluate a world model on an offline dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Callable, Dict, Text

from absl import app
from absl import flags
from absl import logging
import gin
import numpy as np
import tensorflow.compat.v1 as tf

from world_models.loops import train_eval
from world_models.planners import planners
from world_models.simulate import simulate
from world_models.tasks import tasks
from world_models.utils import npz

FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", None, "Model checkpoint directory.")
flags.DEFINE_string("data_dir", None, "data directory.")
flags.DEFINE_string("output_dir", None, "output directory.")
flags.DEFINE_multi_string(
    "config_path", None,
    "Newline separated list of paths to a world models gin configs.")
flags.DEFINE_multi_string("config_param", None,
                          "Newline separated list of Gin parameter bindings.")
flags.DEFINE_bool("enable_eager", True, "Enable eager execution mode.")
flags.DEFINE_integer("num_virtual_gpus", -1, "If >1, enables virtual gpus.")
flags.DEFINE_boolean("train", False, "Train the model on data before eval.")
flags.DEFINE_string("train_data_dir", None, "train data path.")


def frame_error(predicted_frames, ground_truth_frames):
  """Frame prediction error as average L2 norm between pixels."""
  batch, prediction_horizon = predicted_frames.shape[:2]
  return np.mean(
      np.linalg.norm(
          np.reshape(
              np.asarray(predicted_frames, dtype=np.float32),
              [batch, prediction_horizon, -1, 1]) - np.reshape(
                  np.asarray(ground_truth_frames, dtype=np.float32),
                  [batch, prediction_horizon, -1, 1]),
          axis=-1),
      axis=-1)


def reward_error(predicted_rewards, ground_truth_rewards):
  """Reward prediction error as L2 norm."""
  return np.linalg.norm(predicted_rewards - ground_truth_rewards, axis=-1)


@gin.configurable(
    blacklist=["eval_dir", "train_dir", "model_dir", "result_dir"])
def offline_evaluate(
    predict_fn: Callable[[np.ndarray, Any], Dict[Text, np.ndarray]],
    observe_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, Any], Any],
    reset_fn: Callable[..., Any],
    train_fn: Callable[[Text], None] = None,
    train_dir: Text = None,
    enable_train: bool = False,
    train_eval_iterations: int = 0,
    online_eval_task: tasks.Task = None,
    online_eval_planner: planners.Planner = None,
    online_eval_episodes: int = 0,
    eval_dir: Text = None,
    model_dir: Text = None,
    result_dir: Text = None,
    episode_length: int = None,
    num_episodes: int = 100,
    prediction_horizon: int = 1,
    batch: int = 128):
  """offline model evaluation."""
  assert eval_dir, "eval_dir is required"
  assert model_dir, "model_dir is required"
  assert result_dir, "result_dir is required"
  assert episode_length, "episode_length is required"

  if enable_train:
    assert train_dir, "train_dir is required for training"
    assert train_eval_iterations, ("train_eval_iterations is required for "
                                   "training")
    for i in range(train_eval_iterations):
      train_fn(train_dir)
      result_dir_at_step = os.path.join(result_dir, "%d" % i)
      eval_once(
          result_dir=result_dir_at_step,
          eval_dir=eval_dir,
          episode_length=episode_length,
          prediction_horizon=prediction_horizon,
          batch=batch,
          num_episodes=num_episodes,
          reset_fn=reset_fn,
          observe_fn=observe_fn,
          predict_fn=predict_fn)
      if online_eval_episodes:
        summary_dir = os.path.join(result_dir, "online_eval")
        episodes, predictions, score = simulate.simulate(
            online_eval_task, online_eval_planner, online_eval_episodes)
        train_eval.visualize(summary_dir, i, episodes, predictions,
                             {"score": score})
  else:
    eval_once(
        result_dir=result_dir,
        eval_dir=eval_dir,
        episode_length=episode_length,
        prediction_horizon=prediction_horizon,
        batch=batch,
        num_episodes=num_episodes,
        reset_fn=reset_fn,
        observe_fn=observe_fn,
        predict_fn=predict_fn)


def eval_once(result_dir, eval_dir, episode_length, prediction_horizon, batch,
              num_episodes, reset_fn, observe_fn, predict_fn):
  """Run offline eval once and store the results in `result_dir`."""
  dataset = npz.load_dataset_from_directory(eval_dir, episode_length, batch)
  iterator = dataset.as_numpy_iterator()

  state = None
  reward_path = os.path.join(result_dir, "rewards")
  reward_error_at_prediction_horizon = np.zeros((prediction_horizon))
  frame_error_at_prediction_horizon = np.zeros((prediction_horizon))
  logging.info("Staring evaluation")
  predictions = {}
  for b, episodes in enumerate(iterator):
    if b * batch >= num_episodes:
      break
    if episodes["image"].dtype != np.uint8:
      episodes["image"] = np.clip(episodes["image"] * 255, 0,
                                  255).astype(np.uint8)
    state = reset_fn(state=state, proposals=batch)
    for i in range(episode_length - prediction_horizon):
      timestep = {key: value[:, i:i + 1] for key, value in episodes.items()}
      frame = timestep["image"]
      reward = timestep["reward"]
      action = timestep["action"]
      future_actions = episodes["action"][:, i:i + prediction_horizon]
      future_frames = episodes["image"][:, i:i + prediction_horizon]
      future_rewards = episodes["reward"][:, i:i + prediction_horizon]
      state = observe_fn(frame, action, reward, state)
      predictions = predict_fn(future_actions, state)
      if "reward" in predictions:
        npz.save_dictionary(
            {
                "pred": predictions["reward"],
                "true": future_rewards
            }, reward_path)
        reward_error_at_prediction_horizon += np.sum(
            reward_error(predictions["reward"], future_rewards), axis=0)
      if "image" in predictions:
        frame_error_at_prediction_horizon += np.sum(
            frame_error(predictions["image"], future_frames), axis=0)
    logging.info("Finished evaluation on %d episodes", batch)

  reward_error_at_prediction_horizon /= num_episodes * (
      episode_length - prediction_horizon)
  frame_error_at_prediction_horizon /= num_episodes * (
      episode_length - prediction_horizon)
  logging.info("Finished evaluation")
  results = {}
  if "reward" in predictions:
    logging.info(
        "Average reward L2 norm error for different prediction horizons: %s",
        reward_error_at_prediction_horizon)
    results["reward_error"] = reward_error_at_prediction_horizon
  else:
    logging.info("predict_fn does not predict rewards."
                 " L2 norm on reward prediction could not be calculated.")
  if "image" in predictions:
    logging.info(
        "Average frame L2 norm error for different prediction horizons: %s",
        frame_error_at_prediction_horizon)
    results["image_error"] = frame_error_at_prediction_horizon
  else:
    logging.info("predict_fn does not predict frames."
                 " L2 norm on frame prediction could not be calculated.")
  npz.save_dictionary(results, result_dir)


def main(argv):
  del argv  # Unused
  if FLAGS.enable_eager:
    tf.enable_eager_execution()

  config_params = FLAGS.config_param or []
  config_params += [
      "model_dir='%s'" % FLAGS.model_dir,
      "episodes_dir='%s'" % FLAGS.output_dir
  ]
  gin.parse_config_files_and_bindings(FLAGS.config_path, config_params)

  if FLAGS.num_virtual_gpus > -1:
    gpus = tf.config.experimental.list_physical_devices("GPU")

    total_gpu_mem_limit = 8192
    per_gpu_mem_limit = total_gpu_mem_limit / FLAGS.num_virtual_gpus
    virtual_gpus = [
        tf.config.experimental.VirtualDeviceConfiguration(
            memory_limit=per_gpu_mem_limit)
    ] * FLAGS.num_virtual_gpus
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0], virtual_gpus)
    logical_gpus = tf.config.experimental.list_logical_devices("GPU")
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

  offline_evaluate(  # pylint:disable=no-value-for-parameter
      result_dir=FLAGS.output_dir,
      model_dir=FLAGS.model_dir,
      eval_dir=FLAGS.data_dir,
      train_dir=FLAGS.train_data_dir,
      enable_train=FLAGS.train)


if __name__ == "__main__":
  flags.mark_flags_as_required(["output_dir"])
  flags.mark_flags_as_required(["config_path"])
  app.run(main)
