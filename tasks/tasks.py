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

"""Implementation of tasks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text

from dm_control import suite
import gin
import gym
from world_models.utils import wrappers


class Task(object):
  """Base class for tasks."""

  @property
  def name(self) -> Text:
    raise NotImplementedError

  def create_env(self) -> gym.Env:
    raise NotImplementedError

  def create_nonvisual_env(self) -> gym.Env:
    raise NotImplementedError


@gin.configurable
class DeepMindControl(Task):
  """Deepmind Control Suite environment."""

  def __init__(self,
               domain_name: Text = gin.REQUIRED,
               task_name: Text = gin.REQUIRED,
               camera_id: int = 0,
               max_duration: int = 1000,
               action_repeat: int = 1):
    self._domain_name = domain_name
    self._task_name = task_name
    self._camera_id = camera_id
    self._max_duration = max_duration
    self._action_repeat = action_repeat

  @property
  def name(self):
    return self._domain_name + ":" + self._task_name

  def create_env(self):
    env = suite.load(self._domain_name, self._task_name)
    env = wrappers.DeepMindEnv(env, camera_id=self._camera_id)
    env = wrappers.MaximumDuration(env, duration=self._max_duration)
    env = wrappers.ActionRepeat(env, n=self._action_repeat)
    env = wrappers.RenderObservation(env)
    env = wrappers.ConvertTo32Bit(env)
    return env

  def create_nonvisual_env(self):
    env = suite.load(self._domain_name, self._task_name)
    env = wrappers.DeepMindEnv(env, camera_id=self._camera_id)
    env = wrappers.ActionRepeat(env, n=self._action_repeat)
    return env

  def __reduce__(self):
    args = (self._domain_name, self._task_name, self._camera_id,
            self._max_duration, self._action_repeat)
    return self.__class__, args


@gin.configurable
class Atari(Task):
  """ATARI envs from OpenAI gym."""

  def __init__(self,
               game: Text = gin.REQUIRED,
               width: int = 64,
               height: int = 64,
               channels: int = 3,
               max_duration: int = 1000,
               action_repeat: int = 1):
    import atari_py  # pylint: disable=unused-import, unused-variable
    assert channels == 1 or channels == 3
    self._game = game
    self._width = width
    self._height = height
    self._channels = channels
    self._max_duration = max_duration
    self._action_repeat = action_repeat

  @property
  def name(self):
    return "atari_%s" % self._game

  def create_env(self):
    env = gym.make(self._game)
    env = wrappers.ObservationDict(env)
    env = wrappers.MaximumDuration(env, duration=self._max_duration)
    env = wrappers.ActionRepeat(env, n=self._action_repeat)
    env = wrappers.RenderObservation(env)
    env = wrappers.ConvertTo32Bit(env)
    return env
