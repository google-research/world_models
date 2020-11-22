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

"""Environment wrappers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
from PIL import Image
from world_models.utils import nested


class ObservationDict(gym.Wrapper):
  """Changes the observation space to be a dict."""

  def __init__(self, env, key='observ'):
    self._key = key
    self.env = env

  def __getattr__(self, name):
    return getattr(self.env, name)

  @property
  def observation_space(self):
    spaces = {self._key: self.env.observation_space}
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return self.env.action_space

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    obs = {self._key: np.array(obs)}
    return obs, reward, done, info

  def reset(self):
    obs = self.env.reset()
    obs = {self._key: np.array(obs)}
    return obs


class ActionRepeat(gym.Wrapper):
  """Repeats the same action `n` times and returns the last step results."""

  def __init__(self, env, n):
    super(ActionRepeat, self).__init__(env)
    assert n >= 1
    self._n = n

  def __getattr__(self, name):
    return getattr(self.env, name)

  def step(self, action):
    done = False
    total_reward = 0
    current_step = 0
    while current_step < self._n and not done:
      observ, reward, done, info = self.env.step(action)
      total_reward += reward
      current_step += 1
    return observ, total_reward, done, info


class ActionNormalize(gym.Env):
  """Normalizes the action space."""

  def __init__(self, env):
    self._env = env
    self._mask = np.logical_and(
        np.isfinite(env.action_space.low), np.isfinite(env.action_space.high))
    self._low = np.where(self._mask, env.action_space.low, -1)
    self._high = np.where(self._mask, env.action_space.high, 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    return gym.spaces.Box(low, high, dtype=np.float32)

  def step(self, action):
    original = (action + 1) / 2 * (self._high - self._low) + self._low
    original = np.where(self._mask, original, action)
    return self._env.step(original)

  def reset(self):
    return self._env.reset()

  def render(self, mode='human'):
    return self._env.render(mode=mode)


class MaximumDuration(gym.Wrapper):
  """Force sets `done` after the specified duration."""

  def __init__(self, env, duration):
    super(MaximumDuration, self).__init__(env)
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self.env, name)

  def step(self, action):
    if self._step is None:
      raise RuntimeError('Must reset environment.')
    observ, reward, done, info = self.env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      self._step = None
    return observ, reward, done, info

  def reset(self):
    self._step = 0
    return self.env.reset()


class MinimumDuration(gym.Wrapper):
  """Force resets `done` before the specified duration."""

  def __init__(self, env, duration):
    super(MinimumDuration, self).__init__(env)
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self.env, name)

  def step(self, action):
    observ, reward, done, info = self.env.step(action)
    self._step += 1
    if self._step < self._duration:
      done = False
    return observ, reward, done, info

  def reset(self):
    self._step = 0
    return self.env.reset()


class ConvertTo32Bit(gym.Wrapper):
  """Converts observation and rewards to int/float32."""

  def __getattr__(self, name):
    return getattr(self.env, name)

  def step(self, action):
    observ, reward, done, info = self.env.step(action)
    observ = nested.map(self._convert_observ, observ)
    reward = self._convert_reward(reward)
    return observ, reward, done, info

  def reset(self):
    observ = self.env.reset()
    observ = nested.map(self._convert_observ, observ)
    return observ

  def _convert_observ(self, observ):
    if not np.isfinite(observ).all():
      raise ValueError('Infinite observation encountered.')
    if observ.dtype == np.float64:
      return observ.astype(np.float32)
    if observ.dtype == np.int64:
      return observ.astype(np.int32)
    return observ

  def _convert_reward(self, reward):
    if not np.isfinite(reward).all():
      raise ValueError('Infinite reward encountered.')
    return np.array(reward, dtype=np.float32)


class RenderObservation(gym.Env):
  """Changes the observation space to rendered frames."""

  def __init__(self, env, size=(64, 64), dtype=np.uint8, key='image'):
    assert isinstance(env.observation_space, gym.spaces.Dict)
    self.env = env
    self._size = size
    self._dtype = dtype
    self._key = key

  def __getattr__(self, name):
    return getattr(self.env, name)

  @property
  def observation_space(self):
    high = {np.uint8: 255, np.float: 1.0}[self._dtype]
    image = gym.spaces.Box(0, high, self._size + (3,), dtype=self._dtype)
    spaces = self.env.observation_space.spaces.copy()
    assert self._key not in spaces
    spaces[self._key] = image
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return self.env.action_space

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    obs[self._key] = self._render_image()
    return obs, reward, done, info

  def reset(self):
    obs = self.env.reset()
    obs[self._key] = self._render_image()
    return obs

  def _render_image(self):
    """Renders the environment and processes the image."""
    image = self.env.render('rgb_array')
    if image.shape[:2] != self._size:
      image = np.array(Image.fromarray(image).resize(self._size))
    if self._dtype and image.dtype != self._dtype:
      if image.dtype in (np.float32, np.float64) and self._dtype == np.uint8:
        image = (image * 255).astype(self._dtype)
      elif image.dtype == np.uint8 and self._dtype in (np.float32, np.float64):
        image = image.astype(self._dtype) / 255
      else:
        message = 'Cannot convert observations from {} to {}.'
        raise NotImplementedError(message.format(image.dtype, self._dtype))
    return image

class DeepMindEnv(gym.Env):
  """Wrapper for deepmind MuJoCo environments to expose gym env methods."""
  metadata = {'render.modes': ['rgb_array']}
  reward_range = (-np.inf, np.inf)

  def __init__(self, env, render_size=(64, 64), camera_id=0):
    self._env = env
    self._render_size = render_size
    self._camera_id = camera_id

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    components = {}
    for key, value in self._env.observation_spec().items():
      components[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    return gym.spaces.Dict(components)

  @property
  def action_space(self):
    action_spec = self._env.action_spec()
    return gym.spaces.Box(
        action_spec.minimum, action_spec.maximum, dtype=np.float32)

  def step(self, action):
    time_step = self._env.step(action)
    obs = dict(time_step.observation)
    reward = time_step.reward or 0
    done = time_step.last()
    info = {'discount': time_step.discount}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    return dict(time_step.observation)

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    del args  # Unused
    del kwargs  # Unused
    return self._env.physics.render(
        *self._render_size, camera_id=self._camera_id)

  def get_state(self):
    return (
        np.array(self.physics.data.qpos),
        np.array(self.physics.data.qvel),
        np.array(self.physics.data.ctrl),
        np.array(self.physics.data.act))

  def set_state(self, state):
    with self.physics.reset_context():
      self.physics.data.qpos[:] = state[0]
      self.physics.data.qvel[:] = state[1]
      self.physics.data.ctrl[:] = state[2]
      self.physics.data.act[:] = state[3]
