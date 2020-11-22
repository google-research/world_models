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
"""A simple example of an agent implemented in Jax."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from flax import jax_utils
from flax import nn
from flax import optim
from flax import struct
from flax.training import checkpoints
import gin
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np

from world_models.utils import npz


# pylint:disable=missing-function-docstring


def configurable_module(module):
  if not issubclass(module, nn.Module):
    raise ValueError("this decorator can only be used on flax.nn.Module class.")

  def wrapper(**kwargs):
    return module.partial(**kwargs)

  wrapper.__name__ = module.__name__

  return gin.configurable(wrapper)


@struct.dataclass
class TrainState:
  step: int
  optimizer: optim.Optimizer


@configurable_module
class JaxPureReward(nn.Module):
  """A simple pure reward predictor with Jax."""

  def apply(self, actions, num_layers, hidden_dims):
    timesteps = actions.shape[1]
    # flatten time into batch
    actions = jnp.reshape(actions, (-1,) + actions.shape[2:])
    # embed actions
    x = nn.Dense(actions, hidden_dims)
    for _ in range(num_layers):
      x = nn.Dense(x, hidden_dims)
      x = nn.LayerNorm(x)
      x = nn.relu(x)
    x = nn.Dense(x, 1)
    x = jnp.reshape(x, (-1, timesteps, 1))
    return x


@gin.configurable
def create_model(module, task):
  """Initializes the model and returns it."""
  action_space = task.create_env().action_space
  rng_key = jax.random.PRNGKey(0)
  _, params = module.init_by_shape(rng_key,
                                   [((1, 1) + action_space.shape, jnp.float32)])
  model = nn.Model(module, params)
  return model


@gin.configurable
def reset_fn(**kwargs):
  del kwargs  # Unused
  return jax_utils.replicate({})


@gin.configurable(blacklist=["images", "actions", "rewards", "state"])
def observe_fn(images, actions, rewards, state):
  del images, actions, rewards  # Unused
  return state


@gin.configurable
def create_predict_fn(model):
  @functools.partial(jax.pmap)
  def predict(actions):
    return model(actions)

  def predict_fn(actions, state):
    del state  # Unused
    actions = jnp.reshape(actions,
                          (jax.local_device_count(), -1) + actions.shape[-2:])
    predictions = predict(actions)
    predictions = jnp.reshape(predictions, (-1,) + predictions.shape[-2:])
    return {"reward": jax.device_get(predictions)}

  return predict_fn


@gin.configurable
def create_train_fn(model, model_dir, duration, batch, train_steps,
    learning_rate):
  optimizer = optim.Adam()
  opt = optimizer.create(model)
  state = TrainState(optimizer=opt, step=0)  # pytype:disable=wrong-keyword-args
  state = checkpoints.restore_checkpoint(model_dir, state)
  state = jax_utils.replicate(state)
  iterator = None

  @functools.partial(jax.pmap, axis_name="batch")
  def train_step(obs, state):
    actions = obs["action"]
    rewards = obs["reward"]
    step = state.step
    optimizer = state.optimizer

    def loss(model):
      predictions = model(actions)
      l = (rewards - predictions) ** 2
      l = jnp.mean(l)
      return l

    grad_fn = jax.value_and_grad(loss)
    l, grads = grad_fn(state.optimizer.target)
    grads = lax.pmean(grads, axis_name="batch")
    new_optimizer = optimizer.apply_gradient(grads, learning_rate=learning_rate)
    new_state = state.replace(step=step + 1, optimizer=new_optimizer)
    return new_state, l

  def train(data_path):
    nonlocal iterator
    nonlocal state

    if iterator is None:
      dataset = npz.load_dataset_from_directory(data_path, duration, batch)
      iterator = dataset.make_one_shot_iterator()
      iterator = map(
          lambda x: jax.tree_map(
              lambda x: np.reshape(
                  x, (jax.local_device_count(), -1) + x.numpy().shape[1:]),
              x),
          iterator)
      iterator = jax_utils.prefetch_to_device(iterator, 2)

    for _ in range(train_steps):
      obs = next(iterator)
      state, l = train_step(obs, state)
    local_state = get_first_device(state)
    l = get_first_device(l)
    checkpoints.save_checkpoint(model_dir, local_state, local_state.step)

  return train


def get_first_device(value):
  return jax.tree_map(lambda x: x[0], value)
