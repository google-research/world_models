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

"""Basic layers for video."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import tensorflow.compat.v1.layers as tfl

from world_models.imported_models import common
from tensorflow.contrib import layers as tfcl
from tensorflow.contrib import rnn as contrib_rnn



def encode_to_shape(inputs, shape, scope):
  """Encode the given tensor to given image shape."""
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    w, h = shape[1], shape[2]
    x = inputs
    x = tfl.flatten(x)
    x = tfl.dense(x, w * h, activation=None, name="enc_dense")
    x = tf.reshape(x, (-1, w, h, 1))
    return x


def decode_to_shape(inputs, shape, scope):
  """Encode the given tensor to given image shape."""
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    x = inputs
    x = tfl.flatten(x)
    x = tfl.dense(x, shape[2], activation=None, name="dec_dense")
    x = tf.expand_dims(x, axis=1)
    return x


def basic_lstm(inputs, state, num_units, name=None):
  """Basic LSTM."""
  input_shape = common.shape_list(inputs)
  # reuse parameters across time-steps.
  cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, name=name, reuse=tf.AUTO_REUSE)
  if state is None:
    state = cell.zero_state(input_shape[0], tf.float32)
  outputs, new_state = cell(inputs, state)
  return outputs, new_state


def lstm_cell(inputs,
              state,
              num_units,
              use_peepholes=False,
              cell_clip=0.0,
              initializer=None,
              num_proj=None,
              num_unit_shards=None,
              num_proj_shards=None,
              reuse=None,
              name=None):
  """Full LSTM cell."""
  input_shape = common.shape_list(inputs)
  cell = tf.nn.rnn_cell.LSTMCell(
      num_units,
      use_peepholes=use_peepholes,
      cell_clip=cell_clip,
      initializer=initializer,
      num_proj=num_proj,
      num_unit_shards=num_unit_shards,
      num_proj_shards=num_proj_shards,
      reuse=reuse,
      name=name,
      state_is_tuple=False)
  if state is None:
    state = cell.zero_state(input_shape[0], tf.float32)
  outputs, new_state = cell(inputs, state)
  return outputs, new_state


def conv_lstm_2d(inputs,
                 state,
                 output_channels,
                 kernel_size=5,
                 name=None,
                 spatial_dims=None):
  """2D Convolutional LSTM."""
  input_shape = common.shape_list(inputs)
  batch_size, input_channels = input_shape[0], input_shape[-1]
  if spatial_dims is None:
    input_shape = input_shape[1:]
  else:
    input_shape = spatial_dims + [input_channels]

  cell = contrib_rnn.ConvLSTMCell(
      2, input_shape, output_channels, [kernel_size, kernel_size], name=name)
  if state is None:
    state = cell.zero_state(batch_size, tf.float32)
  outputs, new_state = cell(inputs, state)
  return outputs, new_state


def scheduled_sample_count(ground_truth_x, generated_x, batch_size,
                           scheduled_sample_var):
  """Sample batch with specified mix of groundtruth and generated data points.

  Args:
    ground_truth_x: tensor of ground-truth data points.
    generated_x: tensor of generated data points.
    batch_size: batch size
    scheduled_sample_var: number of ground-truth examples to include in batch.

  Returns:
    New batch with num_ground_truth sampled from ground_truth_x and the rest
    from generated_x.
  """
  num_ground_truth = scheduled_sample_var
  idx = tf.random_shuffle(tf.range(batch_size))
  ground_truth_idx = tf.gather(idx, tf.range(num_ground_truth))
  generated_idx = tf.gather(idx, tf.range(num_ground_truth, batch_size))

  ground_truth_examps = tf.gather(ground_truth_x, ground_truth_idx)
  generated_examps = tf.gather(generated_x, generated_idx)

  output = tf.dynamic_stitch([ground_truth_idx, generated_idx],
                             [ground_truth_examps, generated_examps])
  # if batch size is known set it.
  if isinstance(batch_size, int):
    output.set_shape([batch_size] + common.shape_list(output)[1:])
  return output


def inverse_exp_decay(max_step, min_value=0.01, step=None):
  """Inverse-decay exponentially from min_value to 1.0 reached at max_step."""
  inv_base = tf.exp(tf.log(min_value) / float(max_step))
  if step is None:
    step = tf.train.get_global_step()
  if step is None:
    return 1.0
  step = common.to_float(step)
  return inv_base**tf.maximum(float(max_step) - step, 0.0)


def inverse_lin_decay(max_step, min_value=0.01, step=None):
  """Inverse-decay linearly from min_value to 1.0 reached at max_step."""
  if step is None:
    step = tf.train.get_global_step()
  if step is None:
    return 1.0
  step = common.to_float(step)
  progress = tf.minimum(step / float(max_step), 1.0)
  return progress * (1.0 - min_value) + min_value


def inject_additional_input(layer, inputs, name, mode="concat"):
  """Injects the additional input into the layer.

  Args:
    layer: layer that the input should be injected to.
    inputs: inputs to be injected.
    name: TF scope name.
    mode: how the infor should be added to the layer: "concat" concats as
      additional channels. "multiplicative" broadcasts inputs and multiply them
      to the channels. "multi_additive" broadcasts inputs and multiply and add
      to the channels.

  Returns:
    updated layer.

  Raises:
    ValueError: in case of unknown mode.
  """
  inputs = common.to_float(inputs)
  layer_shape = common.shape_list(layer)
  input_shape = common.shape_list(inputs)
  zeros_mask = tf.zeros(layer_shape, dtype=tf.float32)
  if mode == "concat":
    emb = encode_to_shape(inputs, layer_shape, name)
    layer = tf.concat(values=[layer, emb], axis=-1)
  elif mode == "multiplicative":
    filters = layer_shape[-1]
    input_reshaped = tf.reshape(inputs, [-1, 1, 1, input_shape[-1]])
    input_mask = tf.layers.dense(input_reshaped, filters, name=name)
    input_broad = input_mask + zeros_mask
    layer *= input_broad
  elif mode == "multi_additive":
    filters = layer_shape[-1]
    input_reshaped = tf.reshape(inputs, [-1, 1, 1, input_shape[-1]])
    input_mul = tf.layers.dense(input_reshaped, filters, name=name + "_mul")
    layer *= tf.nn.sigmoid(input_mul)
    input_add = tf.layers.dense(input_reshaped, filters, name=name + "_add")
    layer += input_add
  else:
    raise ValueError("Unknown injection mode: %s" % mode)

  return layer


def scheduled_sample_prob(ground_truth_x, generated_x, batch_size,
                          scheduled_sample_var):
  """Probability based scheduled sampling.

  Args:
    ground_truth_x: tensor of ground-truth data points.
    generated_x: tensor of generated data points.
    batch_size: batch size
    scheduled_sample_var: probability of choosing from ground_truth.

  Returns:
    New batch with randomly selected data points.
  """
  probability_threshold = scheduled_sample_var
  probability_of_generated = tf.random_uniform([batch_size])
  return tf.where(probability_of_generated > probability_threshold, generated_x,
                  ground_truth_x)


def dna_transformation(prev_image, dna_input, dna_kernel_size, relu_shift):
  """Apply dynamic neural advection to previous image.

  Args:
    prev_image: previous image to be transformed.
    dna_input: hidden lyaer to be used for computing DNA transformation.
    dna_kernel_size: dna kernel size.
    relu_shift: shift for ReLU function.

  Returns:
    List of images transformed by the predicted CDNA kernels.
  """
  # Construct translated images.
  prev_image_pad = tf.pad(prev_image, [[0, 0], [2, 2], [2, 2], [0, 0]])
  image_height = int(prev_image.get_shape()[1])
  image_width = int(prev_image.get_shape()[2])

  inputs = []
  for xkern in range(dna_kernel_size):
    for ykern in range(dna_kernel_size):
      inputs.append(
          tf.expand_dims(
              tf.slice(prev_image_pad, [0, xkern, ykern, 0],
                       [-1, image_height, image_width, -1]), [3]))
  inputs = tf.concat(axis=3, values=inputs)

  # Normalize channels to 1.
  kernel = tf.nn.relu(dna_input - relu_shift) + relu_shift
  kernel = tf.expand_dims(kernel / tf.reduce_sum(kernel, [3], keep_dims=True),
                          [4])
  return tf.reduce_sum(kernel * inputs, [3], keep_dims=False)


def cdna_transformation(prev_image, cdna_input, num_masks, color_channels,
                        dna_kernel_size, relu_shift):
  """Apply convolutional dynamic neural advection to previous image.

  Args:
    prev_image: previous image to be transformed.
    cdna_input: hidden lyaer to be used for computing CDNA kernels.
    num_masks: number of masks and hence the number of CDNA transformations.
    color_channels: the number of color channels in the images.
    dna_kernel_size: dna kernel size.
    relu_shift: shift for ReLU function.

  Returns:
    List of images transformed by the predicted CDNA kernels.
  """
  batch_size = tf.shape(cdna_input)[0]
  height = int(prev_image.get_shape()[1])
  width = int(prev_image.get_shape()[2])

  # Predict kernels using linear function of last hidden layer.
  cdna_kerns = tfl.dense(
      cdna_input,
      dna_kernel_size * dna_kernel_size * num_masks,
      name="cdna_params",
      activation=None)

  # Reshape and normalize.
  cdna_kerns = tf.reshape(
      cdna_kerns, [batch_size, dna_kernel_size, dna_kernel_size, 1, num_masks])
  cdna_kerns = (tf.nn.relu(cdna_kerns - relu_shift) + relu_shift)
  norm_factor = tf.reduce_sum(cdna_kerns, [1, 2, 3], keep_dims=True)
  cdna_kerns /= norm_factor

  # Treat the color channel dimension as the batch dimension since the same
  # transformation is applied to each color channel.
  # Treat the batch dimension as the channel dimension so that
  # depthwise_conv2d can apply a different transformation to each sample.
  cdna_kerns = tf.transpose(cdna_kerns, [1, 2, 0, 4, 3])
  cdna_kerns = tf.reshape(
      cdna_kerns, [dna_kernel_size, dna_kernel_size, batch_size, num_masks])
  # Swap the batch and channel dimensions.
  prev_image = tf.transpose(prev_image, [3, 1, 2, 0])

  # Transform image.
  transformed = tf.nn.depthwise_conv2d(prev_image, cdna_kerns, [1, 1, 1, 1],
                                       "SAME")

  # Transpose the dimensions to where they belong.
  transformed = tf.reshape(
      transformed, [color_channels, height, width, batch_size, num_masks])
  transformed = tf.transpose(transformed, [3, 1, 2, 0, 4])
  transformed = tf.unstack(transformed, axis=-1)
  return transformed


def tile_and_concat(image, latent, concat_latent=True):
  """Tile latent and concatenate to image across depth.

  Args:
    image: 4-D Tensor, (batch_size X height X width X channels)
    latent: 2-D Tensor, (batch_size X latent_dims)
    concat_latent: If set to False, the image is returned as is.

  Returns:
    concat_latent: 4-D Tensor, (batch_size X height X width X channels+1)
      latent tiled and concatenated to the image across the channels.
  """
  if not concat_latent:
    return image
  image_shape = common.shape_list(image)
  latent_shape = common.shape_list(latent)
  height, width = image_shape[1], image_shape[2]
  latent_dims = latent_shape[1]
  height_multiples = height // latent_dims
  pad = height - (height_multiples * latent_dims)
  latent = tf.reshape(latent, (-1, latent_dims, 1, 1))
  latent = tf.tile(latent, (1, height_multiples, width, 1))
  latent = tf.pad(latent, [[0, 0], [pad // 2, pad // 2], [0, 0], [0, 0]])
  return tf.concat([image, latent], axis=-1)


def get_gaussian_tensor(mean, log_var):
  z = tf.random_normal(tf.shape(mean), 0, 1, dtype=tf.float32)
  z = mean + tf.exp(log_var / 2.0) * z
  return z


def conv_latent_tower(images,
                      time_axis,
                      latent_channels=1,
                      min_logvar=-5,
                      is_training=False,
                      random_latent=False,
                      tiny_mode=False,
                      small_mode=False):
  """Builds convolutional latent tower for stochastic model.

  At training time this tower generates a latent distribution (mean and std)
  conditioned on the entire video. This latent variable will be fed to the
  main tower as an extra variable to be used for future frames prediction.
  At inference time, the tower is disabled and only returns latents sampled
  from N(0,1).
  If the multi_latent flag is on, a different latent for every timestep would
  be generated.

  Args:
    images: tensor of ground truth image sequences
    time_axis: the time axis  in images tensor
    latent_channels: number of latent channels
    min_logvar: minimum value for log_var
    is_training: whether or not it is training mode
    random_latent: whether or not generate random latents
    tiny_mode: whether or not it is tiny_mode. tiny_mode sets the number of conv
      channels to 1 at each layer. useful for testing the integration tests.
    small_mode: whether or not it is small_mode. small mode is the same model
      with less conv and lstm layers and also lower number of channels. suitable
      for videos with less complexity and testing.

  Returns:
    latent_mean: predicted latent mean
    latent_logvar: predicted latent log variance
  """
  conv_size = common.tinyify([32, 64, 64], tiny_mode, small_mode)
  with tf.variable_scope("latent", reuse=tf.AUTO_REUSE):
    images = common.to_float(images)
    images = tf.unstack(images, axis=time_axis)
    images = tf.concat(images, axis=3)

    x = images
    x = make_even_size(x)
    x = tfl.conv2d(
        x,
        conv_size[0], [3, 3],
        strides=(2, 2),
        padding="SAME",
        activation=tf.nn.relu,
        name="latent_conv1")
    x = tfcl.layer_norm(x)
    if not small_mode:
      x = tfl.conv2d(
          x,
          conv_size[1], [3, 3],
          strides=(2, 2),
          padding="SAME",
          activation=tf.nn.relu,
          name="latent_conv2")
      x = tfcl.layer_norm(x)
    x = tfl.conv2d(
        x,
        conv_size[2], [3, 3],
        strides=(1, 1),
        padding="SAME",
        activation=tf.nn.relu,
        name="latent_conv3")
    x = tfcl.layer_norm(x)

    nc = latent_channels
    mean = tfl.conv2d(
        x,
        nc, [3, 3],
        strides=(2, 2),
        padding="SAME",
        activation=None,
        name="latent_mean")
    logv = tfl.conv2d(
        x,
        nc, [3, 3],
        strides=(2, 2),
        padding="SAME",
        activation=tf.nn.relu,
        name="latent_std")
    logvar = logv + min_logvar

    # No latent tower at inference time, just standard gaussian.
    if not is_training:
      return tf.zeros_like(mean), tf.zeros_like(logvar)

    # No latent in the first phase
    ret_mean, ret_logvar = tf.cond(
        random_latent, lambda: (tf.zeros_like(mean), tf.zeros_like(logvar)),
        lambda: (mean, logvar))

  return ret_mean, ret_logvar


def beta_schedule(schedule, global_step, final_beta, decay_start, decay_end):
  """Get KL multiplier (beta) based on the schedule."""
  if decay_start > decay_end:
    raise ValueError("decay_end is smaller than decay_end.")

  # Since some of the TF schedules do not support incrementing a value,
  # in all of the schedules, we anneal the beta from final_beta to zero
  # and then reverse it at the bottom.
  if schedule == "constant":
    decayed_value = 0.0
  elif schedule == "linear":
    decayed_value = tf.train.polynomial_decay(
        learning_rate=final_beta,
        global_step=global_step - decay_start,
        decay_steps=decay_end - decay_start,
        end_learning_rate=0.0)
  elif schedule == "noisy_linear_cosine_decay":
    decayed_value = tf.train.noisy_linear_cosine_decay(
        learning_rate=final_beta,
        global_step=global_step - decay_start,
        decay_steps=decay_end - decay_start)
  # TODO(mechcoder): Add log_annealing schedule.
  else:
    raise ValueError("Unknown beta schedule.")

  increased_value = final_beta - decayed_value
  increased_value = tf.maximum(0.0, increased_value)

  beta = tf.case(
      pred_fn_pairs=[(tf.less(global_step, decay_start), lambda: 0.0),
                     (tf.greater(global_step, decay_end), lambda: final_beta)],
      default=lambda: increased_value)
  return beta


def kl_divergence(mu, log_var, mu_p=0.0, log_var_p=0.0):
  """KL divergence of diagonal gaussian N(mu,exp(log_var)) and N(0,1).

  Args:
    mu: mu parameter of the distribution.
    log_var: log(var) parameter of the distribution.
    mu_p: optional mu from a learned prior distribution
    log_var_p: optional log(var) from a learned prior distribution

  Returns:
    the KL loss.
  """

  batch_size = common.shape_list(mu)[0]
  prior_distribution = tfp.distributions.Normal(
      mu_p, tf.exp(tf.multiply(0.5, log_var_p)))
  posterior_distribution = tfp.distributions.Normal(
      mu, tf.exp(tf.multiply(0.5, log_var)))
  kld = tfp.distributions.kl_divergence(posterior_distribution,
                                        prior_distribution)
  return tf.reduce_sum(kld) / common.to_float(batch_size)


def make_even_size(x):
  """Pad x to be even-sized on axis 1 and 2, but only if necessary."""
  x_shape = x.get_shape().as_list()
  assert len(x_shape) > 2, "Only 3+-dimensional tensors supported."
  shape = [dim if dim is not None else -1 for dim in x_shape]
  new_shape = x_shape  # To make sure constant shapes remain constant.
  if x_shape[1] is not None:
    new_shape[1] = 2 * int(math.ceil(x_shape[1] * 0.5))
  if x_shape[2] is not None:
    new_shape[2] = 2 * int(math.ceil(x_shape[2] * 0.5))
  if shape[1] % 2 == 0 and shape[2] % 2 == 0:
    return x
  if shape[1] % 2 == 0:
    x, _ = pad_to_same_length(x, x, final_length_divisible_by=2, axis=2)
    x.set_shape(new_shape)
    return x
  if shape[2] % 2 == 0:
    x, _ = pad_to_same_length(x, x, final_length_divisible_by=2, axis=1)
    x.set_shape(new_shape)
    return x
  x, _ = pad_to_same_length(x, x, final_length_divisible_by=2, axis=1)
  x, _ = pad_to_same_length(x, x, final_length_divisible_by=2, axis=2)
  x.set_shape(new_shape)
  return x


def pad_to_same_length(x, y, final_length_divisible_by=1, axis=1):
  """Pad tensors x and y on axis 1 so that they have the same length."""
  if axis not in [1, 2]:
    raise ValueError("Only axis=1 and axis=2 supported for now.")
  with tf.name_scope("pad_to_same_length", values=[x, y]):
    x_length = common.shape_list(x)[axis]
    y_length = common.shape_list(y)[axis]
    if (isinstance(x_length, int) and isinstance(y_length, int) and
        x_length == y_length and final_length_divisible_by == 1):
      return x, y
    max_length = tf.maximum(x_length, y_length)
    if final_length_divisible_by > 1:
      # Find the nearest larger-or-equal integer divisible by given number.
      max_length += final_length_divisible_by - 1
      max_length //= final_length_divisible_by
      max_length *= final_length_divisible_by
    length_diff1 = max_length - x_length
    length_diff2 = max_length - y_length

    def padding_list(length_diff, arg):
      if axis == 1:
        return [[[0, 0], [0, length_diff]],
                tf.zeros([tf.rank(arg) - 2, 2], dtype=tf.int32)]
      return [[[0, 0], [0, 0], [0, length_diff]],
              tf.zeros([tf.rank(arg) - 3, 2], dtype=tf.int32)]

    paddings1 = tf.concat(padding_list(length_diff1, x), axis=0)
    paddings2 = tf.concat(padding_list(length_diff2, y), axis=0)
    res_x = tf.pad(x, paddings1)
    res_y = tf.pad(y, paddings2)
    # Static shapes are the same except for axis=1.
    x_shape = x.shape.as_list()
    x_shape[axis] = None
    res_x.set_shape(x_shape)
    y_shape = y.shape.as_list()
    y_shape[axis] = None
    res_y.set_shape(y_shape)
    return res_x, res_y
