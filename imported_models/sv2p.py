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

"""SV2P: Stochastic Variational Video Prediction.

   based on the following paper:
   https://arxiv.org/abs/1710.11252
   by Mohammad Babaeizadeh, Chelsea Finn, Dumitru Erhan,
      Roy H. Campbell and Sergey Levine
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow

from world_models.imported_models import common
from world_models.imported_models import layers
from world_models.imported_models import reward_models
from world_models.utils import visualization
from tensorflow.contrib import layers as contrib_layers

tfl = tf.layers
tfcl = contrib_layers
tfs = tensorflow.compat.v2.summary


class SV2P(object):
  """Stochastic Variational Video Prediction From Basic Model!"""

  def __init__(self, hparams):
    self.hparams = hparams
    if hparams.merged_reward_model:
      self._body = tf.make_template("body", self.__body_merged_model)
    else:
      self._body = tf.make_template("body", self.__body_separate_model)
    self.infer = tf.make_template("infer", self.__infer)
    self.train = tf.make_template("train", self.__train)
    self.loss = tf.make_template("loss", self.__loss)

    self.posterior = tf.make_template("posterior", self.conv_latent_tower)
    self.prior = tf.make_template("prior", self.gaussian_prior)
    self.latent_tow = self.posterior if self.hparams.stochastic else self.prior

  def __train(self, features):
    """Trains the model."""
    self.is_training = True
    frames, actions, rewards = self.extract_features(features)
    latent_mean, latent_logvar = self.latent_tow(features["frames"])
    latent = layers.get_gaussian_tensor(latent_mean, latent_logvar)
    extra_loss = layers.kl_divergence(latent_mean, latent_logvar)
    pred_frames, pred_rewards = self._body(frames, actions, rewards, latent)
    predictions = {
        "frames": tf.stack(pred_frames[:-1], axis=1),
        "rewards": tf.stack(pred_rewards[:-1], axis=1)
    }
    return self.loss(predictions, features, extra_loss), predictions["rewards"]

  def __infer(self, features):
    """Produce predictions from the model by running it."""
    self.is_training = False
    frames, actions, rewards = self.extract_features(features)
    mean, logvar = self.gaussian_prior(features["frames"])
    latent = layers.get_gaussian_tensor(mean, logvar)

    pred_frames, pred_rewards = self._body(frames, actions, rewards, latent)

    extra_predicted_frames = self.hparams.video_num_input_frames
    predictions = {
        "frames": tf.stack(pred_frames[extra_predicted_frames:], axis=1),
        "rewards": tf.stack(pred_rewards[extra_predicted_frames:], axis=1)
    }
    return predictions

  def __loss(self, predictions, features, extra_loss):
    """Calculates the loss."""
    reward_loss = tf.constant(0.0, tf.float32)

    pred = predictions["rewards"]
    loss_func = tf.keras.losses.MSE
    targ = features["rewards"][:, 1:]
    reward_loss += tf.reduce_mean(tfl.flatten(loss_func(targ, pred)), axis=-1)

    pred = predictions["frames"]
    targ = tf.image.convert_image_dtype(features["frames"][:, 1:], tf.float32)
    loss_func = tf.keras.losses.MSE
    frames_loss = tf.reduce_mean(tfl.flatten(loss_func(targ, pred)), axis=-1)

    total_loss = (
        frames_loss + reward_loss * self.hparams.loss_reward_multiplier +
        extra_loss * self.hparams.loss_extra_multiplier)

    tfs.scalar("loss/total", tf.reduce_mean(total_loss))
    tfs.scalar("loss/frames", tf.reduce_mean(frames_loss))
    tfs.scalar("loss/reward", tf.reduce_mean(reward_loss))
    tfs.scalar("loss/extra", tf.reduce_mean(extra_loss))
    visualization.side_by_side_frames("vis/frames", [targ, pred])

    return total_loss

  def extract_features(self, features):
    frames = tf.unstack(features["frames"], axis=1)
    actions = tf.unstack(features["actions"], axis=1)
    rewards = tf.unstack(features["rewards"], axis=1)
    return frames, actions, rewards

  @property
  def trackables(self):
    return {"model": self._body}

  @property
  def video_len(self):
    hp = self.hparams
    return hp.video_num_input_frames + hp.video_num_target_frames

  def get_iteration_num(self):
    return tf.train.get_global_step()

  def reward_prediction(self, mid_outputs):
    """Select reward predictor based on hparams."""
    x = reward_models.reward_prediction_mid(mid_outputs)
    x = tfl.flatten(x)
    x = tfl.dense(
        x,
        self.hparams.reward_prediction_classes,
        activation=None,
        name="reward_map")
    return x

  def upsample(self, x, num_outputs, strides):
    x = tfl.conv2d_transpose(
        x, num_outputs, (3, 3), strides=strides, activation=tf.nn.relu)
    return x[:, 1:, 1:, :]

  def shape_list(self, x):
    return x.shape.as_list()

  def inject_additional_input(self, layer, inputs, name):
    """Injects the additional input into the layer.

    Args:
      layer: layer that the input should be injected to.
      inputs: inputs to be injected.
      name: TF scope name.

    Returns:
      updated layer.

    Raises:
      ValueError: in case of unknown mode.
    """
    inputs = common.to_float(inputs)
    layer_shape = self.shape_list(layer)
    emb = layers.encode_to_shape(inputs, layer_shape, name)
    layer = tf.concat(values=[layer, emb], axis=-1)
    return layer

  def bottom_part_tower(self, input_image, action, reward, latent, lstm_state,
                        lstm_size, conv_size):
    """The bottom part of predictive towers.

    With the current (early) design, the main prediction tower and
    the reward prediction tower share the same arcitecture. TF Scope can be
    adjusted as required to either share or not share the weights between
    the two towers.

    Args:
      input_image: the current image.
      action: the action taken by the agent.
      reward: the previous reward. observed or predicted.
      latent: the latent vector.
      lstm_state: the current internal states of conv lstms.
      lstm_size: the size of lstms.
      conv_size: the size of convolutions.

    Returns:
      - the output of the partial network.
      - intermidate outputs for skip connections.
    """
    lstm_func = layers.conv_lstm_2d
    input_image = layers.make_even_size(input_image)

    layer_id = 0
    enc0 = tfl.conv2d(
        input_image,
        conv_size[0], [5, 5],
        strides=(2, 2),
        activation=tf.nn.relu,
        padding="SAME",
        name="scale1_conv1")
    enc0 = tfcl.layer_norm(enc0, scope="layer_norm1")

    hidden1, lstm_state[layer_id] = lstm_func(
        enc0, lstm_state[layer_id], lstm_size[layer_id], name="state1")
    hidden1 = tfcl.layer_norm(hidden1, scope="layer_norm2")
    layer_id += 1

    hidden2, lstm_state[layer_id] = lstm_func(
        hidden1, lstm_state[layer_id], lstm_size[layer_id], name="state2")
    hidden2 = tfcl.layer_norm(hidden2, scope="layer_norm3")
    hidden2 = layers.make_even_size(hidden2)
    enc1 = tfl.conv2d(
        hidden2,
        hidden2.get_shape()[3], [3, 3],
        strides=(2, 2),
        padding="SAME",
        activation=tf.nn.relu,
        name="conv2")
    layer_id += 1

    hidden3, lstm_state[layer_id] = lstm_func(
        enc1, lstm_state[layer_id], lstm_size[layer_id], name="state3")
    hidden3 = tfcl.layer_norm(hidden3, scope="layer_norm4")
    layer_id += 1

    hidden4, lstm_state[layer_id] = lstm_func(
        hidden3, lstm_state[layer_id], lstm_size[layer_id], name="state4")
    hidden4 = tfcl.layer_norm(hidden4, scope="layer_norm5")
    hidden4 = layers.make_even_size(hidden4)
    enc2 = tfl.conv2d(
        hidden4,
        hidden4.get_shape()[3], [3, 3],
        strides=(2, 2),
        padding="SAME",
        activation=tf.nn.relu,
        name="conv3")
    layer_id += 1

    enc2 = self.inject_additional_input(enc2, action, "action_enc")
    if reward is not None:
      enc2 = self.inject_additional_input(enc2, reward, "reward_enc")
    with tf.control_dependencies([latent]):
      enc2 = tf.concat([enc2, latent], axis=3)

    enc3 = tfl.conv2d(
        enc2,
        hidden4.get_shape()[3], [1, 1],
        strides=(1, 1),
        padding="SAME",
        activation=tf.nn.relu,
        name="conv4")

    hidden5, lstm_state[layer_id] = lstm_func(
        enc3, lstm_state[layer_id], lstm_size[layer_id], name="state5")
    hidden5 = tfcl.layer_norm(hidden5, scope="layer_norm6")
    layer_id += 1
    return hidden5, (enc0, enc1), layer_id

  def construct_predictive_tower(self, input_image, action, reward, lstm_state,
                                 latent):
    """Main prediction tower."""
    lstm_func = layers.conv_lstm_2d
    frame_shape = self.shape_list(input_image)
    _, img_height, img_width, color_channels = frame_shape
    batch_size = tf.shape(input_image)[0]
    # the number of different pixel motion predictions
    # and the number of masks for each of those predictions
    num_masks = self.hparams.num_masks

    lstm_size = [32, 32, 64, 64, 128, 64, 32]
    conv_size = [32]

    with tf.variable_scope("bottom", reuse=tf.AUTO_REUSE):
      hidden5, skips, layer_id = self.bottom_part_tower(input_image, action,
                                                        reward, latent,
                                                        lstm_state, lstm_size,
                                                        conv_size)
    enc0, enc1 = skips

    enc4 = self.upsample(hidden5, self.shape_list(hidden5)[-1], [2, 2])

    enc1_shape = self.shape_list(enc1)
    enc4 = enc4[:, :enc1_shape[1], :enc1_shape[2], :]  # Cut to shape.

    hidden6, lstm_state[layer_id] = lstm_func(
        enc4,
        lstm_state[layer_id],
        lstm_size[5],
        name="state6",
        spatial_dims=enc1_shape[1:-1])  # 16x16
    hidden6 = tfcl.layer_norm(hidden6, scope="layer_norm7")
    # Skip connection.
    hidden6 = tf.concat(axis=3, values=[hidden6, enc1])  # both 16x16
    layer_id += 1

    enc5 = self.upsample(hidden6, self.shape_list(hidden6)[-1], [2, 2])

    enc0_shape = self.shape_list(enc0)
    hidden7, lstm_state[layer_id] = lstm_func(
        enc5,
        lstm_state[layer_id],
        lstm_size[6],
        name="state7",
        spatial_dims=enc0_shape[1:-1])  # 32x32
    hidden7 = tfcl.layer_norm(hidden7, scope="layer_norm8")
    layer_id += 1

    # Skip connection.
    hidden7 = tf.concat(axis=3, values=[hidden7, enc0])  # both 32x32

    enc6 = self.upsample(hidden7, self.shape_list(hidden7)[-1], [2, 2])
    enc6 = tfcl.layer_norm(enc6, scope="layer_norm9")

    enc7 = tfl.conv2d_transpose(
        enc6,
        color_channels, [1, 1],
        strides=(1, 1),
        padding="SAME",
        name="convt4",
        activation=None)
    # This allows the network to also generate one image from scratch,
    # which is useful when regions of the image become unoccluded.
    transformed = [tf.nn.sigmoid(enc7)]

    cdna_input = tfl.flatten(hidden5)
    transformed += layers.cdna_transformation(input_image, cdna_input,
                                              num_masks, int(color_channels),
                                              self.hparams.dna_kernel_size,
                                              self.hparams.relu_shift)

    masks = tfl.conv2d(
        enc6,
        filters=num_masks + 1,
        kernel_size=[1, 1],
        strides=(1, 1),
        name="convt7",
        padding="SAME")
    masks = masks[:, :img_height, :img_width, ...]
    shape = tf.stack(
        [batch_size, int(img_height),
         int(img_width), num_masks + 1])
    masks = tf.reshape(
        tf.nn.softmax(tf.reshape(masks, [-1, num_masks + 1])), shape)
    mask_list = tf.split(axis=3, num_or_size_splits=num_masks + 1, value=masks)
    output = mask_list[0] * input_image
    for layer, mask in zip(transformed, mask_list[1:]):
      output += layer * mask

    mid_outputs = [enc0, enc1, enc4, enc5, enc6]
    return output, lstm_state, mid_outputs

  def gaussian_prior(self, images):
    batch_size = tf.shape(images)[0]
    # TODO(mbz): this only works for 64x64 image size.
    assert images.shape[2] == 64
    assert images.shape[3] == 64
    shape = tf.stack([batch_size, 8, 8, 1])
    return tf.zeros(shape), tf.zeros(shape)

  def conv_latent_tower(self, images):
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

    Returns:
      latent_mean: predicted latent mean
      latent_logvar: predicted latent log variance
    """
    conv_size = [32, 64, 64]
    latent_channels = self.hparams.latent_channels
    min_logvar = self.hparams.latent_min_logvar
    images = tf.concat(tf.unstack(common.to_float(images), axis=1), axis=-1)
    with tf.variable_scope("latent", reuse=tf.AUTO_REUSE):
      x = images
      x = tfl.conv2d(
          x,
          conv_size[0], [3, 3],
          strides=(2, 2),
          padding="SAME",
          activation=tf.nn.relu,
          name="latent_conv1")
      x = tfcl.layer_norm(x)
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
      return mean, logvar

  def scheduled_sample_prob(self, gt_frame, pred_frame):
    prob = tf.math.divide_no_nan(
        common.to_float(self.get_iteration_num()),
        common.to_float(self.hparams.scheduled_sampling_iterations))
    prob = tf.nn.relu(1.0 - prob)
    return tf.cond(
        tf.math.less_equal(tf.random.uniform([]), prob), lambda: gt_frame,
        lambda: pred_frame)

  def build_merged_model(self, all_frames, all_actions, all_rewards, latent):
    """Main video processing function."""
    hparams = self.hparams

    res_frames, res_rewards = [], []
    internal_states = [None] * 7

    pred_image = all_frames[0]
    pred_reward = all_rewards[0]
    for i in range(self.video_len):
      cur_action = all_actions[i]

      done_warm_start = (i >= hparams.video_num_input_frames)
      if done_warm_start:
        if self.is_training:
          cur_frame = self.scheduled_sample_prob(all_frames[i], pred_image)
          cur_reward = self.scheduled_sample_prob(all_rewards[i], pred_reward)
        else:
          cur_frame = pred_image
          cur_reward = pred_reward
      else:
        cur_frame = all_frames[i]
        cur_reward = all_rewards[i]

      with tf.variable_scope("main", reuse=tf.AUTO_REUSE):
        pred_image, internal_states, mids = self.construct_predictive_tower(
            cur_frame, cur_action, cur_reward, internal_states, latent)
        if hparams.reward_model_stop_gradient:
          mids = [tf.stop_gradient(x) for x in mids]
        pred_reward = self.reward_prediction(mids)

        res_frames.append(pred_image)
        res_rewards.append(pred_reward)

    return [res_frames, res_rewards]

  def build_video_model(self, all_frames, all_actions, latent):
    """Main video processing function."""
    hparams = self.hparams

    res_frames = []
    internal_states = [None] * 7

    pred_image = all_frames[0]
    for i in range(self.video_len):
      cur_action = all_actions[i]

      done_warm_start = (i >= hparams.video_num_input_frames)
      if done_warm_start:
        if self.is_training:
          cur_frame = self.scheduled_sample_prob(all_frames[i], pred_image)
        else:
          cur_frame = pred_image
      else:
        cur_frame = all_frames[i]

      with tf.variable_scope("main", reuse=tf.AUTO_REUSE):
        pred_image, internal_states, _ = self.construct_predictive_tower(
            cur_frame, cur_action, None, internal_states, latent)
        res_frames.append(pred_image)
    return res_frames

  def build_reward_model(self, frames, rewards):
    frames = frames[:self.video_len - 1]
    res_rewards = reward_models.reward_prediction_video_conv(
        frames, rewards, self.video_len)
    return tf.unstack(res_rewards, axis=1)

  def preprocess(self, frames, actions, rewards):
    frames = [tf.image.convert_image_dtype(x, tf.float32) for x in frames]
    return frames, actions, rewards

  def __body_merged_model(self, frames, actions, rewards, latent):
    """Body function."""
    frames, actions, rewards = self.preprocess(frames, actions, rewards)
    res_frames, res_rewards = self.build_merged_model(frames, actions, rewards,
                                                      latent)
    return res_frames, res_rewards

  def __body_separate_model(self, frames, actions, rewards, latent):
    """Body function."""
    frames, actions, rewards = self.preprocess(frames, actions, rewards)
    res_frames = self.build_video_model(frames, actions, latent)
    if self.hparams.reward_model_stop_gradient:
      input_frames = [tf.stop_gradient(x) for x in res_frames]
    else:
      input_frames = res_frames
    input_rewards = rewards[:self.hparams.video_num_input_frames]
    res_rewards = self.build_reward_model(input_frames, input_rewards)
    return res_frames, res_rewards
