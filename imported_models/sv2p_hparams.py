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

"""Forked from T2T. Waiting to be cleaned up."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import types


def sv2p_hparams():
  """SV2P model hparams."""
  hparams = types.SimpleNamespace()
  hparams.video_num_input_frames = 4
  hparams.video_num_target_frames = 4

  hparams.merged_reward_model = False
  hparams.reward_model_stop_gradient = True
  hparams.reward_prediction_classes = 1

  hparams.loss_reward_multiplier = 1.0
  hparams.loss_extra_multiplier = 1e-3

  hparams.stochastic = False
  hparams.latent_channels = 1
  hparams.latent_min_logvar = -5.0

  hparams.num_masks = 10
  hparams.relu_shift = 1e-12
  hparams.dna_kernel_size = 5

  hparams.scheduled_sampling_iterations = 10000
  return hparams
