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
"""End to end test suite."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import gin
import os
import tensorflow.compat.v1 as tf
from world_models.loops import train_eval


class E2ETest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    tf.enable_eager_execution()
    super(E2ETest, cls).setUpClass()

  @parameterized.parameters(
      'configs/tests/planet_cem_cheetah.gin',
      'configs/tests/planet_cem_atari.gin',
      'configs/tests/sv2p_tfcem_cheetah.gin',
      'configs/tests/sv2p_cem_atari.gin',
      'configs/tests/jax_pure_reward_cem_cheetah.gin',
  )
  def testConfig(self, config_path):
    tmp_dir = self.create_tempdir()
    config_params = train_eval.get_gin_override_params(tmp_dir)
    test_srcdir = absltest.get_default_test_srcdir()
    config_path = os.path.join(test_srcdir, config_path)
    gin.parse_config_files_and_bindings([config_path], config_params)
    train_eval.train_eval_loop()


if __name__ == '__main__':
  absltest.main()
