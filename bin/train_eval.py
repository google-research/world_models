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

import os
from absl import app
from absl import flags
import gin
import tensorflow.compat.v1 as tf

from world_models.loops import train_eval

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", None, "Output directory.")
flags.DEFINE_multi_string(
    "config_path", None,
    "Newline separated list of paths to a world models gin configs.")
flags.DEFINE_multi_string("config_param", None,
                          "Newline separated list of Gin parameter bindings.")
flags.DEFINE_bool("enable_eager", True, "Enable eager execution mode.")
flags.DEFINE_integer("num_virtual_gpus", -1, "If >1, enables virtual gpus.")
flags.DEFINE_boolean("offline_train", False, "Train model on offline data.")
flags.DEFINE_string("offline_train_data_dir", None,
                    "Data dir to be used for offline training.")

def main(argv):
  del argv  # Unused
  if FLAGS.enable_eager:
    tf.enable_eager_execution()
  tf.config.set_soft_device_placement(True)

  config_params = FLAGS.config_param or []
  config_params += train_eval.get_gin_override_params(FLAGS.output_dir)
  base_config_path = os.path.dirname(FLAGS.config_path[0])
  gin.add_config_file_search_path(base_config_path)
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

  train_eval.train_eval_loop(
      offline_train=FLAGS.offline_train,
      offline_train_data_dir=FLAGS.offline_train_data_dir)


if __name__ == "__main__":
  flags.mark_flags_as_required(["output_dir"])
  flags.mark_flags_as_required(["config_path"])

  app.run(main)

