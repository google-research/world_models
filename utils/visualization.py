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

"""Gif summary utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import subprocess

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2.summary as tfs
from tensorflow.python.ops import summary_op_util


def encode_gif(images, fps):
  """Encodes numpy images into gif string.

  Args:
    images: A 5-D `uint8` `np.array` (or a list of 4-D images) of shape
      `[batch_size, time, height, width, channels]` where `channels` is 1 or 3.
    fps: frames per second of the animation

  Returns:
    The encoded gif string.

  Raises:
    IOError: If the ffmpeg command returns an error.
  """
  ffmpeg = 'ffmpeg'
  h, w, c = images[0].shape
  cmd = [
      ffmpeg, '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-r',
      '%.02f' % fps, '-s',
      '%dx%d' % (w, h), '-pix_fmt', {
          1: 'gray',
          3: 'rgb24'
      }[c], '-i', '-', '-filter_complex',
      '[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse', '-r',
      '%.02f' % fps, '-f', 'gif', '-'
  ]
  proc = subprocess.Popen(
      cmd,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE)
  for image in images:
    proc.stdin.write(image.tostring())
  out, err = proc.communicate()
  if proc.returncode:
    err = '\n'.join([' '.join(cmd), err.decode('utf8')])
    raise IOError(err)
  del proc
  return out


def py_gif_summary(tag, images, max_outputs, fps):
  """Outputs a `Summary` protocol buffer with gif animations.

  Args:
    tag: Name of the summary.
    images: A 5-D `uint8` `np.array` of shape `[batch_size, time, height, width,
      channels]` where `channels` is 1 or 3.
    max_outputs: Max number of batch elements to generate gifs for.
    fps: frames per second of the animation

  Returns:
    The serialized `Summary` protocol buffer.

  Raises:
    ValueError: If `images` is not a 5-D `uint8` array with 1 or 3 channels.
  """
  is_bytes = isinstance(tag, bytes)
  if is_bytes:
    tag = tag.decode('utf-8')
  images = np.asarray(images, dtype=np.uint8)
  if images.ndim != 5:
    raise ValueError('Tensor must be 5-D for gif summary.')
  batch_size, _, height, width, channels = images.shape
  if channels not in (1, 3):
    raise ValueError('Tensors must have 1 or 3 channels for gif summary.')
  summ = tf.Summary()
  num_outputs = min(batch_size, max_outputs)
  for i in range(num_outputs):
    image_summ = tf.Summary.Image()
    image_summ.height = height
    image_summ.width = width
    image_summ.colorspace = channels  # 1: grayscale, 3: RGB
    try:
      image_summ.encoded_image_string = encode_gif(images[i], fps)
    except (IOError, OSError) as e:
      tf.logging.warning(
          'Unable to encode images to a gif string because either ffmpeg is '
          'not installed or ffmpeg returned an error: %s. Falling back to an '
          'image summary of the first frame in the sequence.', e)
      try:
        from PIL import Image  # pylint: disable=g-import-not-at-top
        with io.BytesIO() as output:
          Image.fromarray(images[i][0]).save(output, 'PNG')
          image_summ.encoded_image_string = output.getvalue()
      except Exception:  # pylint: disable=broad-except
        tf.logging.warning(
            'Gif summaries requires ffmpeg or PIL to be installed: %s', e)
        image_summ.encoded_image_string = (''.encode('utf-8')
                                           if is_bytes else '')
    if num_outputs == 1:
      summ_tag = '{}/gif'.format(tag)
    else:
      summ_tag = '{}/gif/{}'.format(tag, i)
    summ.value.add(tag=summ_tag, image=image_summ)
  summ_str = summ.SerializeToString()
  return summ_str


def gif_summary(name, tensor, max_outputs, fps, collections=None, family=None):
  """Outputs a `Summary` protocol buffer with gif animations.

  Args:
    name: Name of the summary.
    tensor: A 5-D `uint8` `Tensor` of shape `[batch_size, time, height, width,
      channels]` where `channels` is 1 or 3.
    max_outputs: Max number of batch elements to generate gifs for.
    fps: frames per second of the animation
    collections: Optional list of tf.GraphKeys.  The collections to add the
      summary to.  Defaults to [tf.GraphKeys.SUMMARIES]
    family: Optional; if provided, used as the prefix of the summary tag name,
      which controls the tab name used for display on Tensorboard.

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
  tensor = tf.convert_to_tensor(tensor)
  if tensor.dtype in (tf.float32, tf.float64):
    tensor = tf.cast(255.0 * tensor, tf.uint8)
  with summary_op_util.summary_scope(
      name, family, values=[tensor]) as (tag, scope):
    val = tf.py_func(
        py_gif_summary, [tag, tensor, max_outputs, fps],
        tf.string,
        stateful=False,
        name=scope)
    summary_op_util.collect(val, collections, [tf.GraphKeys.SUMMARIES])
  return val


def matplot_figure_to_tensor(fig):
  buf = io.BytesIO()
  fig.savefig(buf, format='png')
  buf.seek(0)
  return tf.image.decode_png(buf.getvalue(), channels=4)


def plot_1d_signals(signals, labels, size):
  """Plot a 1d signals and converts into an image tensor."""
  import matplotlib  # pylint: disable=g-import-not-at-top
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

  images = []
  for i in range(size):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for x, l in zip(signals[i], labels):
      ax.plot(x, label=l)
    ax.legend()
    image = matplot_figure_to_tensor(fig)
    plt.close(fig)
    images.append(image)
  stacked = tf.stack(images, axis=0)
  return stacked


def py_plot_1d_signal(name, signals, labels, max_outputs=3, step=None):
  """Visualizes a list of 1d signals.

  Args:
    name: name of the summary.
    signals: a [batch, lines, steps] np.array list of 1d arrays.
    labels: a [lines] list of labels for each signal.
    max_outputs: the maximum number of plots to add to summaries.
    step: an explicit step or None.

  Returns:
    the summary result.
  """
  image = plot_1d_signals(signals, labels, min(max_outputs, signals.shape[0]))
  return tfs.image(name, image, step, max_outputs=max_outputs)


def tf_plot_1d_signal(name, signals, labels, max_outputs=3, step=None):
  """Visualizes a list of 1d signals.

  Args:
    name: name of the summary.
    signals: a [batch, lines, steps] tensor, each line a 1d signal.
    labels: a [lines] list of labels for each signal.
    max_outputs: the maximum number of plots to add to summaries.
    step: an explicit step or None.

  Returns:
    the summary result.
  """
  image = tf.py_function(
      plot_1d_signals,
      (signals, labels, tf.math.minimum(max_outputs, tf.shape(signals)[0])),
      tf.uint8)
  return tfs.image(name, image, step, max_outputs=max_outputs)


def side_by_side_frames(name, tensors):
  """Visualizes frames side by side.

  Args:
    name: name of the summary.
    tensors: a list of video tensors to be merged side by side.

  Returns:
    the summary result.
  """
  x = tf.concat(tensors, axis=3)
  x = tf.concat(tf.unstack(x, axis=1), axis=1)
  return tfs.image(name, x)



