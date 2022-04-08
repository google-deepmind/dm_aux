# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Waveform and spectrogram augmentation functions.

Waveform augmentations:
  - waveform_masking
  - random_polarity_flipping
  - preemphasis

Spectrogram augmentations:
  - spec_augment
  - freq_jitter
  - time_stretch
  - random_time_warp

Augmentations that can be used for both waveforms and spectrograms:
  - additive_gaussian
  - audio_mixing
  - time_jitter
"""

import functools
from typing import Optional

import chex
from dm_aux import spectral
import jax
import jax.numpy as jnp
import scipy.signal


################################################################################
# Waveform augmentations
################################################################################


def waveform_masking(
    key: chex.PRNGKey,
    waveform: chex.Array,
    max_stripe_length: int,
    num_stripes: int) -> chex.Array:
  """Randomly masks stripes along the time dimension of a waveform.

  Args:
    key: random key.
    waveform: input waveform to be augmented of shape [batch_size, time_steps].
    max_stripe_length: the length of each mask stripe.
    num_stripes: the number of stripes.

  Returns:
    Augmented waveform.
  """
  return _drop_stripes(key, waveform, max_stripe_length, num_stripes, axis=1)


def random_polarity_flipping(
    key: chex.PRNGKey,
    waveform: chex.Array,
    flip_prob: float = 0.5) -> chex.Array:
  """Randomly flips the polarity of the `waveform`.

  Args:
    key: random key.
    waveform: input waveform of shape [batch_size, ...].
    flip_prob: the probability of flipping the waveform.

  Returns:
    Augmented waveform.
  """
  batch_size = waveform.shape[0]
  num_dims = len(waveform.shape)
  shape = [batch_size] + [1] * (num_dims - 1)
  sign = (jax.random.uniform(key, shape=shape) > flip_prob).astype(
      waveform.dtype)
  return (sign * 2 - 1) * waveform


def preemphasis(waveform: chex.Array, coef: float = 0.97) -> chex.Array:
  """Scales up the high frequency components in the `waveform`.

  Args:
    waveform: Input waveform of shape [..., time_steps].
    coef: Pre-emphasis coefficient.

  Returns:
    Pre-emphasized waveform.
  """
  return jnp.concatenate([
      waveform[..., :1], waveform[..., 1:] - coef * waveform[..., :-1],
  ], axis=-1)


################################################################################
# Spectrogram augmentations
################################################################################


def spec_augment(
    key: chex.PRNGKey,
    spectrogram: chex.Array,
    max_time_stripe_length: int,
    num_time_stripes: int,
    max_freq_stripe_length: int,
    num_freq_stripes: int) -> chex.Array:
  """Randomly applies the time and frequency mask stripes to a spectrogram.

  Aka, SpecAugment:
    Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D.
    and Le, Q.V., 2019. Specaugment: A simple data augmentation method
    for automatic speech recognition. arXiv preprint arXiv:1904.08779.

  Args:
    key: random key.
    spectrogram: spectrogram to be augmented of shape [batch_size, time_steps,
      num_bins].
    max_time_stripe_length: the length of each mask stripe on the time
      dimension.
    num_time_stripes: the number of time stripes.
    max_freq_stripe_length: the length of each mask stripe on the frequency
      dimension.
    num_freq_stripes: the number of frequency stripes.

  Returns:
    Augmented spectrogram.
  """
  key1, key2 = jax.random.split(key)
  spectrogram = _drop_stripes(
      key1, spectrogram, max_time_stripe_length, num_time_stripes, axis=1)
  return _drop_stripes(
      key2, spectrogram, max_freq_stripe_length, num_freq_stripes, axis=2)


def freq_jitter(
    key: chex.PRNGKey,
    spectrogram: chex.Array,
    max_amount: int,
    pad_mode: Optional[str] = 'constant') -> chex.Array:
  """Randomly jitter the `spectrogram` along the frequency dimension.

  Args:
    key: random key.
    spectrogram: input spectrogram of shape [batch_size, time_steps,
      num_freq_bins, ...].
    max_amount: max steps of freq jittering.
    pad_mode: the mode of `jax.numpy.pad` method. Used to define the values of
      the padded part.

  Returns:
    Augmented spectrogram.
  """
  return jax.vmap(functools.partial(
      _jitter, max_amount=max_amount, axis=1, pad_mode=pad_mode))(
          key=jax.random.split(key, num=spectrogram.shape[0]),
          audio=spectrogram)


def time_stretch(
    spectrogram: chex.Array,
    fixed_rate: float = 1.0,) -> chex.Array:
  """Stretches spectrogram in time without changing the pitch.

  Args:
    spectrogram: input complex spectrogram.
    fixed_rate: rate of time stretch. Default to 1 which means no change.

  Returns:
    Stretched complex spectrogram.
  """
  return spectral.phase_vocoder(spectrogram, fixed_rate)


def random_time_warp(
    key: chex.PRNGKey,
    spectrogram: chex.Array,
    sigma: float = 1.0,
    scale: float = 1.0,
    taper_off: bool = True) -> chex.Array:
  """Randomly warps a spectrogram along time with a Gaussian displacement field.

  Args:
    key: random key.
    spectrogram: input spectrogram of shape [batch_size, time_steps,
      num_freq_bins].
    sigma: the standard deviation of the Gaussian displacement field.
    scale: the scaling constant for the displacement field.
    taper_off: whether to taper off the displacement field.

  Returns:
    Randomly warped spectrogram.
  """
  # Displacement field.
  length = spectrogram.shape[1]
  field = jax.random.uniform(key, (length,), minval=-1.0, maxval=1.0)
  field = jnp.convolve(field, _gaussian_kernel(sigma), mode='same') * scale
  if taper_off:
    field *= scipy.signal.get_window('hann', length, fftbins=True)

  def _warp(x, field):
    """Warps a one-dimensional signal with a given displacement field."""
    assert x.shape == field.shape
    length = x.shape[0]
    coords = jnp.arange(length) + field
    return jax.scipy.ndimage.map_coordinates(
        x, coords[None], order=1, mode='constant')

  # Vmap the warping along the batch and frequency dimension.
  return jax.vmap(jax.vmap(
      functools.partial(_warp, field=field), in_axes=1, out_axes=1))(
          spectrogram)


################################################################################
# Augmentations can be used for both waveforms and spectrograms
################################################################################


def additive_gaussian(
    key: chex.PRNGKey,
    audio: chex.Array,
    noise_level_in_db: chex.Numeric,
    ) -> chex.Array:
  """Augments the audio with additive white Gaussian noise.

  Args:
    key: random key.
    audio: input waveform to be augmented.
    noise_level_in_db: the standard deviation of the noise in dB, normalized to
      the maximum value in audio.

  Returns:
    Augmented waveform.
  """
  noise = jax.random.normal(key, shape=audio.shape)
  noise_level = 10. ** (noise_level_in_db / 20.) * jnp.abs(audio).max()
  return audio + noise_level * noise


def audio_mixing(
    key: chex.PRNGKey,
    audio: chex.Array,
    mix_lambda: chex.Array) -> chex.Array:
  r"""Randomly mixes two audio samples from the batch.

  Given two samples x1 and x2, the augmented version of x1 is:

    \bar{x1} = \lambda * x1 + (1 - \lambda) * x2,

  where lambda is a random number in [0, 1].

  Originally in:
    H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, “mixup: Beyond
    empirical risk minimization,” in International Conference on Learning
    Representations (ICLR), 2018.

  Args:
    key: random key.
    audio: input to be augmented, whose leading dimension is the batch size.
      It returns audio if the batch size is 1.
    mix_lambda: the mixing ratio lambda. It can be either a scalar or a vector
      of length equal to the batch size of `audio`.

  Returns:
    Augmented audio.
  """
  batch_size = audio.shape[0]
  num_dims = len(audio.shape)
  if batch_size == 1:
    return audio
  assert len(mix_lambda.shape) == 1, (
      'mix_lambda should be a scalar or a vector, getting a '
      f'{len(mix_lambda.shape)}-d array.')
  assert len(mix_lambda) == batch_size, (
      f'Length of mix_lambda ({len(mix_lambda)}) is not equal to the batch '
      f'size ({batch_size})')
  mix_lambda = jnp.reshape(mix_lambda, [batch_size] + [1] * (num_dims - 1))
  random_inds = jax.random.permutation(key, jnp.arange(batch_size))
  return audio * mix_lambda + audio[random_inds, ...] * (1.0 - mix_lambda)


def time_jitter(
    key: chex.PRNGKey,
    audio: chex.Array,
    max_amount: int,
    pad_mode: Optional[str] = 'constant') -> chex.Array:
  """Randomly jitters the `audio` along the time dimension.

  Args:
    key: random key.
    audio: input audio of shape [batch_size, time_steps, ...].
    max_amount: max steps of time jittering.
    pad_mode: the mode of `jax.numpy.pad` method. Used to define the values of
      the padded part.

  Returns:
    Augmented audio.
  """
  time_jitter_fn = functools.partial(
      _jitter, max_amount=max_amount, axis=0, pad_mode=pad_mode)
  return jax.vmap(time_jitter_fn)(
      key=jax.random.split(key, num=audio.shape[0]), audio=audio)


def _drop_stripes(
    key: chex.PRNGKey,
    x: chex.Array,
    max_stripe_length: int,
    num_stripes: int,
    axis: int) -> chex.Array:
  """Randomly masks stripes along the `axis` dimension.

  For example, below shows stripes along `axis=1`, with `max_stripe_length=4`
  and `num_stripes=1`:

    [[1, 1, 0, 0, 0, 1, 1, 1, 1,],
     [1, 1, 1, 0, 1, 1, 1, 1, 1,],
     [1, 1, 1, 1, 1, 1, 0, 0, 1,],
     [0, 0, 0, 0, 1, 1, 1, 1, 1,]]

  Args:
    key: random key.
    x: input to be augmented, whose leading dimension is the batch dimension.
    max_stripe_length: the length of each mask stripe.
    num_stripes: the number of stripes.
    axis: the axis along which masks will be applied.

  Returns:
    Augmented x.
  """
  batch_size = x.shape[0]
  max_length = x.shape[axis]
  num_dims = len(x.shape)

  def _mask(key: chex.PRNGKey) -> chex.Array:
    key1, key2 = jax.random.split(key)
    shape = [batch_size] + [1] * (num_dims - 1)
    stripe_length = jax.random.randint(
        key1, shape=shape, minval=0, maxval=max_stripe_length)
    start = jax.random.randint(
        key2, shape=shape, minval=0, maxval=max_length - stripe_length)
    mask_shape = [1] * num_dims
    mask_shape[axis] = max_length
    mask = jnp.repeat(
        jnp.arange(max_length).reshape(mask_shape), batch_size, axis=0)
    return jnp.logical_not((mask > start) * (mask < start + stripe_length))

  for _ in range(num_stripes):
    key, subkey = jax.random.split(key)
    x *= _mask(subkey)
  return x


def _jitter(
    key: chex.PRNGKey,
    audio: chex.Array,
    max_amount: int,
    axis: int,
    pad_mode: Optional[str] = 'constant') -> chex.Array:
  """Randomly jitters the `audio` along the `axis` dimension.

  Args:
    key: random key.
    audio: input audio. If the leading dim is batch, it requires a vmap/pmap
      for this method to work.
    max_amount: max steps of the jitter along the `axis`.
    axis: the dimension of `audio` to be jittered.
    pad_mode: the mode of `jax.numpy.pad` method. Used to define the values of
      the padded part.

  Returns:
    Augmented audio.
  """
  num_dims = len(audio.shape)
  pad_shape = [[0, 0]] * num_dims
  pad_shape[axis] = [max_amount, max_amount]
  padded_audio = jnp.pad(audio, pad_shape, mode=pad_mode)
  offset = jax.random.randint(key, shape=(), minval=0, maxval=2*max_amount - 1)
  start = [0] * num_dims
  start[axis] = offset
  return jax.lax.dynamic_slice(padded_audio, start, audio.shape)


def _gaussian_kernel(sigma: float) -> chex.Array:
  """Gaussian kernel."""
  radius = int(4 * sigma + 0.5)  # Heuristic taken from scipy.
  x = jnp.arange(-radius, radius+1)
  phi_x = jnp.exp(-0.5 / (sigma ** 2) * (x ** 2))
  return phi_x / phi_x.sum()
