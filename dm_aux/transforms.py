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
"""Utilities for audio signal processing."""

import chex
import jax.numpy as jnp


def mu_law(wav: chex.Array,
           mu: int = 255,
           quantize: bool = True) -> chex.Array:
  """mu-law companding with optional quantization to `log2(mu + 1)` bits.

  https://en.wikipedia.org/wiki/%CE%9C-law_algorithm

  Args:
    wav: Input wav signal, whose values should be in the range of [-1, +1].
    mu: The compression number. Default to 255.
    quantize: Whether to quantize the compressed values to `mu + 1` integer
      numbers.

  Returns:
    mu-law compressed wav.

  Raises:
    ValueError if `mu` is not positive.
  """
  if mu <= 0:
    raise ValueError(f'Parameter mu should be positive, got {mu}.')
  mu_wav = jnp.sign(wav) * jnp.log(1 + mu * abs(wav)) / jnp.log(1 + mu)
  # Clipping the `mu_wav` to avoid numerical inaccuracy on hardware accerlators.
  mu_wav = jnp.clip(mu_wav, -1.0, 1.0)
  if not quantize:
    return mu_wav
  bins = jnp.linspace(-1, 1, mu + 1, endpoint=True)
  q_wav = jnp.digitize(mu_wav, bins=bins, right=True) - (mu + 1) // 2
  return q_wav.astype(jnp.int32)


def inv_mu_law(compressed_wav: chex.Array,
               mu: int = 255,
               quantize: bool = True) -> chex.Array:
  """mu-law expansion.

  https://en.wikipedia.org/wiki/%CE%9C-law_algorithm

  Args:
    compressed_wav: Input compressed signal.
    mu: The compression number. Default to 255.
    quantize: Whether `compressed_wav` is `log2(mu + 1)`-bit quantized.

  Returns:
    Mu-law expanded version of `compressed_wav` in [-1, +1].

  Raises:
    ValueError if `mu` is not positive.
  """
  if mu <= 0:
    raise ValueError(f'Parameter mu should be positive, getting {mu}.')
  compressed_wav = compressed_wav.astype(jnp.float32)
  mu_wav = compressed_wav * 2.0 / (mu + 1) if quantize else compressed_wav
  return jnp.sign(mu_wav) / mu * (jnp.power(mu + 1, jnp.abs(mu_wav)) - 1)


def power_to_db(power: chex.Array,
                ref: float = 1.0,
                amin: float = 1e-10) -> chex.Array:
  """Converts a power spectrogram to decibel (dB), i.e. `10 * log10(power/ref)`.

  Args:
    power: Input power.
    ref: The reference value to scale the input.
    amin: The minimum value for `power` and/or `ref`.

  Returns:
    Input `power` in dB.
  """
  # Stable version of 10 * log10(power / ref_value)
  log_power = 10.0 * jnp.log10(jnp.maximum(amin, power))
  log_power -= 10.0 * jnp.log10(jnp.maximum(amin, ref))
  return log_power


def db_to_power(log_power: chex.Array, ref: float = 1.0) -> chex.Array:
  """Converts a spectrogram in dB to its power form.

  Equivalent to `ref*10**(log_power/10)`.

  Args:
    log_power: Input power spectrogram in dB.
    ref: The reference value to scale the output.

  Returns:
    Power spectrogram.
  """
  return ref * jnp.power(10.0, 0.1 * log_power)


def amplitude_to_db(amplitude: chex.Array,
                    ref: float = 1.0,
                    amin: float = 1e-5) -> chex.Array:
  """Converts an amplitude spectrogram to decibel (dB).

  Equivalent to `power_to_db(amplitude**2)`.

  Args:
    amplitude: Input amplitude spectrogram.
    ref: The reference value to scale the input.
    amin: The minimum value for `amplitude` and/or `ref`.

  Returns:
    Input `amplitude` in dB.
  """
  return power_to_db(
      jnp.square(amplitude), ref=ref**2, amin=amin**2)


def db_to_amplitude(log_power: chex.Array, ref: float = 1.0) -> chex.Array:
  """Converts a spectrogram in dB to an amplitude spectrogram.

  Equivalent to `power_to_db(x) ** 0.5`.

  Args:
    log_power: Input power spectrogram in dB.
    ref: The reference value to scale the output.

  Returns:
    Amplitude spectrogram.
  """
  return jnp.sqrt(db_to_power(log_power, ref=ref**2))
